import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path('data')
OUT = Path('backtest_output')
COST = 0.0008
TOP_DECILE = 0.10
BOT_DECILE = 0.10
TRAIN_BARS = 200
TEST_BARS = 100
STEP = 100

# Cap tiers from enriched universe
meta_csv = ROOT / 'universe_enriched.csv'
cap_map = {}
cap_map_quote = {}
if meta_csv.exists():
    meta = pd.read_csv(meta_csv)
    for _, row in meta.iterrows():
        base = str(row.get('symbol', '')).strip()
        quote = str(row.get('quote', 'USDT')).strip()
        tier = str(row.get('tier', 'all')).strip()
        cap_map[base] = tier
        cap_map_quote[f"{base}_{quote}"] = tier

# Load all available 1d data
frames = []
for p in sorted(ROOT.glob('*_1d_max.csv')):
    raw = p.name.replace('_1d_max.csv', '')
    base = raw.split('_')[0]
    frame = cap_map.get(base, 'all')
    cap_map.setdefault(raw.split('_binance')[0].split('_mexc')[0], frame)
    try:
        df = pd.read_csv(p, parse_dates=['ts'])
        df['ts'] = pd.to_datetime(df['ts'], errors='coerce').dt.tz_localize(None)
        df = df.dropna(subset=['close','volume']).sort_values('ts').reset_index(drop=True)
        df['symbol'] = base
        df['cap_tier'] = cap_map.get(base, 'all')
        frames.append(df[['ts','symbol','cap_tier','close','volume']])
    except Exception:
        continue
if not frames:
    raise RuntimeError('No data found')

data = pd.concat(frames, ignore_index=True)
print(f'Universe: {data["symbol"].nunique()} symbols, {len(data)} rows')
print('Cap tiers:', data['cap_tier'].value_counts().to_dict())

# Build returns panel
prices = data.pivot_table(index='ts', columns='symbol', values='close').sort_index()
caps = data.drop_duplicates('symbol').set_index('symbol')['cap_tier']
rets = prices.pct_change().dropna(how='all')
dates = rets.index

# Long-short cross-sectional momentum with slow lookback
def regime_filter(market_series, mode='sma50'):
    if mode == 'sma50':
        ma = market_series.rolling(50).mean()
        return market_series > ma
    if mode == 'adx25':
        h = market_series.rolling(2).max(); l = market_series.rolling(2).min()
        tr = pd.concat([h-l, (h-market_series.shift(1)).abs(), (l-market_series.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        plus_di = (h.diff().clip(lower=0) / atr.replace(0, np.nan)).ewm(alpha=1/14, min_periods=14, adjust=False).mean() * 100
        minus_di = (-l.diff().clip(upper=0) / atr.replace(0, np.nan)).ewm(alpha=1/14, min_periods=14, adjust=False).mean() * 100
        adx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        return adx > 25
    return pd.Series(True, index=market_series.index)

def backtest_cross_sectional(lookback, regime_mode='sma50', use_vol_trim=True):
    market = rets.mean(axis=1)
    reg = regime_filter(market, regime_mode).reindex(rets.index).fillna(False)
    folds = []
    port_parts = []
    bench_parts = []
    fold_start = 0
    while fold_start + TRAIN_BARS + TEST_BARS <= len(dates):
        test_start = fold_start + TRAIN_BARS
        test_end = min(len(dates), test_start + TEST_BARS)
        test_dates = dates[test_start:test_end]
        if len(test_dates) < 30:
            break
        test_rets = rets.loc[test_dates]
        test_market = market.loc[test_dates]
        test_reg = reg.loc[test_dates]

        # compute slow momentum over lookback ending at test_start-1
        min_look = max(20, lookback // 2)
        look_start = max(0, test_start - lookback)
        hist = prices.iloc[look_start:test_start]
        mom = (hist.iloc[-1] / hist.iloc[0] - 1).dropna()

        # align symbols with caps
        mom_valid = mom[mom.index.isin(caps.index)]
        t = pd.Series(caps[mom_valid.index], index=mom_valid.index).replace('', 'all')

        valid = []
        for tier in ['large','mid','small','tail','all']:
            sub_t = t[t == tier]
            if len(sub_t) < 5:
                continue
            sub_mom = mom_valid[sub_t.index].dropna()
            if len(sub_mom) < 8:
                continue
            sub_mom = sub_mom.sort_values()
            n = len(sub_mom)
            top_n = max(1, int(np.floor(n * TOP_DECILE)))
            bot_n = max(1, int(np.floor(n * BOT_DECILE)))
            longs = sub_mom.iloc[-top_n:].index.tolist()
            shorts = sub_mom.iloc[:bot_n].index.tolist()
            valid.append({
                'tier': tier,
                'longs': longs,
                'shorts': shorts,
            })

        if not valid:
            fold_start += STEP
            continue

        # build daily returns for this fold
        port = pd.Series(0.0, index=test_dates)
        bench = test_market.reindex(test_dates).fillna(0)
        coverage = {'longs': 0, 'shorts': 0, 'days': 0, 'trades': 0}
        for v in valid:
            longs = [s for s in v['longs'] if s in test_rets.columns]
            shorts = [s for s in v['shorts'] if s in test_rets.columns]
            if not longs and not shorts:
                continue
            lr = test_rets[longs].mean(axis=1).fillna(0) if longs else 0.0
            sr = test_rets[shorts].mean(axis=1).fillna(0) if shorts else 0.0
            wt = 1.0 / len(valid)
            port += wt * (lr - sr)
            coverage['longs'] += len(longs)
            coverage['shorts'] += len(shorts)

        # regime filter
        coverage['days'] = int(test_reg.sum())
        port = port.where(test_reg, 0.0)

        # turnover/costs approximated by daily changes
        port_daily_rets = port.shift(1).fillna(0) * test_rets.mean(axis=1).fillna(0)
        # simpler cost model: 2x top/bottom turnover per fold reset
        port_daily_rets -= COST * 0.5

        valid_rets = port_daily_rets.dropna()
        if len(valid_rets) > 0:
            sr = float(np.sqrt(365) * valid_rets.mean() / (valid_rets.std()+1e-9))
            folds.append({'fold': len(folds)+1, 'start': str(test_dates[0].date()), 'end': str(test_dates[-1].date()), 'bars': len(valid_rets), 'sharpe': sr, 'return': float((1+valid_rets).prod()-1)})
            port_parts.append(valid_rets.values)
            bench_parts.append(bench.reindex(valid_rets.index).fillna(0).values)

        fold_start += STEP

    if not port_parts:
        return [], [], []
    s_all = np.concatenate(port_parts)
    b_all = np.concatenate(bench_parts)
    n = min(len(s_all), len(b_all))
    return s_all[:n].tolist(), b_all[:n].tolist(), folds

configs = [
    ('mom60_regime', 60, 'sma50'),
    ('mom90_regime', 90, 'sma50'),
    ('mom60_adx25', 60, 'adx25'),
    ('mom90_adx25', 90, 'adx25'),
]

all_rets = []
bench_rets = None
summary = {}
for name, lb, reg in configs:
    print(f'\nRunning {name} lookback={lb} regime={reg}')
    port, bench, folds = backtest_cross_sectional(lb, reg)
    summary[name] = folds
    if not port:
        print('  no output')
        continue
    all_rets.append(port)
    if bench_rets is None:
        bench_rets = bench
    sr = float(np.sqrt(365) * np.mean(port) / (np.std(port)+1e-9))
    print(f'  folds={len(folds)} overall_sharpe={sr:.3f}')
    for f in folds:
        print(f"    fold {f['fold']:2d} {f['start']}->{f['end']} bars={f['bars']} sharpe={f['sharpe']:.3f} ret={f['return']:.2%}")

# pooled SPA
if all_rets:
    min_len = min(len(r) for r in all_rets + [bench_rets])
    all_rets = [r[:min_len] for r in all_rets]
    bench_rets = bench_rets[:min_len]
    try:
        from spa_hsu_test import stepwise_spa_test
        spa = stepwise_spa_test(all_rets, bench_rets, alpha=0.10)
        print('\nPooled SPA:', spa)
    except Exception as e:
        print('SPA failed:', e)

out = OUT / f'xrank_longshort_{pd.Timestamp.now():%Y%m%d_%H%M%S}.json'
out.parent.mkdir(exist_ok=True)
pd.Series({'configs': configs, 'summary': summary}).to_json(out, orient='split')
print('\nWrote', out)
