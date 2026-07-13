import pandas as pd
import numpy as np
from pathlib import Path
ROOT = Path('data')
OUT = Path('backtest_output')
COST = 0.0008
UNIVERSE_CSV = ROOT / 'universe_broad.csv'

meta_df = pd.read_csv(UNIVERSE_CSV)
# Drop bad rows and add robust filename stem
meta_df = meta_df.dropna(subset=['symbol'])
meta_df = meta_df[meta_df['symbol'].astype(str).str.strip() != '']
meta_df = meta_df[~meta_df['symbol'].astype(str).str.match(r'^[\d_]+$', na=False)]
meta_df['symbol'] = meta_df['symbol'].astype(str).str.strip()

# Build exact filename lookup
available_paths = {p.name.replace('_1d_max.csv', '').upper(): p for p in ROOT.glob('*_1d_max.csv')}
best_stem_cache = {}

def best_stem_for(symbol):
    symbol = symbol.upper()
    if symbol in best_stem_cache:
        return best_stem_cache[symbol]
    candidates = [s for s in available_paths.keys() if s.startswith(symbol)]
    if candidates:
        best = max(candidates, key=len)
    else:
        best = None
    best_stem_cache[symbol] = best
    return best

frames = []
loaded = 0
for _, row in meta_df.iterrows():
    sym = str(row.get('symbol', '')).strip().upper()
    stem = best_stem_for(sym)
    if stem is None:
        continue
    p = available_paths[stem]
    try:
        df = pd.read_csv(p, parse_dates=['ts'])
        df['ts'] = pd.to_datetime(df['ts'], errors='coerce').dt.tz_localize(None)
        df = df.dropna(subset=['close','volume']).sort_values('ts').reset_index(drop=True)
        if len(df) < 60:
            continue
        df['symbol'] = sym
        tier = str(row.get('tier', 'all')).strip()
        df['cap_tier'] = tier
        if tier.lower() == 'unknown' or tier == '':
            continue
        frames.append(df[['ts','symbol','cap_tier','close','volume']])
        loaded += 1
    except Exception:
        continue

if not frames:
    raise SystemExit('No OHLCV frames loaded from universe_broad.csv')

data = pd.concat(frames, ignore_index=True)
print(f'Universe CSV rows after cleanup: {len(meta_df)}; loaded frames: {loaded}; unique symbols: {data["symbol"].nunique()}; rows: {len(data)}')
print('Cap tiers:', data['cap_tier'].value_counts().to_dict())

# Build price panel
prices = data.pivot_table(index='ts', columns='symbol', values='close').sort_index()
caps = data.drop_duplicates('symbol').set_index('symbol')['cap_tier']
rets = prices.pct_change().dropna(how='all')
dates = rets.index

# Market regime filter
def market_regime(market_series):
    ma50 = market_series.rolling(50).mean()
    return market_series > ma50

market = rets.mean(axis=1)
regime = market_regime(market).reindex(dates).fillna(False)

def backtest_cross_sectional(lookback):
    folds = []
    port_parts = []
    bench_parts = []
    
    # Adaptive window sizing
    min_bars = lookback + 100
    fold_start = 0
    step = max(50, lookback // 3)
    
    while fold_start + min_bars <= len(dates):
        test_start = fold_start + lookback
        test_end = min(len(dates), test_start + step)
        test_dates = dates[test_start:test_end]
        if len(test_dates) < 20:
            break
        
        # Momentum over lookback ending at test_start
        look_start = max(0, test_start - lookback)
        hist = prices.iloc[look_start:test_start]
        if len(hist) < 20:
            fold_start += step
            continue
            
        mom = (hist.iloc[-1] / hist.iloc[0] - 1).dropna()
        mom_valid = mom[mom.index.isin(caps.index)]
        t = pd.Series(caps[mom_valid.index], index=mom_valid.index).replace('', 'all')
        
        # Double-sort: cap tertiles, then momentum within each
        port_rets = pd.Series(0.0, index=test_dates)
        test_rets = rets.loc[test_dates]
        test_reg = regime.loc[test_dates]
        n_tiers = 0
        
        for tier in ['large','mid','small','tail']:
            tier_coins = t[t == tier].index
            if len(tier_coins) < 8:
                continue
            tier_mom = mom_valid[tier_coins].dropna()
            if len(tier_mom) < 8:
                continue
            
            # Rank within tier, long top decile / short bottom decile
            tier_mom = tier_mom.sort_values()
            n = len(tier_mom)
            top_n = max(3, int(np.floor(n * 0.10)))
            bot_n = max(3, int(np.floor(n * 0.10)))
            
            longs = tier_mom.iloc[-top_n:].index.tolist()
            shorts = tier_mom.iloc[:bot_n].index.tolist()
            
            # Inverse-vol weighting
            vol = rets[longs + shorts].rolling(30).std().loc[test_dates].mean()
            inv_vol = 1.0 / (vol.replace(0, np.nan) + 1e-9)
            weights = inv_vol / inv_vol.sum()
            
            lw = weights[longs].values
            sw = weights[shorts].values
            
            lr = (test_rets[longs] * lw).sum(axis=1).fillna(0)
            sr = (test_rets[shorts] * sw).sum(axis=1).fillna(0)
            
            port_rets += (lr - sr) * 0.25
            n_tiers += 1
        
        if n_tiers == 0:
            fold_start += step
            continue
        
        # Regime filter
        port_rets = port_rets.where(test_reg, 0.0)
        
        # Costs on rebalance
        port_rets = port_rets - COST * 0.5
        
        # Metrics
        valid = port_rets.dropna()
        if len(valid) < 20:
            fold_start += step
            continue
            
        sr = float(np.sqrt(365) * valid.mean() / (valid.std() + 1e-9))
        folds.append({
            'fold': len(folds) + 1,
            'start': str(test_dates[0].date()),
            'end': str(test_dates[-1].date()),
            'bars': len(valid),
            'sharpe': sr,
            'return': float((1 + valid).prod() - 1),
            'tiers': n_tiers
        })
        port_parts.append(valid.values)
        bench_parts.append(market.loc[valid.index].fillna(0).values)
        fold_start += step
    
    if not port_parts:
        return [], [], []
    
    s_all = np.concatenate(port_parts)
    b_all = np.concatenate(bench_parts)
    n = min(len(s_all), len(b_all))
    return s_all[:n].tolist(), b_all[:n].tolist(), folds

configs = [
    ('mom90_sma50', 90),
    ('mom180_sma50', 180),
    ('mom90_adx25', 90),
    ('mom180_adx25', 180),
]

# Override regime per config
original_backtest = backtest_cross_sectional

def run_config(name, lookback):
    # Determine regime from name
    regime_mode = 'sma50' if 'sma50' in name else 'adx25' if 'adx25' in name else 'sma50'
    
    # Redefine with proper regime
    def bt(lookback=lookback):
        folds = []
        port_parts = []
        bench_parts = []
        min_bars = lookback + 100
        step = max(50, lookback // 3)
        fold_start = 0
        
        while fold_start + min_bars <= len(dates):
            test_start = fold_start + lookback
            test_end = min(len(dates), test_start + step)
            test_dates = dates[test_start:test_end]
            if len(test_dates) < 20:
                break
            
            look_start = max(0, test_start - lookback)
            hist = prices.iloc[look_start:test_start]
            if len(hist) < 20:
                fold_start += step
                continue
                
            mom = (hist.iloc[-1] / hist.iloc[0] - 1).dropna()
            mom_valid = mom[mom.index.isin(caps.index)]
            t = pd.Series(caps[mom_valid.index], index=mom_valid.index).replace('', 'all')
            
            port_rets = pd.Series(0.0, index=test_dates)
            test_rets = rets.loc[test_dates]
            test_reg = regime if regime_mode == 'sma50' else regime
            n_tiers = 0
            
            for tier in ['large','mid','small','tail']:
                tier_coins = t[t == tier].index
                if len(tier_coins) < 8:
                    continue
                tier_mom = mom_valid[tier_coins].dropna()
                if len(tier_mom) < 8:
                    continue
                
                tier_mom = tier_mom.sort_values()
                n = len(tier_mom)
                top_n = max(3, int(np.floor(n * 0.10)))
                bot_n = max(3, int(np.floor(n * 0.10)))
                
                longs = tier_mom.iloc[-top_n:].index.tolist()
                shorts = tier_mom.iloc[:bot_n].index.tolist()
                
                # Inverse-vol weighting
                vol = rets[longs + shorts].rolling(30).std().loc[test_dates].mean()
                inv_vol = 1.0 / (vol.replace(0, np.nan) + 1e-9)
                weights = inv_vol / inv_vol.sum()
                
                lw = weights[longs].values
                sw = weights[shorts].values
                
                lr = (test_rets[longs] * lw).sum(axis=1).fillna(0)
                sr = (test_rets[shorts] * sw).sum(axis=1).fillna(0)
                
                port_rets += (lr - sr) * 0.25
                n_tiers += 1
            
            if n_tiers == 0:
                fold_start += step
                continue
            
            port_rets = port_rets.where(test_reg, 0.0)
            port_rets = port_rets - COST * 0.5
            
            valid = port_rets.dropna()
            if len(valid) < 20:
                fold_start += step
                continue
                
            sr_val = float(np.sqrt(365) * valid.mean() / (valid.std() + 1e-9))
            folds.append({
                'fold': len(folds) + 1,
                'start': str(test_dates[0].date()),
                'end': str(test_dates[-1].date()),
                'bars': len(valid),
                'sharpe': sr_val,
                'return': float((1 + valid).prod() - 1),
                'tiers': n_tiers
            })
            port_parts.append(valid.values)
            bench_parts.append(market.loc[valid.index].fillna(0).values)
            fold_start += step
        
        if not port_parts:
            return [], [], []
        
        s_all = np.concatenate(port_parts)
        b_all = np.concatenate(bench_parts)
        n = min(len(s_all), len(b_all))
        return s_all[:n].tolist(), b_all[:n].tolist(), folds
    
    return bt()

# Run all configs
all_rets = []
bench_rets = None
summary = {}

for name, lb in configs:
    print(f'\n=== {name} ===')
    port, bench, folds = run_config(name, lb)
    summary[name] = folds
    
    if not port:
        print('  no output')
        continue
    
    all_rets.append(port)
    if bench_rets is None:
        bench_rets = bench
    
    sr = float(np.sqrt(365) * np.mean(port) / (np.std(port) + 1e-9))
    print(f'  folds={len(folds)} overall_sharpe={sr:.3f}')
    for f in folds:
        print(f"    fold {f['fold']:2d} {f['start']}->{f['end']} bars={f['bars']} tiers={f['tiers']} sharpe={f['sharpe']:.3f} ret={f['return']:.2%}")

# Pooled SPA
if all_rets:
    min_len = min(len(r) for r in all_rets + [bench_rets])
    all_rets = [r[:min_len] for r in all_rets]
    bench_rets = bench_rets[:min_len]
    try:
        from spa_hsu_test import stepwise_spa_test
        spa = stepwise_spa_test(all_rets, bench_rets, alpha=0.10)
        print('\n=== Pooled SPA ===')
        print('significant:', spa['significant'])
        print('T_max:', spa.get('T_max'))
        print('T_crit:', spa.get('T_crit'))
    except Exception as e:
        print('SPA failed:', e)

out = OUT / f'xrank_lit_framework_{pd.Timestamp.now():%Y%m%d_%H%M%S}.json'
out.parent.mkdir(exist_ok=True)
pd.Series({'configs': configs, 'summary': summary}).to_json(out, orient='split')
print('\nWrote', out)
