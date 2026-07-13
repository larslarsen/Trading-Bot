import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('data')
OUT = Path('backtest_output')
COST = 0.0008
TOP_N = 5

LIQUID_TAIL = [
    'BLUB_USDT','RATS_USDT','HIPPO_USDT','SOLV_USDT','TOWNS_USDT',
    'ACT_USDT','ARPA_USDT','AGT_USDT','PIXEL_USDT','VEX_USDT',
    'BMT_USDT','FHE_USDT','RESOLV_USDT','BIGTIME_USDT','PROMPT_USDT',
    'COOKIE_USDT','PARTI_USDT','SHELL_USDT','SAGA_USDT','SWARMS_USDT',
    'PI_USDT','ZEUS_USDT','WCT_USDT','THE_USDT','SUNDOG_USDT',
    'A8_USDT','MITO_USDT','GTC_USDT','TREE_USDT','UPC_USDT',
    'FTT_USDT','PYR_USDT','TRADOOR_USDT','PSG_USDT','BEL_USDT',
]

def _pick(symbol):
    for suffix in ['', '_binance', '_mexc']:
        p = ROOT / f'{symbol}{suffix}_1d_max.csv'
        if p.exists():
            return p
    return None

def load_symbol(symbol):
    p = _pick(symbol)
    if p is None:
        return None
    df = pd.read_csv(p, parse_dates=['ts'])
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['close','high','low','volume']).sort_values('ts').reset_index(drop=True)
    return df

def add_signals(df):
    df = df.copy()
    delta = df['close'].diff()
    rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0).rolling(14).mean() + 1e-9)
    df['rsi'] = 100 - 100/(1+rs)
    df['mom_7'] = df['close'] / df['close'].shift(7) - 1
    h = df['high'].rolling(20).max()
    l = df['low'].rolling(20).min()
    df['donchian_pos'] = (df['close'] > h.shift(1)).astype(int)
    df['triple_rsi_pos'] = (df['rsi'] < 30).astype(int)
    return df

def cross_sectional_portfolio(symbol_dfs):
    # align by date
    closes = pd.DataFrame({s: df.set_index('ts')['close'] for s, df in symbol_dfs.items() if df is not None})
    if closes.empty:
        return None
    closes = closes.sort_index().dropna(how='all')

    # build signals per day
    signals = pd.DataFrame(index=closes.index)
    ranks = pd.DataFrame(index=closes.index)
    for s, df in symbol_dfs.items():
        if df is None:
            continue
        sub = df.set_index('ts')[['rsi','mom_7','donchian_pos','triple_rsi_pos']].reindex(closes.index).fillna(0)
        # cross-sectional ranks each day
        rsi_rank = sub['rsi'].rank(pct=True, na_option='keep')
        mom_rank = sub['mom_7'].rank(pct=True, ascending=True, na_option='keep')  # low momentum out
        signals[s] = (sub['triple_rsi_pos'] + sub['donchian_pos'] + (1 - rsi_rank) + (1 - mom_rank)) / 4

    # long top N by composite score, flat on rest
    positions = signals.where(signals.rank(axis=1, ascending=False) <= TOP_N, 0).fillna(0)
    # normalize weights each day
    pos_sum = positions.sum(axis=1)
    pos_sum[pos_sum == 0] = np.nan
    weights = positions.div(pos_sum, axis=0).fillna(0)

    # dollar-neutral short bottom N? Literature mixed. Use cash first.
    # daily returns
    rets = closes.pct_change().fillna(0)
    port_rets = (weights.shift(1) * rets).sum(axis=1).fillna(0)

    # costs on rebalances
    w_prev = weights.shift(1).fillna(0)
    turnover = (weights - w_prev).abs().sum(axis=1)
    port_rets = port_rets - turnover * COST

    return port_rets.loc[port_rets.index >= (ret_dates := closes.dropna().index[0])].dropna()

# build universe dict
symbol_dfs = {}
for coin in LIQUID_TAIL:
    df = load_symbol(coin)
    if df is None:
        continue
    df = add_signals(df).dropna(subset=['rsi','mom_7'])
    if len(df) > 120:
        symbol_dfs[coin] = df

port = cross_sectional_portfolio(symbol_dfs)
if port is None:
    print('No portfolio')
    raise SystemExit

arr = port.values
sharpe = float(np.sqrt(365) * np.mean(arr) / (np.std(arr, ddof=1)+1e-9))
total = float(np.prod(1+arr) - 1)
wr = float(np.mean(arr > 0))

print(f'\n=== Cross-sectional ranked tail portfolio (top {TOP_N}) ===')
print(f'Sharpe={sharpe:.3f} return={total:.2%} wr={wr*100:.1f}% bars={len(arr)}')
print(port.head(10).to_string())
print('...')
print(port.tail(5).to_string())

out = OUT / f'xrank_tail_top{TOP_N}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.json'
out.parent.mkdir(exist_ok=True)
pd.Series(arr, index=port.index).to_json(out, orient='split', date_format='iso')
print('\nWrote', out)
