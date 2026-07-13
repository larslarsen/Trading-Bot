import json
from pathlib import Path
import numpy as np
from spa_hsu_test import studentized_performance, stepwise_spa_test

OUT = Path('/home/lars/trading-bot/backtest_output')
latest = sorted(OUT.glob('tail_vs_large_*.json'))[-1]
rows = json.loads(latest.read_text())
print('Loaded', latest.name, 'rows', len(rows))

usable = [r for r in rows if len(r.get('port_rets', [])) >= 3]
max_len = max(len(r['port_rets']) for r in usable)
print('usable', len(usable), 'max_len', max_len)

# Build benchmark as average across all strategy arrays, truncated to common length
bench_stack = []
for r in usable:
    a = np.array(r['port_rets'], dtype=float)
    bench_stack.append(a)
min_len = min(len(a) for a in bench_stack)
bench = np.mean(np.array([a[:min_len] for a in bench_stack]), axis=0).tolist()

all_rets = []
labels = []
for r in usable:
    a = np.array(r['port_rets'], dtype=float)[:min_len]
    labels.append(f"{r['tier']}_{r['coin']}_{r['rule']}")
    all_rets.append(a.tolist())

print('Strategies:', len(all_rets))
for i, arr in enumerate(all_rets):
    T, mu, v = studentized_performance(arr, bench)
    print(f"  {labels[i]:45s} T={T:7.3f} mu_diff={mu:+.6f} var={v:.6f} n={len(arr)}")

spa = stepwise_spa_test(all_rets, bench, alpha=0.10)
print('\nSPA result:', spa)
