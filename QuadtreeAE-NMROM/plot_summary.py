"""Build a summary figure comparing the adaptive-mesh and uniform-mesh
benchmarks across the 4 methods (cold / warm / MLP / FOM)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ADAPT = Path('f_to_u_runs/exp2_intk/benchmark_results.tsv')
UNIF  = Path('f_to_u_runs/exp4_uniform/benchmark_results.tsv')

def load(p):
    rows, avg = [], None
    for line in p.read_text().splitlines():
        if line.startswith('i\t') or not line.strip():
            continue
        f = line.split('\t')
        if f[0] == 'AVG':
            avg = {
                'rel_cold': float(f[4]), 'rel_warm': float(f[5]),
                'rel_mlp':  float(f[6]), 'rel_fom':  float(f[7]),
                't_cold':   float(f[11]), 't_warm':  float(f[12]),
                't_fom':    float(f[13]),
                'iters_cold': float(f[14]), 'iters_warm': float(f[15]),
            }
        else:
            rows.append({
                'i': int(f[0]), 'k1': float(f[1]), 'k2': float(f[2]),
                'rel_cold': float(f[4]), 'rel_warm': float(f[5]),
                'rel_mlp':  float(f[6]), 'rel_fom':  float(f[7]),
                't_cold':   float(f[11]), 't_warm':  float(f[12]),
                't_fom':    float(f[13]),
                'iters_cold': int(float(f[14])), 'iters_warm': int(float(f[15])),
            })
    return rows, avg

a_rows, a_avg = load(ADAPT)
u_rows, u_avg = load(UNIF)

assert len(a_rows) == len(u_rows) == 6

# Match cases by (k1, k2) so we plot consistent x-axis
def by_k(rows): return {(r['k1'], r['k2']): r for r in rows}
a_by = by_k(a_rows); u_by = by_k(u_rows)
ks = sorted(a_by.keys())
case_labels = [f"({int(k1)},{int(k2)})" for k1, k2 in ks]

fig = plt.figure(figsize=(14, 9))
gs  = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

# ── Panel 1: rel-L2 per case (both meshes overlaid) ──────────────────────
ax = fig.add_subplot(gs[0, :])
x = np.arange(len(ks))
w = 0.22
methods = [
    ('rel_cold', 'COLD',     '#999999', '#cccccc'),
    ('rel_warm', 'WARM',     '#1f77b4', '#7ab7e0'),
    ('rel_mlp',  'MLP only', '#2ca02c', '#a8d8a8'),
    ('rel_fom',  'FOM',      '#d62728', '#f0a0a0'),
]
for i, (key, label, c_dark, c_light) in enumerate(methods):
    ax.bar(x + (i - 1.5)*w, [a_by[k][key] for k in ks], w*0.45,
           label=f'{label} (adaptive)', color=c_dark, edgecolor='black', linewidth=0.4)
    ax.bar(x + (i - 1.5)*w + w*0.5, [u_by[k][key] for k in ks], w*0.45,
           label=f'{label} (uniform)', color=c_light, edgecolor='black', linewidth=0.4, hatch='//')
ax.set_yscale('log')
ax.set_ylim(5e-4, 2)
ax.set_ylabel('rel-L2 vs analytical (log)')
ax.set_xticks(x)
ax.set_xticklabels([f"k={lbl}" for lbl in case_labels])
ax.set_title('Per-case accuracy: adaptive (solid) vs uniform mesh (hatched)')
ax.axhline(0.01, color='gray', lw=0.5, ls='--', alpha=0.5)
ax.text(len(ks)-1.05, 0.011, '1% rel-L2', fontsize=8, color='gray', va='bottom', ha='right')
ax.legend(ncol=4, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.08))
ax.grid(True, axis='y', alpha=0.3, which='both')

# ── Panel 2: average rel-L2 (bar chart) ──────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
labels = ['COLD', 'WARM', 'MLP', 'FOM']
keys   = ['rel_cold', 'rel_warm', 'rel_mlp', 'rel_fom']
colors = ['#999999', '#1f77b4', '#2ca02c', '#d62728']
adapt_vals = [a_avg[k] for k in keys]
unif_vals  = [u_avg[k] for k in keys]
xx = np.arange(len(labels))
bars1 = ax.bar(xx - 0.20, adapt_vals, 0.38, label='adaptive', color=colors, alpha=0.7, edgecolor='black')
bars2 = ax.bar(xx + 0.20, unif_vals,  0.38, label='uniform', color=colors, hatch='//', alpha=0.95, edgecolor='black')
for b, v in zip(bars1, adapt_vals): ax.text(b.get_x()+b.get_width()/2, v*1.05, f"{v:.3f}", ha='center', fontsize=8)
for b, v in zip(bars2, unif_vals):  ax.text(b.get_x()+b.get_width()/2, v*1.05, f"{v:.3f}", ha='center', fontsize=8)
ax.set_yscale('log')
ax.set_xticks(xx); ax.set_xticklabels(labels)
ax.set_ylabel('avg rel-L2 vs analytical (log)')
ax.set_title('Mean rel-L2 across 6 test cases')
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.3, which='both')

# ── Panel 3: GN iters comparison ─────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
adapt_iters_cold = [a_by[k]['iters_cold'] for k in ks]
adapt_iters_warm = [a_by[k]['iters_warm'] for k in ks]
unif_iters_cold  = [u_by[k]['iters_cold'] for k in ks]
unif_iters_warm  = [u_by[k]['iters_warm'] for k in ks]
ax.bar(x - 0.30, adapt_iters_cold, 0.18, label='COLD adaptive', color='#999999', edgecolor='black')
ax.bar(x - 0.10, adapt_iters_warm, 0.18, label='WARM adaptive', color='#1f77b4', edgecolor='black')
ax.bar(x + 0.10, unif_iters_cold,  0.18, label='COLD uniform',  color='#cccccc', hatch='//', edgecolor='black')
ax.bar(x + 0.30, unif_iters_warm,  0.18, label='WARM uniform',  color='#7ab7e0', hatch='//', edgecolor='black')
ax.axhline(48, ls='--', lw=0.5, color='red', alpha=0.5)
ax.text(len(ks)-0.4, 48.5, 'max iters', fontsize=8, color='red')
ax.set_xticks(x); ax.set_xticklabels(case_labels, fontsize=8)
ax.set_ylabel('GN iterations to convergence')
ax.set_xlabel('test case  (k1, k2)')
ax.set_title('Iterations: WARM converges faster than COLD')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, axis='y', alpha=0.3)

plt.suptitle('F→u NM-ROM: cold-start vs MLP-warm-start vs MLP-only vs FOM', fontsize=12)
plt.tight_layout()
out = Path('f_to_u_runs/summary.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved {out}")
