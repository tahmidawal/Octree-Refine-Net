"""Final summary plot for the fine-cell-only NM-ROM iteration.
Shows per-case bars and aggregate stats comparing the methods on the
adaptive integer-k mesh."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

TSV = Path('f_to_u_runs/exp2_intk/benchmark_results.tsv')

rows, avg = [], None
for line in TSV.read_text().splitlines():
    if line.startswith('i\t') or not line.strip():
        continue
    fld = line.split('\t')
    if fld[0] == 'AVG':
        # cols: cols[6:] in the same order as the writer
        keys = ['rel_cold','rel_warm','rel_fine','rel_mlp','rel_fom',
                'rel_warm_fineonly','rel_fine_fineonly','rel_mlp_fineonly','rel_fom_fineonly',
                'rel_warm_coarse','rel_fine_coarse','rel_mlp_coarse',
                'rel_cold_fom','rel_warm_fom','rel_fine_fom','rel_mlp_fom',
                't_cold','t_warm','t_fine','t_fom',
                'iters_cold','iters_warm','iters_fine']
        avg = {k: float(v) for k, v in zip(keys, fld[6:])}
    else:
        rows.append({
            'i': int(fld[0]), 'k1': float(fld[1]), 'k2': float(fld[2]),
            'N': int(fld[3]), 'n_fine': int(fld[4]), 'n_coarse': int(fld[5]),
            'rel_cold': float(fld[6]),  'rel_warm': float(fld[7]),
            'rel_fine': float(fld[8]),  'rel_mlp':  float(fld[9]), 'rel_fom': float(fld[10]),
            'rel_warm_fineonly': float(fld[11]), 'rel_fine_fineonly': float(fld[12]),
            'rel_mlp_fineonly':  float(fld[13]), 'rel_fom_fineonly': float(fld[14]),
            'rel_warm_coarse': float(fld[15]), 'rel_fine_coarse': float(fld[16]),
            'rel_mlp_coarse': float(fld[17]),
            't_cold': float(fld[22]), 't_warm': float(fld[23]),
            't_fine': float(fld[24]), 't_fom':  float(fld[25]),
            'iters_cold': int(float(fld[26])), 'iters_warm': int(float(fld[27])),
            'iters_fine': int(float(fld[28])),
        })

assert len(rows) == 6 and avg is not None

case_labels = [f"k=({int(r['k1'])},{int(r['k2'])})" for r in rows]
x = np.arange(len(rows))

fig = plt.figure(figsize=(15, 11))
gs  = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 0.8])

# ── Per-case full-mesh rel-L2 ──────────────────────────────────────────
ax = fig.add_subplot(gs[0, :])
methods = [
    ('rel_cold', 'COLD',           '#999999'),
    ('rel_warm', 'WARM (full)',    '#1f77b4'),
    ('rel_fine', 'WARM (fine-only) [NEW]', '#9467bd'),
    ('rel_mlp',  'MLP only',       '#2ca02c'),
    ('rel_fom',  'FOM',            '#d62728'),
]
w = 0.16
for i, (key, label, color) in enumerate(methods):
    vals = [r[key] for r in rows]
    ax.bar(x + (i - 2) * w, vals, w*0.95, label=label, color=color, edgecolor='black', linewidth=0.4)
ax.set_yscale('log'); ax.set_ylim(2e-2, 2)
ax.set_ylabel('rel-L2 vs analytical (log)')
ax.set_xticks(x); ax.set_xticklabels(case_labels)
ax.set_title('Full-mesh rel-L2: fine-cell NM-ROM vs alternatives (adaptive integer-k)')
ax.legend(ncol=5, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.07))
ax.grid(True, axis='y', alpha=0.3, which='both')

# ── Fine vs coarse breakdown ──────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
labels = ['WARM full', 'WARM fine\n[NEW]', 'MLP only', 'FOM']
fine_vals   = [avg['rel_warm_fineonly'], avg['rel_fine_fineonly'],
               avg['rel_mlp_fineonly'],  avg['rel_fom_fineonly']]
coarse_vals = [avg['rel_warm_coarse'],   avg['rel_fine_coarse'],
               avg['rel_mlp_coarse'],    0.0]   # FOM coarse not tracked
xx = np.arange(len(labels))
b1 = ax.bar(xx - 0.20, fine_vals,   0.38, label='fine cells',   color='#9467bd', edgecolor='black')
b2 = ax.bar(xx + 0.20, coarse_vals, 0.38, label='coarse cells', color='#7f7f7f', edgecolor='black')
for b, v in zip(b1, fine_vals):
    ax.text(b.get_x()+b.get_width()/2, v*1.05, f"{v:.3f}", ha='center', fontsize=8)
for b, v in zip(b2, coarse_vals):
    if v > 0:
        ax.text(b.get_x()+b.get_width()/2, v*1.05, f"{v:.3f}", ha='center', fontsize=8)
ax.set_yscale('log'); ax.set_ylim(0.02, 1)
ax.set_xticks(xx); ax.set_xticklabels(labels)
ax.set_ylabel('avg rel-L2 vs analytical (log)')
ax.set_title('Avg rel-L2 broken down by mesh region\n(WARM-fine matches MLP on fine, anchors coarse)')
ax.legend(); ax.grid(True, axis='y', alpha=0.3, which='both')

# ── Wall-clock time per method ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
methods_t = ['COLD', 'WARM\n(full)', 'WARM\n(fine)', 'FOM CG']
times = [avg['t_cold'], avg['t_warm'], avg['t_fine'], avg['t_fom']]
colors = ['#999999', '#1f77b4', '#9467bd', '#d62728']
b = ax.bar(np.arange(len(methods_t)), times, 0.6, color=colors, edgecolor='black')
for bb, v in zip(b, times):
    ax.text(bb.get_x()+bb.get_width()/2, v*1.10, f"{v:.2f}s", ha='center', fontsize=9)
ax.set_yscale('log'); ax.set_xticks(np.arange(len(methods_t)))
ax.set_xticklabels(methods_t)
ax.set_ylabel('avg wall time per case (s, log)')
ax.set_title('Per-case time: fine-cell NM-ROM is 37× faster than warm-full')
ax.grid(True, axis='y', alpha=0.3, which='both')

# ── Iterations per case ──────────────────────────────────────────────
ax = fig.add_subplot(gs[2, :])
ic = [r['iters_cold'] for r in rows]
iw = [r['iters_warm'] for r in rows]
ifn = [r['iters_fine'] for r in rows]
ax.bar(x - 0.27, ic,  0.25, label='COLD',          color='#999999', edgecolor='black')
ax.bar(x,        iw,  0.25, label='WARM (full)',   color='#1f77b4', edgecolor='black')
ax.bar(x + 0.27, ifn, 0.25, label='WARM (fine)',   color='#9467bd', edgecolor='black')
ax.axhline(48, ls='--', lw=0.5, color='red', alpha=0.5)
ax.text(len(rows)-0.35, 48.5, 'max iters', fontsize=8, color='red')
ax.set_xticks(x); ax.set_xticklabels(case_labels)
ax.set_ylabel('GN iterations to converge')
ax.set_title('Per-case GN iterations: fine-cell solver converges in 3–7 iters')
ax.legend(); ax.grid(True, axis='y', alpha=0.3)

plt.suptitle('Fine-cell-only NM-ROM (with Tikhonov anchor to MLP) on adaptive quadtree mesh — 6 test cases', fontsize=12)
plt.tight_layout()
out = Path('f_to_u_runs/finecell_summary.png')
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f"Saved {out}")
