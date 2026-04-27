"""Build a 4-panel summary figure from a benchmark run's results.tsv:
  - per-case rel-L2 (log) for COLD / WARM-full / WARM-fine / MLP / FOM
  - mean rel-L2 broken down by fine vs coarse cells
  - mean per-case wall time
  - per-case GN iterations

Usage:
  python plot_finecell_summary.py --exp-dir runs/expN [--out runs/summary.png]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def parse_tsv(path):
    rows, avg = [], None
    keys = ['rel_cold','rel_warm','rel_fine','rel_mlp','rel_fom',
            'rel_warm_fineonly','rel_fine_fineonly','rel_mlp_fineonly','rel_fom_fineonly',
            'rel_warm_coarse','rel_fine_coarse','rel_mlp_coarse',
            'rel_cold_fom','rel_warm_fom','rel_fine_fom','rel_mlp_fom',
            't_cold','t_warm','t_fine','t_fom',
            'iters_cold','iters_warm','iters_fine']
    for line in path.read_text().splitlines():
        if line.startswith('i\t') or not line.strip():
            continue
        f = line.split('\t')
        if f[0] == 'AVG':
            avg = {k: float(v) for k, v in zip(keys, f[6:])}
        else:
            rows.append({
                'i': int(f[0]), 'k1': float(f[1]), 'k2': float(f[2]),
                'N': int(f[3]), 'n_fine': int(f[4]), 'n_coarse': int(f[5]),
                'rel_cold': float(f[6]),  'rel_warm': float(f[7]),
                'rel_fine': float(f[8]),  'rel_mlp':  float(f[9]),
                'rel_fom':  float(f[10]),
                'rel_warm_fineonly': float(f[11]), 'rel_fine_fineonly': float(f[12]),
                'rel_mlp_fineonly':  float(f[13]), 'rel_fom_fineonly':  float(f[14]),
                'rel_warm_coarse': float(f[15]), 'rel_fine_coarse': float(f[16]),
                'rel_mlp_coarse':  float(f[17]),
                't_cold': float(f[22]), 't_warm': float(f[23]),
                't_fine': float(f[24]), 't_fom':  float(f[25]),
                'iters_cold': int(float(f[26])), 'iters_warm': int(float(f[27])),
                'iters_fine': int(float(f[28])),
            })
    return rows, avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp-dir', required=True, help='Experiment directory containing benchmark_results.tsv')
    p.add_argument('--out',     default=None,
                   help='Output PNG path. Defaults to <exp-dir>/finecell_summary.png')
    args = p.parse_args()

    exp_dir = Path(args.exp_dir)
    tsv = exp_dir / 'benchmark_results.tsv'
    if not tsv.exists():
        raise SystemExit(f"Could not find {tsv}. Run 'bench' subcommand first.")

    rows, avg = parse_tsv(tsv)
    if not rows or avg is None:
        raise SystemExit(f"{tsv} has no rows or no AVG line.")

    out = Path(args.out) if args.out else exp_dir / 'plots' / 'finecell_summary.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    case_labels = [f"k=({int(r['k1'])},{int(r['k2'])})" for r in rows]
    x = np.arange(len(rows))

    fig = plt.figure(figsize=(15, 11))
    gs  = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 0.8])

    ax = fig.add_subplot(gs[0, :])
    methods = [
        ('rel_cold', 'COLD',                       '#999999'),
        ('rel_warm', 'WARM (full residual)',       '#1f77b4'),
        ('rel_fine', 'WARM (fine-cells-only)',     '#9467bd'),
        ('rel_mlp',  'MLP only',                   '#2ca02c'),
        ('rel_fom',  'FOM (CG on FVM)',            '#d62728'),
    ]
    w = 0.16
    for i, (key, label, color) in enumerate(methods):
        vals = [r[key] for r in rows]
        ax.bar(x + (i - 2) * w, vals, w*0.95, label=label,
               color=color, edgecolor='black', linewidth=0.4)
    ax.set_yscale('log'); ax.set_ylim(1e-3, 2)
    ax.set_ylabel('rel-L2 vs analytical (log)')
    ax.set_xticks(x); ax.set_xticklabels(case_labels)
    ax.set_title('Per-case full-mesh rel-L2 vs analytical')
    ax.legend(ncol=5, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.07))
    ax.grid(True, axis='y', alpha=0.3, which='both')

    # rel-L2 broken down by region
    ax = fig.add_subplot(gs[1, 0])
    labels = ['WARM\nfull', 'WARM\nfine', 'MLP\nonly', 'FOM']
    fine_vals   = [avg['rel_warm_fineonly'], avg['rel_fine_fineonly'],
                   avg['rel_mlp_fineonly'],  avg['rel_fom_fineonly']]
    coarse_vals = [avg['rel_warm_coarse'],   avg['rel_fine_coarse'],
                   avg['rel_mlp_coarse'],    0.0]
    xx = np.arange(len(labels))
    b1 = ax.bar(xx - 0.20, fine_vals,   0.38, label='fine cells',   color='#9467bd', edgecolor='black')
    b2 = ax.bar(xx + 0.20, coarse_vals, 0.38, label='coarse cells', color='#7f7f7f', edgecolor='black')
    for b, v in zip(b1, fine_vals):
        ax.text(b.get_x()+b.get_width()/2, max(v, 1e-3)*1.07, f"{v:.3f}", ha='center', fontsize=8)
    for b, v in zip(b2, coarse_vals):
        if v > 0:
            ax.text(b.get_x()+b.get_width()/2, v*1.07, f"{v:.3f}", ha='center', fontsize=8)
    ax.set_yscale('log')
    ax.set_xticks(xx); ax.set_xticklabels(labels)
    ax.set_ylabel('avg rel-L2 vs analytical (log)')
    ax.set_title('Avg rel-L2: fine cells vs coarse cells')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3, which='both')

    # Per-method wall time
    ax = fig.add_subplot(gs[1, 1])
    methods_t = ['COLD', 'WARM\n(full)', 'WARM\n(fine)', 'FOM CG']
    times = [avg['t_cold'], avg['t_warm'], avg['t_fine'], avg['t_fom']]
    colors = ['#999999', '#1f77b4', '#9467bd', '#d62728']
    b = ax.bar(np.arange(len(methods_t)), times, 0.6, color=colors, edgecolor='black')
    for bb, v in zip(b, times):
        ax.text(bb.get_x()+bb.get_width()/2, max(v, 1e-3)*1.10, f"{v:.2f}s",
                ha='center', fontsize=9)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(methods_t))); ax.set_xticklabels(methods_t)
    ax.set_ylabel('avg wall time per case (s, log)')
    ax.set_title('Per-case wall time')
    ax.grid(True, axis='y', alpha=0.3, which='both')

    # Iters per case
    ax = fig.add_subplot(gs[2, :])
    ic  = [r['iters_cold'] for r in rows]
    iw  = [r['iters_warm'] for r in rows]
    ifn = [r['iters_fine'] for r in rows]
    ax.bar(x - 0.27, ic,  0.25, label='COLD',        color='#999999', edgecolor='black')
    ax.bar(x,        iw,  0.25, label='WARM (full)', color='#1f77b4', edgecolor='black')
    ax.bar(x + 0.27, ifn, 0.25, label='WARM (fine)', color='#9467bd', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(case_labels)
    ax.set_ylabel('GN iterations')
    ax.set_title('Per-case GN iterations to converge')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle(f'F→u NM-ROM benchmark — {exp_dir.name}', fontsize=12)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
