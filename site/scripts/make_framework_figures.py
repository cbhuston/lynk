"""
Architecture diagram for the framework report, plus a saved run of examples/quickstart.py.

Run from the repo root:  python site/scripts/make_framework_figures.py
"""

import contextlib
import io
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'site' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

try:
    import lightgbm  # noqa: F401  (import before pandas on Windows builds with mixed OpenMP runtimes)
except ImportError:
    pass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK2, BLUE_FILL, BLUE_EDGE, GRAY_FILL, GRAY_EDGE = '#52514e', '#eaf2fc', '#9cc0ec', '#f1f0ec', '#d9d8d2'


def box(ax, x, y, w, h, title, body, fill, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.015', fc=fill, ec=edge, lw=1))
    ax.text(x + 0.012, y + h - 0.035, title, fontsize=8.5, fontweight='bold', va='top')
    ax.text(x + 0.012, y + h - 0.095, body, fontsize=7.2, va='top', color='#1f1f1d', linespacing=1.35)


# Step 1: The architecture diagram
fig, ax = plt.subplots(figsize=(8.0, 4.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.0, 0.98, 'lynk.core  (market agnostic)', fontsize=9.5, fontweight='bold', color='#2a78d6', va='top')
core = [('probability', 'group softmax\nnormalization\nodds <-> probability'), ('models', 'LightGBM conditional\nlogit, baseline as\ninit score'),
        ('ratings', 'online Kalman\nentity ratings'), ('kelly', 'single, fractional,\nmutually exclusive,\nscenario, risk-limited'),
        ('parimutuel', 'own-money impact\nmax-EV bet, breakage\nplace / show splits'), ('portfolio', 'concave solver across\npools, exact gradients'),
        ('backtest', 'ledgers, summaries,\ngrid search, bankroll\npaths'), ('validation', 'date splits, walk-\nforward, leakage\nchecks, pre-registration'),
        ('metrics', 'event log-loss,\ncalibration, block\nbootstrap CIs')]
for i, (t, b) in enumerate(core):
    col, row = i % 5, i // 5
    box(ax, 0.0 + col * 0.2, 0.62 - row * 0.27, 0.185, 0.24, t, b, BLUE_FILL, BLUE_EDGE)
ax.text(0.0, 0.29, 'market modules  (one folder per market)', fontsize=9.5, fontweight='bold', color=INK2, va='top')
markets = [('horse_racing', 'schema, features,\nfinish_order (Harville),\nbets (win, place, show,\nexacta, DD), synthetic', GRAY_FILL, '#8a8983'),
           ('sports (next)', 'exchange order books,\npoint-spread and\ntotals models', '#fbfbfa', GRAY_EDGE),
           ('prediction markets\n(next)', 'binary contracts,\nfee schedules,\nresolution risk', '#fbfbfa', GRAY_EDGE)]
for i, (t, b, fc, ec) in enumerate(markets):
    box(ax, 0.0 + i * 0.335, 0.0, 0.31, 0.25, t, b, fc, ec)
fig.savefig(OUT / 'framework_architecture.png', bbox_inches='tight', dpi=200)
plt.close(fig)

# Step 2: Run the quickstart and keep its printed output for the report
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    runpy.run_path(str(ROOT / 'examples' / 'quickstart.py'), run_name='__main__')
(OUT / 'quickstart_output.txt').write_text(buffer.getvalue())
print(buffer.getvalue())
