"""Figure 1: the sixty seed orbits, each drawn at its own scale."""
import json, numpy as np
from figstyle import *
from runlib import SEEDS
from cr3bp_core import trajectory


def sci(v):
    e = int(np.floor(np.log10(abs(v))))
    m = v/10**e
    if round(m, 1) >= 10:
        m /= 10; e += 1
    return f'{m:.1f}e{e}' if e not in (0, -1) else f'{v:.2g}'


fig, axs = plt.subplots(4, 3, figsize=(6.6, 3.3))
for r, sy in enumerate(SYSORDER):
    for c, fam in enumerate(FAMORDER):
        ax = axs[r, c]
        ss = sorted([s for s in SEEDS if s['sys'] == sy and s['family'] == fam], key=lambda s: s['level'])
        for j, s in enumerate(ss):
            _, X = trajectory(np.array(s['x0']), s['T'], s['mu'], 800)
            u = X[:, 1] if fam == 'halo' else X[:, 0]
            v = X[:, 2] if fam == 'halo' else X[:, 1]
            cu, cv = 0.5*(u.max() + u.min()), 0.5*(v.max() + v.min())
            sc = max(u.max() - u.min(), v.max() - v.min())/2
            xs = (u - cu)/sc*0.95 + 2.3*j
            ys = (v - cv)/sc*0.95
            ax.plot(xs, ys, color=FAMCOL[fam], lw=0.9)
            # secondary (if inside the drawing window)
            ps = ((1 - s['mu']) - cu)/sc*0.95
            if abs(ps) <= 1.0 and fam != 'halo':
                ax.plot(ps + 2.3*j, (0 - cv)/sc*0.95, marker='o', ms=2.2, color=C['ink2'], lw=0)
            ax.text(2.3*j, -1.32, sci(s['Ar']), ha='center', va='top', fontsize=5.6, color=C['ink'])
            nu = s['nu']
            nut = '1' if nu < 1.0005 else (f'{nu:.2f}' if nu < 10 else f'{nu:.0f}')
            ax.text(2.3*j, 1.18, nut,
                    ha='center', va='bottom', fontsize=5.4, color=C['ink2'])
        ax.set_xlim(-1.25, 2.3*4 + 1.25); ax.set_ylim(-1.75, 1.55)
        ax.set_aspect('equal'); ax.axis('off')
        if r == 0:
            ax.set_title(FAMNAME[fam] + (' ($y$-$z$)' if fam == 'halo' else ' ($x$-$y$)'), fontsize=8)
        if c == 0:
            ax.text(-1.6, 0, SYSNAME[sy], rotation=90, ha='right', va='center', fontsize=7.5)
fig.text(0.5, 0.005, 'each orbit drawn at its own scale; number below = amplitude $A_r$ (LU), '
         'above = stability index $\\nu$; dot = secondary body', ha='center', fontsize=6.5, color=C['ink2'])
fig.subplots_adjust(left=0.05, right=0.995, top=0.935, bottom=0.045, wspace=0.03, hspace=0.0)
fig.savefig('figs/fig_seeds.png', facecolor='white')
print('fig_seeds written')
