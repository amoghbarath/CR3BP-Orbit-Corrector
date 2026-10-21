"""Figures 3-7 of the revised manuscript (run after make_results.py)."""
import json, numpy as np, collections
from figstyle import *
from runlib import SEEDS, ID_THRESH

EPS = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
R = [json.loads(l) for l in open('results/campaign_main.jsonl')]
B = json.load(open('results/basin.json'))
SHOW = ['BA', 'BA0', 'SDC', 'BAP', 'ORIG', 'CHORD', 'FP']


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k/n
    den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = z*np.sqrt(p*(1 - p)/n + z*z/(4*n*n))/den
    return c - h, c + h


# ---------------------------------------------------------------- Fig 3
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.55), sharey=True)
ax = axs[0]
for m in SHOW:
    p, lo, hi = [], [], []
    for li in range(7):
        L = [o['m'][m]['cls'] for o in R if o['li'] == li]
        k = sum(x == 'recovered' for x in L); n = len(L)
        p.append(100*k/n); a, b = wilson(k, n); lo.append(100*a); hi.append(100*b)
    ax.fill_between(EPS, lo, hi, color=MCOL[m], alpha=0.12, lw=0)
    ax.plot(EPS, p, color=MCOL[m], marker=MMARK[m], ms=3.5, lw=1.2, label=MSHORT[m])
ax.set_xscale('log'); ax.set_xlabel('relative perturbation $\\varepsilon$')
ax.set_ylabel('runs recovering the intended orbit (%)')
ax.set_ylim(-3, 103); ax.grid(True, alpha=0.6)
ax.set_title('(a) against perturbation size', fontsize=8.5, loc='left')
bins = np.array([0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, np.inf])
cen = np.array([3e-4, 3e-3, 1.7e-2, 5.5e-2, 1.7e-1, 5.5e-1, 1.7, 5.5, 20])
ax = axs[1]
for m in SHOW:
    p, lo, hi, xc = [], [], [], []
    for b in range(len(bins) - 1):
        L = [o['m'][m]['cls'] for o in R if np.isfinite(o['pre']) and bins[b] <= o['pre'] < bins[b+1]]
        if len(L) < 15:
            continue
        k = sum(x == 'recovered' for x in L); n = len(L)
        p.append(100*k/n); a, c = wilson(k, n); lo.append(100*a); hi.append(100*c); xc.append(cen[b])
    ax.fill_between(xc, lo, hi, color=MCOL[m], alpha=0.12, lw=0)
    ax.plot(xc, p, color=MCOL[m], marker=MMARK[m], ms=3.5, lw=1.2, label=MSHORT[m])
ax.set_xscale('log'); ax.set_xlabel('scaled initial residual $\\|\\hat{F}_0\\|$')
ax.grid(True, alpha=0.6)
ax.set_title('(b) against initial residual', fontsize=8.5, loc='left')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.06), handlelength=1.8)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig('figs/fig_recovery.png', facecolor='white', bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------- Fig 4
LIN = json.load(open('results/linearity.json'))
fig, axs = plt.subplots(2, 2, figsize=(6.0, 4.6))
axs = axs.ravel()
ax = axs[0]
for b, s in zip(B, SEEDS):
    ax.scatter(b['kappa'], b['kemp'], s=16, marker=SYSMARK[s['sys']], facecolor=FAMCOL[s['family']],
               edgecolor='white', linewidth=0.4, zorder=3)
xx = np.logspace(0, 6, 10)
ax.plot(xx, xx, color=C['ink2'], lw=0.7, ls='--')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$\\kappa$ (scaled monodromy)'); ax.set_ylabel('median $\\|\\hat{F}_0\\|/\\varepsilon$')
ax.set_title('(a) residual growth', fontsize=8.5, loc='left'); ax.grid(True, alpha=0.5)


def e50_panel(ax, xs, xl, tt, line=None):
    for b, s, xv in zip(B, SEEDS, xs):
        e = b['e50_BA']; fl = b['flag_BA']
        mk = SYSMARK[s['sys']]
        if fl == 'ok':
            ax.scatter(xv, e, s=16, marker=mk, facecolor=FAMCOL[s['family']], edgecolor='white',
                       linewidth=0.4, zorder=3)
        else:
            ax.scatter(xv, e, s=16, marker=mk, facecolor='white', edgecolor=FAMCOL[s['family']],
                       linewidth=0.8, zorder=3)
            f = 1/2.2 if fl == 'below' else 2.2
            ax.annotate('', xy=(xv, e*f), xytext=(xv, e*(0.95 if fl == 'below' else 1.05)),
                        arrowprops=dict(arrowstyle='->', color=FAMCOL[s['family']], lw=0.6))
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(3e-5, 3)
    ax.set_xlabel(xl); ax.grid(True, alpha=0.5)
    ax.set_title(tt, fontsize=8.5, loc='left')
    ax.set_ylabel('$\\varepsilon_{50}$')
    if line:
        line(ax)


e50_panel(axs[1], [b['Ar'] for b in B], 'orbit amplitude $A_r$ (LU)', '(b) half-recovery size vs amplitude')
ok = [b for b in B if b['flag_BA'] == 'ok' and b['e50_BA']*b['kappa'] < 20]
c = np.median([b['e50_BA']*b['kappa'] for b in ok])
def kline(ax):
    ax.plot(xx, c/xx, color=C['ink2'], lw=0.7, ls='--')
    ax.text(0.97, 0.95, f'$\\varepsilon_{{50}} = {c:.1f}/\\kappa$', transform=ax.transAxes,
            ha='right', va='top', fontsize=7, color=C['ink2'])
    ax.set_xlim(1, 2e5)
e50_panel(axs[2], [b['kappa'] for b in B], '$\\kappa$', '(c) half-recovery size vs $\\kappa$', kline)
okl = [(l, b) for l, b in zip(LIN, B) if l['flag_lin'] == 'ok' and b['flag_BA'] == 'ok']
r = np.median([b['e50_BA']/l['eps_lin'] for l, b in okl])
def lline(ax):
    ee = np.logspace(-5, 1, 10)
    ax.plot(ee, r*ee, color=C['ink2'], lw=0.7, ls='--')
    ax.text(0.97, 0.06, f'$\\varepsilon_{{50}} = {r:.1f}\\,\\varepsilon_{{\\mathrm{{lin}}}}$', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7, color=C['ink2'])
    ax.set_xlim(5e-5, 3)
e50_panel(axs[3], [l['eps_lin'] for l in LIN], 'linearization limit $\\varepsilon_{\\mathrm{lin}}$',
          '(d) half-recovery size vs linearization limit', lline)
from matplotlib.lines import Line2D
hf = [Line2D([], [], marker='o', ls='', color=FAMCOL[f], ms=4.5, label=FAMNAME[f]) for f in FAMORDER]
hs = [Line2D([], [], marker=SYSMARK[s], ls='', color=C['ink2'], mfc='none', ms=4.5, label=SYSNAME[s]) for s in SYSORDER]
fig.legend(handles=hf + hs, loc='lower center', ncol=7, fontsize=6.6, bbox_to_anchor=(0.5, -0.03),
           handletextpad=0.2, columnspacing=0.9)
fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.0, h_pad=0.8)
fig.savefig('figs/fig_basin.png', facecolor='white', bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------- Fig 5 (identity + stability index)
from matplotlib.lines import Line2D
hf = [Line2D([], [], marker='o', ls='', color=FAMCOL[f_], ms=4, label=FAMNAME[f_]) for f_ in FAMORDER]
hs = [Line2D([], [], marker=SYSMARK[s_], ls='', color=C['ink2'], mfc='none', ms=4, label=SYSNAME[s_]) for s_ in SYSORDER]
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.35), gridspec_kw=dict(width_ratios=[1.15, 1.0, 0.9]))
ax = axs[0]
edges = np.logspace(-13, 1, 57)
for m, ls in (('BA', '-'), ('ORIG', '-'), ('SDC', '--')):
    v = [o['m'][m]['maxn'] for o in R if o['m'][m]['status'] == 'converged' and 'maxn' in o['m'][m]]
    v = np.clip(v, 1.1e-13, 9)
    ax.hist(v, bins=edges, histtype='step', color=MCOL[m], lw=1.1, ls=ls, label=MSHORT[m])
ax.axvline(ID_THRESH, color=C['ink2'], lw=0.7, ls=':')
ax.set_xscale('log'); ax.set_xlabel('max scaled distance from seed')
ax.set_ylabel('converged runs'); ax.legend(fontsize=5.8, loc='upper left', handlelength=1.5)
ax.set_xticks([1e-12, 1e-9, 1e-6, 1e-3, 1])
ax.set_title('(a) identity of converged orbits', fontsize=8, loc='left')
ax = axs[1]
for m in ('ORIG', 'BA'):
    xs, ys = [], []
    for o in R:
        d = o['m'][m]
        if d['status'] == 'converged' and 'maxn' in d:
            xs.append(max(d['maxn'], 1e-13)); ys.append(max(d['dT_rel'], 1e-16))
    ax.scatter(xs, ys, s=3, color=MCOL[m], alpha=0.45, lw=0, label=MSHORT[m])
ax.axvline(ID_THRESH, color=C['ink2'], lw=0.7, ls=':')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('max scaled distance from seed'); ax.set_ylabel('$|T-T^{\\mathrm{seed}}|/T^{\\mathrm{seed}}$')
ax.set_xticks([1e-12, 1e-8, 1e-4, 1])
ax.legend(fontsize=5.8, loc='upper left', markerscale=2.5, handletextpad=0.2)
ax.set_title('(b) distance and period', fontsize=8, loc='left'); ax.grid(True, alpha=0.5)
ax = axs[2]
for s in SEEDS:
    ax.scatter(s['nu_jpl'], s['nu'], s=14, marker=SYSMARK[s['sys']], facecolor=FAMCOL[s['family']],
               edgecolor='white', linewidth=0.4, zorder=3)
cv = collections.defaultdict(list)
for o in R:
    d = o['m']['BA']
    if d['cls'] == 'recovered' and 'nu' in d:
        cv[o['si']].append(d['nu'])
for si, v in cv.items():
    ax.scatter(SEEDS[si]['nu_jpl'], np.median(v), s=34, marker='o', facecolor='none',
               edgecolor=C['ink2'], linewidth=0.5, zorder=2)
xx = np.logspace(-0.2, 3.4, 10)
ax.plot(xx, xx, color=C['ink2'], lw=0.6, ls='--')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$\\nu$, JPL catalog'); ax.set_ylabel('$\\nu$ computed here')
ax.set_title('(c) stability index', fontsize=8, loc='left'); ax.grid(True, alpha=0.5)
fig.legend(handles=hf + hs, loc='lower center', ncol=7, fontsize=6.2, bbox_to_anchor=(0.5, -0.07),
           handletextpad=0.2, columnspacing=0.8)
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
fig.savefig('figs/fig_identity.png', facecolor='white', bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------- Fig 7 (tolerance)
T = json.load(open('results/tolerance.json'))
tols = [r['tol'] for r in T[0]['rows']]
def band(key, conv_only=False, transform=None):
    med, lo, hi = [], [], []
    for k in range(len(tols)):
        v = []
        for t in T:
            r = t['rows'][k]
            if conv_only and r['status'] != 'converged':
                continue
            if key in r and r[key] is not None and np.isfinite(r[key]):
                v.append(r[key])
        v = np.array(v, float)
        if transform:
            v = transform(v)
        med.append(np.median(v)); lo.append(np.min(v)); hi.append(np.max(v))
    return np.array(med), np.array(lo), np.array(hi)
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.1))
for ax, (key, yl, tt, co) in zip(axs, (('floor', 'smallest reachable $\\|\\hat{F}\\|$', '(a) residual floor', False),
                                       ('dx0', 'scaled change in solution', '(b) solution vs. reference', True),
                                       ('steps', 'integration steps per period', '(c) cost', False))):
    med, lo, hi = band(key, co, (lambda v: np.maximum(v, 1e-16)) if key == 'dx0' else None)
    ax.fill_between(tols, lo, hi, color=C['blue'], alpha=0.15, lw=0)
    ax.plot(tols, med, color=C['blue'], marker='o', ms=3)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.invert_xaxis()
    ax.set_xlabel('RelTol = AbsTol'); ax.set_ylabel(yl); ax.grid(True, alpha=0.5)
    ax.set_title(tt, fontsize=8.5, loc='left')
    if key == 'floor':
        ax.axhline(1e-8, color=C['ink2'], lw=0.7, ls='--')
fig.tight_layout(w_pad=0.8)
fig.savefig('figs/fig_tolerance.png', facecolor='white', bbox_inches='tight')
plt.close(fig)
print('figures written')
