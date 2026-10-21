"""Figure 2: one perturbed seed followed through every corrector.

Seed: Earth-Moon L2 Halo, level 2 (the strongly unstable 3-D member used in
the directional study); perturbation size eps = 1e-3, the same perturbations
as the campaign (rng([seed, 1, trial])).  The first trial in which the
bordered corrector recovers the seed and the original method converges to a
different orbit is shown.  Writes figs/fig_example.png and
results/example_info.json.
"""
import json, numpy as np
from figstyle import *
from runlib import SEEDS, METHODS, run_method, classify, W_of
from metrics import perturb, orbit_identity
from cr3bp_core import trajectory, prop_state

si = [i for i, s in enumerate(SEEDS) if s['sys'] == 'earth-moon' and s['family'] == 'halo'
      and s['level'] == 2][0]
s = SEEDS[si]
x0 = np.array(s['x0']); mu = s['mu']; T = s['T']; W = W_of(s)
LI, EPSV = 1, 1e-3

chosen = None
for k in range(20):
    rng = np.random.default_rng([si, LI, k])
    xp = perturb(x0, EPSV, s['Ar'], s['Av'], rng)
    rb = run_method('BA', s, xp); cb, mb = classify(s, rb)
    ro = run_method('ORIG', s, xp); co, mo = classify(s, ro)
    if cb == 'recovered' and co == 'other':
        chosen = (k, xp)
        break
k, xp = chosen
res = {}
for m in METHODS:
    r = run_method(m, s, xp)
    cls, mm = classify(s, r)
    res[m] = (r, cls, mm)
xT, _, _ = prop_state(xp, T, mu, stm=False)
pre = float(np.linalg.norm((xT - xp)/W))

info = dict(si=si, k=k, eps=EPSV, nu=s['nu'], level=s['level'], pre=pre, Ar=s['Ar'])
for m, (r, cls, mm) in res.items():
    d = dict(status=r.status, cls=cls, iters=r.iters, hist=r.hist, alpha=r.alpha)
    if mm:
        d.update(maxn=mm['maxn'], dT=mm['dT_rel'], dC=mm['dC'])
    info[m] = d
# warm-up length of the original method: iterations before alpha first reaches 1
a = res['ORIG'][0].alpha
info['ORIG']['warmup'] = next((i + 1 for i, v in enumerate(a) if v >= 1.0), None)
info['BA']['n_short'] = int(sum(v < 1 for v in res['BA'][0].alpha))
json.dump(info, open('results/example_info.json', 'w'), indent=1)

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(6.6, 2.55))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1.6, 1], wspace=0.28, hspace=0.12)
ax = fig.add_subplot(gs[:, 0])
P = (1, 2)    # y-z projection
_, Xs = trajectory(x0, T, mu, 1500)
_, Xp = trajectory(xp, T, mu, 1500)
rb = res['BA'][0]; ro = res['ORIG'][0]
_, Xb = trajectory(rb.x0, rb.T, mu, 1500)
_, Xo = trajectory(ro.x0, ro.T, mu, 1500)
ax.plot(Xs[:, P[0]], Xs[:, P[1]], color=C['ink2'], lw=2.6, alpha=0.35, label='seed', solid_capstyle='round')
ax.plot(Xp[:, P[0]], Xp[:, P[1]], color=C['red'], lw=0.8, label='perturbed guess, one period')
ax.plot(Xp[0, P[0]], Xp[0, P[1]], 'o', ms=4, mfc='white', mec=C['red'], mew=1.0)
ax.plot(Xp[-1, P[0]], Xp[-1, P[1]], 's', ms=4, mfc='white', mec=C['red'], mew=1.0)
ax.plot(Xb[:, P[0]], Xb[:, P[1]], color=MCOL['BA'], lw=0.9, ls='--', label=MSHORT['BA'])
ax.plot(Xo[:, P[0]], Xo[:, P[1]], color=MCOL['ORIG'], lw=0.9, label=MSHORT['ORIG'] + ' (different orbit)')
ax.set_xlabel('$y$ (LU)'); ax.set_ylabel('$z$ (LU)')
ax.set_aspect('equal', adjustable='datalim')
ax.grid(True, alpha=0.5)
ax.set_title('(a) orbits ($y$-$z$ projection)', loc='left')
# inset at the point of largest separation between ORIG and seed
d = np.min(np.linalg.norm(Xo[:, None, :3] - Xs[None, ::5, :3], axis=2), axis=1)
j = int(np.argmax(d))
cx, cy = Xo[j, P[0]], Xo[j, P[1]]
w = max(6*d[j], 0.004)
ins = ax.inset_axes([0.3, 0.36, 0.4, 0.3])
for X, kw in ((Xs, dict(color=C['ink2'], lw=2.6, alpha=0.35)), (Xb, dict(color=MCOL['BA'], lw=0.9, ls='--')),
              (Xo, dict(color=MCOL['ORIG'], lw=0.9))):
    ins.plot(X[:, P[0]], X[:, P[1]], **kw)
ins.set_xlim(cx - w, cx + w); ins.set_ylim(cy - w, cy + w)
ins.set_xticks([]); ins.set_yticks([])
for sp in ins.spines.values():
    sp.set_visible(True); sp.set_color(C['ink2']); sp.set_linewidth(0.5)
ax.indicate_inset_zoom(ins, edgecolor=C['ink2'], lw=0.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=6.3, ncol=1, handlelength=2.2)

ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
for m in METHODS:
    r = res[m][0]
    h = np.maximum(np.array(r.hist), 1e-16)
    ax2.plot(np.arange(len(h)), h, color=MCOL[m], marker=MMARK[m], ms=2.6, lw=1.0,
             label=f"{MSHORT[m]}: {'recovered' if res[m][1] == 'recovered' else ('different orbit' if res[m][1] == 'other' else 'failed')}")
    if m in ('BA', 'ORIG', 'BAP', 'CHORD'):
        ax3.plot(np.arange(1, len(r.alpha) + 1), r.alpha, color=MCOL[m], marker=MMARK[m], ms=2.6, lw=1.0)
ax2.axhline(1e-8, color=C['ink2'], lw=0.6, ls=':')
ax2.set_yscale('log'); ax2.set_ylabel('scaled residual $\\|\\hat{F}\\|$')
ax2.set_ylim(1e-14, 1e3); ax2.grid(True, alpha=0.5)
ax2.set_title('(b) scaled residual per iteration', loc='left')
ax3.text(0.99, 0.08, '(c) accepted step length', transform=ax3.transAxes, ha='right', va='bottom', fontsize=8.5)
plt.setp(ax2.get_xticklabels(), visible=False)
ax3.set_yscale('log'); ax3.set_ylabel('step $\\alpha$'); ax3.set_xlabel('iteration')
ax3.set_ylim(5e-4, 2); ax3.grid(True, alpha=0.5)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -1.05), fontsize=6.0, ncol=2, handlelength=1.6,
           columnspacing=1.0)
fig.savefig('figs/fig_example.png', facecolor='white', bbox_inches='tight')
print(json.dumps({m: (info[m]['status'], info[m]['cls'], info[m]['iters']) for m in METHODS}),
      'k', k, 'pre', pre, 'warmup', info['ORIG']['warmup'], 'BA short', info['BA']['n_short'])
print('BA maxn', info['BA'].get('maxn'), 'ORIG maxn', info['ORIG'].get('maxn'), 'ORIG dT', info['ORIG'].get('dT'))
