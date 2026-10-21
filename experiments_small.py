"""Smaller experiments run after the main campaign.

  tol   : integrator-tolerance study (Rec. 4)            -> results/tolerance.json
  dir   : directional sensitivity on one 3-D halo (Rec. 2) -> results/directional.json
  abs   : fixed ABSOLUTE perturbation 1e-5 (the first submission's setting)
                                                        -> results/absolute.json
  orig  : reproduction of the first submission's runs (Must 11)
                                                        -> results/orig_repro.json
"""
import json, sys, time, numpy as np, multiprocessing as mp
from cr3bp_core import prop_state, propagate, aug, TOL
from correctors import correct_ba, correct_orig, correct_sdc
from metrics import perturb, amplitudes, orbit_identity
from runlib import SEEDS, TAU, run_method, classify

TOLS = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 5e-14]


# ---------------------------------------------------------------------------
def tol_task(si):
    s = SEEDS[si]; x0 = np.array(s['x0']); mu = s['mu']; T = s['T']; fix = s['fix']
    W = np.array([s['Ar']]*3 + [s['Av']]*3)
    rng = np.random.default_rng([si, 999])
    xp = perturb(x0, 1e-4, s['Ar'], s['Av'], rng)
    # reference solution at the working tolerance
    ref = correct_ba(xp, T, mu, fix, x0[fix], TAU, W)
    rows = []
    for tol in TOLS:
        Y, st, nsteps = propagate(aug(x0), T, mu, tol, tol)
        clos = float(np.linalg.norm((Y[:6] - x0)/W)) if st == 0 else float('nan')
        t0 = time.time()
        r = correct_ba(xp, T, mu, fix, x0[fix], TAU, W, rtol=tol, atol=tol)
        dt = time.time() - t0
        fl = correct_ba(xp, T, mu, fix, x0[fix], 0.0, W, maxit=10, rtol=tol, atol=tol, absfloor=0.0)
        row = dict(tol=tol, steps=int(nsteps), seed_closure=clos, status=r.status, iters=r.iters,
                   res=float(r.hist[-1]) if r.hist else float('nan'), floor=float(min(fl.hist)),
                   time=dt)
        if r.converged and ref.converged:
            row['dx0'] = float(np.linalg.norm((r.x0 - ref.x0)/W))
            row['dT'] = float(abs(r.T - ref.T)/ref.T)
            lam = np.linalg.eigvals(r.Phi); lm = np.max(np.abs(lam))
            row['nu_err'] = float(abs(0.5*(lm + 1/lm) - s['nu_jpl'])/s['nu_jpl'])
            # accuracy judged at the WORKING tolerance: re-evaluate the residual of
            # the coarse-tolerance solution with the 5e-14 integrator
            xT, _, _ = prop_state(r.x0, r.T, mu)
            row['true_res'] = float(np.linalg.norm((xT - r.x0)/W))
        rows.append(row)
    return dict(si=si, rows=rows)


# ---------------------------------------------------------------------------
def dir_task(args):
    si, comp, k, eps = args
    s = SEEDS[si]; x0 = np.array(s['x0']); mu = s['mu']; T = s['T']; fix = s['fix']
    W = np.array([s['Ar']]*3 + [s['Av']]*3)
    rng = np.random.default_rng([si, comp, k, int(-np.log10(eps))])
    xp = perturb(x0, eps, s['Ar'], s['Av'], rng, comps=[comp])
    xT, _, _ = prop_state(xp, T, mu, stm=False)
    pre = float(np.linalg.norm((xT - xp)/W))
    r = run_method('BA', s, xp)
    cls, mm = classify(s, r)
    return dict(si=si, comp=comp, k=k, eps=eps, pre=pre, status=r.status, cls=cls,
                iters=r.iters, res=float(r.hist[-1]) if r.hist else float('nan'),
                maxn=(mm['maxn'] if mm else None))


# ---------------------------------------------------------------------------
def abs_task(args):
    si, k = args
    s = SEEDS[si]; x0 = np.array(s['x0']); mu = s['mu']
    rng = np.random.default_rng([si, k, 55])
    xp = x0 + 1e-5*rng.standard_normal(6)          # first-submission setting
    out = dict(si=si, k=k, m={})
    for m in ('BA', 'SDC', 'ORIG'):
        r = run_method(m, s, xp)
        cls, mm = classify(s, r)
        out['m'][m] = dict(status=r.status, cls=cls, iters=r.iters,
                           maxn=(mm['maxn'] if mm else None))
    return out


# ---------------------------------------------------------------------------
# Seeds of the first submission (Run.m), with the first submission's
# perturbation (sigma = 1e-4 * largest span, same sigma on position and
# velocity) and stopping rule (||err|| < 1e-9, nondimensional, unscaled).
ORIG_SEEDS = {
    'Earth-Moon Halo': dict(mu=1.215058560962404E-2, T=2.7430316552716643E+0,
        x0=[8.2339046219237078E-1, 8.1862579238471623E-27, 2.3386506364385861E-3,
            -1.6087112757165459E-15, 1.2641757484149027E-1, 6.2156848285716253E-16]),
    'Mars-Phobos Halo': dict(mu=1.611081404409632E-8, T=3.0668041442732004E+0,
        x0=[9.9807079645035135E-1, 2.7874506977774413E-20, 6.0877806655169442E-4,
            2.9480636229135735E-16, 1.8275427634326100E-3, -3.0601198890259435E-16]),
    'Saturn-Titan Halo': dict(mu=2.366393158331484E-4, T=2.9897839170583227E+0,
        x0=[9.5289412551547548E-1, -4.3830648575880424E-22, 3.1412630244385099E-3,
            6.8943909471793705E-18, 3.7313904509608281E-2, -4.4508466448024921E-16]),
    'Earth-Moon Lyapunov': dict(mu=1.215058560962404E-2, T=2.6915795567917442E+0,
        x0=[8.3690888734309465E-1, 2.6569853876973453E-29, -6.5553023513506719E-32,
            6.0314258522129136E-16, 5.2232242080210143E-5, 5.1825223150917431E-31]),
    'Saturn-Titan Lyapunov': dict(mu=2.366393158331484E-4, T=2.9403115441052456E+0,
        x0=[9.5751795493492164E-1, 1.4346778162023298E-19, 7.2085108088187124E-29,
            1.6621590382582759E-16, -1.5359721527862950E-4, 9.3379461908417734E-30]),
}


def orig_task(args):
    name, k = args
    o = ORIG_SEEDS[name]; x0 = np.array(o['x0']); mu = o['mu']; T = o['T']
    from cr3bp_core import trajectory
    _, X = trajectory(x0, T, mu, 2000)
    span = (X.max(axis=0) - X.min(axis=0))[:3].max()
    sig = 1e-4*span
    rng = np.random.default_rng([k, 2026, len(name)])
    xp = x0 + sig*rng.standard_normal(6)
    r = correct_orig(xp, T, mu, 1e-9, np.ones(6))
    # same initial condition with the revised corrector (held coordinate chosen
    # by the same best-conditioning rule as the sixty seeds)
    from metrics import conditioning
    Ar, Av = amplitudes(x0, T, mu)
    W = np.array([Ar]*3 + [Av]*3)
    cands = [0, 2, 4] if 'Halo' in name else [0, 4]
    fix = min(cands, key=lambda c: conditioning(x0, T, mu, c, Ar, Av)['condB'])
    b = correct_ba(xp, T, mu, fix, x0[fix], TAU, W)
    idm = None
    if r.converged:
        try:
            idm = orbit_identity(x0, T, r.x0, r.T, mu, Ar, Av)
        except Exception:
            idm = None
    bm = None
    if b.converged:
        try:
            bm = orbit_identity(x0, T, b.x0, b.T, mu, Ar, Av)
        except Exception:
            bm = None
    return dict(name=name, k=k, span=float(span), sigma=float(sig),
                orig_status=r.status, orig_iters=r.iters, orig_res=float(r.hist[-1]),
                orig_hist=[float(h) for h in r.hist], orig_alpha=[float(a) for a in r.alpha],
                orig_T=(float(r.T) if r.converged else None),
                orig_maxn=(idm['maxn'] if idm else None), orig_dT=(idm['dT_rel'] if idm else None),
                ba_status=b.status, ba_iters=b.iters, ba_maxn=(bm['maxn'] if bm else None),
                Ar=float(Ar), fix=int(fix))


if __name__ == '__main__':
    which = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    with mp.Pool(nproc) as pool:
        if which == 'tol':
            out = pool.map(tol_task, range(len(SEEDS)))
            json.dump(out, open('results/tolerance.json', 'w'))
        elif which == 'dir':
            # Earth-Moon L2 halo, level 2 (3-D, strongly unstable): index lookup
            si = [i for i, s in enumerate(SEEDS) if s['sys'] == 'earth-moon'
                  and s['family'] == 'halo' and s['level'] == 2][0]
            tasks = [(si, c, k, e) for e in (1e-4, 1e-2) for c in range(6) for k in range(30)]
            out = pool.map(dir_task, tasks)
            json.dump(out, open('results/directional.json', 'w'))
        elif which == 'abs':
            tasks = [(si, k) for si in range(len(SEEDS)) for k in range(20)]
            out = pool.map(abs_task, tasks)
            json.dump(out, open('results/absolute.json', 'w'))
        elif which == 'orig':
            tasks = [(n, k) for n in ORIG_SEEDS for k in range(30)]
            out = pool.map(orig_task, tasks)
            json.dump(out, open('results/orig_repro.json', 'w'))
    print('done', which)
