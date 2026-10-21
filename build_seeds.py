"""Select the sixty seed orbits from the JPL Three-Body Periodic Orbit catalog.

Input : seeds/jpl_raw/<system>_<family>.json  -- the complete catalog families,
        downloaded from https://ssd-api.jpl.nasa.gov/periodic_orbits.api
        (sys=..., family=lyapunov&libr=1 | halo&libr=2&branch=N | dro).
Output: seeds/scan.json        amplitude / closest-approach scan of each family
        seeds/jpl_selection.json  the selected catalog rows and their neighbours
        seeds/seeds60.json     the seeds with every derived quantity

Selection rule (Methods, Seed Orbits): in each family, five members with
amplitudes spaced evenly in log(A_r) between the smallest and the largest
member that stays at least 1.05 body radii from both primaries.
"""
import json, os, sys, numpy as np, multiprocessing as mp
from cr3bp_core import trajectory, prop_state, prop_crossing, jacobi
from metrics import amplitudes, floquet, conditioning

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'seeds', 'jpl_raw')
SYSTEMS = ['earth-moon', 'saturn-titan', 'sun-earth', 'mars-phobos']
FAMILIES = ['lyapunov', 'halo', 'dro']
SYSNAME = {'earth-moon': 'Earth-Moon', 'saturn-titan': 'Saturn-Titan',
           'sun-earth': 'Sun-Earth', 'mars-phobos': 'Mars-Phobos'}
FAMNAME = {'lyapunov': 'L1 Lyapunov', 'halo': 'L2 Halo (northern)', 'dro': 'Distant retrograde'}
R_PRIMARY = {'earth-moon': 6378.0, 'saturn-titan': 60268.0, 'sun-earth': 695700.0,
             'mars-phobos': 3396.2}          # km (equatorial radii)
NSCAN = 240
CLEAR = 1.05
COND_MAX = 1e7     # catalog state rejected as a starting point above this condition number


def load(sysn, fam):
    o = json.load(open(os.path.join(RAW, f'{sysn}_{fam}.json')))
    D = np.array([[float(v) for v in row] for row in o['data']])
    return o, D


def scan_row(args):
    sysn, fam, i, row, mu, r1, r2 = args
    x0 = row[:6]; T = row[7]
    try:
        _, X = trajectory(x0, T, mu, 3000, 1e-11, 1e-11)
        if not np.all(np.isfinite(X)):
            return dict(i=i, ok=False)
        d1 = np.sqrt((X[:, 0] + mu)**2 + X[:, 1]**2 + X[:, 2]**2).min()
        d2 = np.sqrt((X[:, 0] - 1 + mu)**2 + X[:, 1]**2 + X[:, 2]**2).min()
        span = X.max(axis=0) - X.min(axis=0)
        return dict(i=i, ok=True, Ar=float(span[:3].max()), Av=float(span[3:].max()),
                    d1=float(d1/r1), d2=float(d2/r2))
    except Exception:
        return dict(i=i, ok=False)


def do_scan():
    tasks = []
    meta = {}
    for sysn in SYSTEMS:
        for fam in FAMILIES:
            o, D = load(sysn, fam)
            mu = float(o['sys']['mass_ratio']); L = float(o['sys']['lunit'])
            r1 = R_PRIMARY[sysn]/L; r2 = float(o['sys']['radius_secondary'])/L
            n = len(D)
            idx = np.unique(np.round(np.linspace(0, n - 1, min(NSCAN, n))).astype(int))
            meta[f'{sysn}|{fam}'] = dict(n=n, mu=mu, r1=r1, r2=r2)
            tasks += [(sysn, fam, int(i), D[i], mu, r1, r2) for i in idx]
    with mp.Pool(2) as pool:
        out = pool.map(scan_row, tasks, chunksize=8)
    scan = {k: dict(meta[k], rows=[]) for k in meta}
    for t, r in zip(tasks, out):
        scan[f'{t[0]}|{t[1]}']['rows'].append(r)
    json.dump(scan, open(os.path.join(HERE, 'seeds', 'scan.json'), 'w'))
    return scan


def choose(scan, sysn, fam, D, extra_exclude=()):
    S = scan[f'{sysn}|{fam}']
    ok = [r for r in S['rows'] if r['ok'] and r['d1'] >= CLEAR and r['d2'] >= CLEAR
          and r['i'] not in extra_exclude]
    A = np.array([r['Ar'] for r in ok])
    la = np.log10(A)
    targets = np.linspace(la.min(), la.max(), 5)
    picks = []
    for t in targets:
        j = int(np.argmin(np.abs(la - t) + 1e3*np.isin([r['i'] for r in ok], picks)))
        picks.append(ok[j]['i'])
    return picks, targets


def neighbours(D, i, k=8):
    sgn = np.sign(D[i, 4])
    same = np.where(np.sign(D[:, 4]) == sgn)[0]
    same = same[same != i]
    d = np.linalg.norm(D[same][:, [0, 2, 4]] - D[i, [0, 2, 4]], axis=1)
    order = same[np.argsort(d)[:k]]
    return [dict(idx=int(j), x=float(D[j, 0]), z=float(D[j, 2]), vy=float(D[j, 4]),
                 T=float(D[j, 7]), C=float(D[j, 6]), nu=float(D[j, 8])) for j in order]


def seed_props(args):
    sysn, fam, level, i, row, o, nb = args
    mu = float(o['sys']['mass_ratio'])
    x0 = row[:6].copy(); T = row[7]
    Ar, Av = amplitudes(x0, T, mu)
    xT, _, _ = prop_state(x0, T, mu, stm=False)
    fl = floquet(x0, T, mu)
    cands = [0, 4] if fam != 'halo' else [0, 2, 4]
    conds = {c: conditioning(x0, T, mu, c, Ar, Av) for c in cands}
    fix = min(cands, key=lambda c: conds[c]['condB'])
    rebased = False
    x_cat = x0.copy()
    if conds[fix]['condB'] > COND_MAX:
        # the catalog state sits at a close pass of a primary: start the seed at
        # the orbit's opposite xz-plane crossing, half a period later
        th, xh, _, st = prop_crossing(x0, T, mu, stm=False)
        assert st == 0 and abs(th/T - 0.5) < 1e-6
        x0 = xh.copy()
        conds = {c: conditioning(x0, T, mu, c, Ar, Av) for c in cands}
        fix = min(cands, key=lambda c: conds[c]['condB'])
        rebased = True
        xT, _, _ = prop_state(x0, T, mu, stm=False)
    cd = conds[fix]
    return dict(sys=sysn, family=fam, level=level, sysname=SYSNAME[sysn], famname=FAMNAME[fam],
                idx=int(i), mu=mu, lunit=float(o['sys']['lunit']), tunit=float(o['sys']['tunit']),
                x0=[float(v) for v in x0], x_cat=[float(v) for v in x_cat], rebased=rebased, T=float(T), C=float(jacobi(x0, mu)), C_jpl=float(row[6]),
                Ar=Ar, Av=Av, fix=int(fix), fix_conds={str(c): conds[c]['condB'] for c in cands},
                nu_jpl=float(row[8]), nu=fl['nu'], lmax=fl['lmax'], smax=fl['smax'],
                closure=float(np.linalg.norm(xT - x0)),
                closure_scaled=float(np.linalg.norm((xT - x0)/np.array([Ar]*3 + [Av]*3))),
                gapA=cd['gapA'], cosL=cd['cos_left'], rightnull=cd['right_null'],
                condB=cd['condB'], svB=[float(v) for v in cd['svB']],
                svA=[float(v) for v in cd['svA']], rankA=cd['rankA'], rankB=cd['rankB'], nb=nb)


if __name__ == '__main__':
    sp = os.path.join(HERE, 'seeds', 'scan.json')
    scan = json.load(open(sp)) if os.path.exists(sp) and '--rescan' not in sys.argv else do_scan()
    excl = json.load(open(os.path.join(HERE, 'seeds', 'exclude.json'))) \
        if os.path.exists(os.path.join(HERE, 'seeds', 'exclude.json')) else {}
    sel, tasks = {}, []
    for sysn in SYSTEMS:
        for fam in FAMILIES:
            o, D = load(sysn, fam)
            picks, targets = choose(scan, sysn, fam, D, excl.get(f'{sysn}|{fam}', []))
            o_meta = dict(sys=o['sys'], family=o['family'])
            rows = []
            for lev, i in enumerate(sorted(picks, key=lambda i: next(r['Ar'] for r in scan[f'{sysn}|{fam}']['rows'] if r['i'] == i)), 1):
                nb = neighbours(D, i)
                rows.append(dict(level=lev, idx=int(i), row=o['data'][i],
                                 neighbours=[dict(idx=n['idx'], row=o['data'][n['idx']]) for n in nb]))
                tasks.append((sysn, fam, lev, i, D[i], o_meta, nb))
            sel[f'{sysn}|{fam}'] = dict(system=o['sys'], targets_log10A=list(map(float, targets)), members=rows)
    json.dump(sel, open(os.path.join(HERE, 'seeds', 'jpl_selection.json'), 'w'), indent=1)
    with mp.Pool(2) as pool:
        seeds = pool.map(seed_props, tasks)
    json.dump(seeds, open(os.path.join(HERE, 'seeds', 'seeds60.json'), 'w'), indent=1)
    for s in seeds:
        print(f"{s['sysname']:13s} {s['family']:9s} L{s['level']} idx={s['idx']:5d} Ar={s['Ar']:.3e} "
              f"nu={s['nu']:.4g}/{s['nu_jpl']:.4g} clos={s['closure']:.1e} fix={s['fix']} "
              f"condB={s['condB']:.2e} gapA={s['gapA']:.1e} cosL={s['cosL']:.6f}")
