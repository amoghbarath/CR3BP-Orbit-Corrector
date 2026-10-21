"""Main robustness campaign: 60 JPL orbits x 7 perturbation levels x 20 trials,
all seven correctors run on identical perturbed initial conditions."""
import json, sys, os, time, numpy as np, multiprocessing as mp
EPS = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
NTRIAL = 20


def task(args):
    si, li, k = args
    from runlib import SEEDS, METHODS, run_method, classify
    from metrics import perturb, nearest_neighbor
    from cr3bp_core import prop_state
    s = SEEDS[si]; x0 = np.array(s['x0']); mu = s['mu']
    W = np.array([s['Ar']]*3 + [s['Av']]*3)
    rng = np.random.default_rng([si, li, k])
    xp = perturb(x0, EPS[li], s['Ar'], s['Av'], rng)
    xT, _, st = prop_state(xp, s['T'], mu, stm=False)
    pre = float(np.linalg.norm((xT - xp)/W)) if st == 0 else float('nan')
    pre_abs = float(np.linalg.norm(xT - xp)) if st == 0 else float('nan')
    xc = np.array(s['x_cat'])           # catalog state (same crossing as the neighbours)
    fam = np.array([[n['x'], 0, n['z'], 0, n['vy'], 0] for n in s['nb']]
                   + [[xc[0], 0, xc[2], 0, xc[4], 0]])
    out = dict(si=si, li=li, eps=EPS[li], k=k, pre=pre, pre_abs=pre_abs,
               dx=list(map(float, xp - x0)), m={})
    for m in METHODS:
        t0 = time.time()
        try:
            r = run_method(m, s, xp)
            cls, mm = classify(s, r)
        except Exception as e:
            out['m'][m] = dict(status='error', cls='failed', err=str(e))
            continue
        d = dict(status=r.status, cls=cls, iters=r.iters, nprop=r.nprop, res=float(r.hist[-1]) if r.hist else float('nan'),
                 hist=[float(h) for h in r.hist], time=time.time() - t0)
        if r.alpha:
            d['alpha'] = [float(a) for a in r.alpha]
        if r.converged:
            d['x0'] = list(map(float, r.x0)); d['T'] = float(r.T)
            if r.F is not None:
                d['res_abs'] = float(np.linalg.norm(r.F))
                d['res_r'] = float(np.linalg.norm(r.F[:3])); d['res_v'] = float(np.linalg.norm(r.F[3:]))
            if mm:
                d.update({kk: mm[kk] for kk in ('maxn', 'rmsn', 'maxr', 'maxv', 'rmsr', 'rmsv',
                                                'dT_rel', 'dC')})
            try:
                nn = nearest_neighbor(r.x0, r.T, mu, fam, len(fam) - 1)
                if nn:
                    d.update(nn_is_seed=nn['is_seed'], nn_d_seed=nn['d_seed'], nn_d2=nn['d2'])
            except Exception:
                pass
            if r.Phi is not None:
                lam = np.linalg.eigvals(r.Phi); lm = np.max(np.abs(lam))
                d['nu'] = float(0.5*(lm + 1/lm))
            if r.sv is not None and m in ('BA', 'BAP', 'BA0'):
                d['sv'] = [float(v) for v in r.sv]
        if r.condB:
            d['condB_last'] = float(r.condB[-1])
        out['m'][m] = d
    return out


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'main'
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    part = sys.argv[3] if len(sys.argv) > 3 else None      # 'i/n': run every n-th task offset i
    fn = f'results/campaign_{tag}.jsonl'
    done = set()
    if os.path.exists(fn):
        for line in open(fn):
            o = json.loads(line); done.add((o['si'], o['li'], o['k']))
    from runlib import SEEDS
    tasks = [(si, li, k) for li in range(len(EPS)) for si in range(len(SEEDS))
             for k in range(NTRIAL) if (si, li, k) not in done]
    rng = np.random.default_rng(123); rng.shuffle(tasks)
    if part:
        i, n_ = map(int, part.split('/'))
        tasks = tasks[i::n_]
    if tag.startswith('test'):
        tasks = tasks[:int(tag[4:] or 20)]
    print(len(tasks), 'tasks', flush=True)
    t0 = time.time(); n = 0
    with mp.Pool(nproc) as pool, open(fn, 'a') as fh:
        for o in pool.imap_unordered(task, tasks, chunksize=4):
            fh.write(json.dumps(o) + '\n'); fh.flush(); n += 1
            if n % 100 == 0:
                el = time.time() - t0
                print(f'{n}/{len(tasks)}  {el/60:.1f} min  eta {el/n*(len(tasks)-n)/60:.1f} min',
                      flush=True)
    print('done', flush=True)
