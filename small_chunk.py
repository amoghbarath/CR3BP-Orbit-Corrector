"""Run the small experiments of experiments_small.py in resumable, time-boxed
chunks (results appended to results/<which>.jsonl; finished tasks skipped).

  python small_chunk.py dir|abs|tol|orig  [budget_seconds] [nproc]
  python small_chunk.py finalize           # jsonl -> the .json files make_results.py reads
"""
import json, os, sys, time, multiprocessing as mp
import experiments_small as E
from runlib import SEEDS

OUT = dict(tol='results/tolerance.json', dir='results/directional.json',
           abs='results/absolute.json', orig='results/orig_repro.json')


def tasks_for(which):
    if which == 'tol':
        return [(si,) for si in range(len(SEEDS))], lambda t: E.tol_task(t[0]), lambda o: (o['si'],)
    if which == 'dir':
        si = [i for i, s in enumerate(SEEDS) if s['sys'] == 'earth-moon' and s['family'] == 'halo'
              and s['level'] == 2][0]
        T = [(si, c, k, e) for e in (1e-4, 1e-2) for c in range(6) for k in range(30)]
        return T, E.dir_task, lambda o: (o['si'], o['comp'], o['k'], o['eps'])
    if which == 'abs':
        return [(si, k) for si in range(len(SEEDS)) for k in range(20)], E.abs_task, lambda o: (o['si'], o['k'])
    if which == 'orig':
        return [(n, k) for n in E.ORIG_SEEDS for k in range(30)], E.orig_task, lambda o: (o['name'], o['k'])
    if which == 'campaign':
        import numpy as np, campaign as CP
        T = [(si, li, k) for li in range(len(CP.EPS)) for si in range(len(SEEDS)) for k in range(CP.NTRIAL)]
        rng = np.random.default_rng(123); rng.shuffle(T)
        T = [tuple(int(v) for v in t) for t in T][::-1]      # from the far end of campaign.py's order
        return T, CP.task, lambda o: (o['si'], o['li'], o['k'])
    raise KeyError(which)


def _run(args):
    which, t = args
    _, fn, _ = tasks_for(which)
    return fn(t)


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'finalize':
        for w, fn in OUT.items():
            p = f'results/{w}.jsonl'
            if os.path.exists(p):
                L = [json.loads(l) for l in open(p)]
                T, _, key = tasks_for(w)
                d = {key(o): o for o in L}
                print(w, len(d), 'of', len(T))
                json.dump([d[k] for k in sorted(d, key=str)], open(fn, 'w'))
        sys.exit()
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 160
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    T, fn, key = tasks_for(which)
    p = f'results/{which}.jsonl'
    done = set()
    if os.path.exists(p):
        for l in open(p):
            done.add(key(json.loads(l)))
    todo = [t for t in T if tuple(t) not in done]
    print(which, 'todo', len(todo), 'of', len(T), flush=True)
    t0 = time.time(); n = 0
    with mp.Pool(nproc) as pool, open(p, 'a') as fh:
        it = pool.imap_unordered(_run, [(which, t) for t in todo], chunksize=1)
        try:
            while True:
                left = budget - (time.time() - t0)
                if left <= 0:
                    break
                o = it.next(timeout=left)
                fh.write(json.dumps(o) + '\n'); fh.flush(); n += 1
        except (StopIteration, mp.TimeoutError):
            pass
        pool.terminate()
    print(which, 'wrote', n, 'remaining', len(todo) - n, f'{time.time() - t0:.0f}s', flush=True)
