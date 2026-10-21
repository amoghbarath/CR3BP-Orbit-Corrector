"""Linearization error of the periodicity residual at each perturbation size:
e_lin = ||F(x0 + d) - (M - I) d|| / ||F(x0 + d)||  (amplitude-scaled norms),
median over the same perturbations as the campaign (first 10 trials).
eps_lin = perturbation size at which the median e_lin first exceeds 0.5
(interpolated in log eps).  Writes results/linearity.json."""
import json, numpy as np
from runlib import SEEDS, W_of
from metrics import perturb
from cr3bp_core import prop_state
EPS = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
out = []
for si, s in enumerate(SEEDS):
    x0 = np.array(s['x0']); W = W_of(s)
    _, M, _ = prop_state(x0, s['T'], s['mu'])
    xT0, _, _ = prop_state(x0, s['T'], s['mu'], stm=False)
    F0 = xT0 - x0
    med = []
    for li, e in enumerate(EPS):
        v = []
        for k in range(10):
            xp = perturb(x0, e, s['Ar'], s['Av'], np.random.default_rng([si, li, k]))
            d = xp - x0
            xT, _, st = prop_state(xp, s['T'], s['mu'], stm=False)
            if st != 0:
                v.append(np.inf); continue
            F = xT - xp
            lin = F0 + (M - np.eye(6)) @ d
            v.append(np.linalg.norm((F - lin)/W)/np.linalg.norm(F/W))
        med.append(float(np.median(v)))
    le = np.log10(EPS); m = np.array(med)
    if m[0] > 0.5:
        el, fl = EPS[0], 'below'
    elif np.all(m <= 0.5):
        el, fl = EPS[-1], 'above'
    else:
        i = int(np.argmax(m > 0.5))
        t = (np.log10(0.5) - np.log10(max(m[i-1], 1e-300)))/(np.log10(m[i]) - np.log10(max(m[i-1], 1e-300)))
        el, fl = 10**(le[i-1] + t*(le[i] - le[i-1])), 'ok'
    out.append(dict(si=si, elin=med, eps_lin=el, flag_lin=fl))
json.dump(out, open('results/linearity.json', 'w'), indent=1)
B = json.load(open('results/basin.json'))
ok = [(o, b) for o, b in zip(out, B) if o['flag_lin'] == 'ok' and b['flag_BA'] == 'ok']
x = np.log10([o['eps_lin'] for o, b in ok]); y = np.log10([b['e50_BA'] for o, b in ok])
print('n', len(ok), 'corr log eps_lin vs log eps50_BA', np.corrcoef(x, y)[0, 1], 'median ratio', np.median(10**(y - x)))
for o, b, s in zip(out, B, SEEDS):
    print(f"{s['family']:8s} {s['sysname']:12s} L{s['level']} Ar={s['Ar']:.1e} eps_lin={o['eps_lin']:.1e} {o['flag_lin']:5s} e50={b['e50_BA']:.1e} {b['flag_BA']:5s} kappa={b['kappa']:.1e}")
