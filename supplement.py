"""Supplementary Data S1: per-trial results behind every figure and table.

Writes CSV files and a README into supplement/.  Every number in the
manuscript is computed from these rows by make_results.py.
"""
import csv, json, os, numpy as np
from runlib import SEEDS, METHODS

OUT = 'supplement'
os.makedirs(OUT, exist_ok=True)
EPS = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
COMP = ['x0', 'y0', 'z0', 'vx0', 'vy0', 'vz0']
MNAME = dict(BA='Bordered + line search', BA0='Bordered, no line search', ORIG='Original method',
             SDC='Standard differential correction', CHORD='Chord', FP='Fixed point',
             BAP='Bordered, projected start')


def g(v, p=6):
    if v is None:
        return ''
    if isinstance(v, float):
        if not np.isfinite(v):
            return ''
        return f'{v:.{p}g}'
    return v


basin = {b['si']: b for b in json.load(open('results/basin.json'))}
lin = {l['si']: l for l in json.load(open('results/linearity.json'))}
from cr3bp_core import prop_state
from runlib import ID_THRESH

def mult(s):
    _, M, _ = prop_state(np.array(s['x0']), s['T'], s['mu'])
    lam = np.linalg.eigvals(M)
    lam = sorted(lam, key=lambda z: (-abs(z), np.angle(z)))
    out = []
    for z in lam:
        out += [g(float(abs(z)), 10), g(float(np.degrees(np.angle(z))), 6)]
    return out


# 1. seeds -------------------------------------------------------------------
with open(f'{OUT}/S1_seeds.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['seed_id', 'system', 'family', 'member', 'jpl_index', 'mu', 'LU_km', 'TU_s',
                'x0', 'y0', 'z0', 'vx0', 'vy0', 'vz0', 'T', 'C', 'A_r_LU', 'A_v_VU', 'A_r_km',
                'held_coordinate', 'nu_jpl', 'nu_recomputed', 'max_abs_multiplier', 'seed_closure',
                'sigma5_over_sigma6_A', 'min_abs_cos_u6_gradC', 'cond_B_scaled', 'sigma7_B_scaled',
                'kappa', 'growth_median_measured', 'eps50_bordered', 'eps50_flag', 'eps_lin', 'eps_lin_flag',
                'started_at_opposite_crossing', 'x0_catalog', 'z0_catalog', 'vy0_catalog'] +
               [f'floquet_{i}_{p}' for i in range(1, 7) for p in ('modulus', 'argument_deg')])
    for i, s in enumerate(SEEDS):
        b = basin[i]
        w.writerow([i, s['sysname'], s['famname'], s['level'], s['idx'], g(s['mu'], 12), g(s['lunit'], 12),
                    g(s['tunit'], 12)] + [g(v, 16) for v in s['x0']] +
                   [g(s['T'], 16), g(s['C'], 16), g(s['Ar']), g(s['Av']), g(s['Ar']*s['lunit']),
                    COMP[s['fix']], g(s['nu_jpl'], 10), g(s['nu'], 10), g(s['lmax'], 8), g(s['closure'], 3),
                    g(s['gapA'], 3), g(s['cosL'], 17), g(s['condB'], 4), g(s['svB'][-1], 4),
                    g(b['kappa'], 4), g(b['kemp'], 4), g(b['e50_BA'], 3), b['flag_BA'],
                    g(lin[i]['eps_lin'], 3), lin[i]['flag_lin'], int(s['rebased']),
                    g(s['x_cat'][0], 16), g(s['x_cat'][2], 16), g(s['x_cat'][4], 16)] + mult(s))

# 2. main campaign ------------------------------------------------------------
R = [json.loads(l) for l in open('results/campaign_main.jsonl')]
R.sort(key=lambda o: (o['si'], o['li'], o['k']))
keys = ['status', 'cls', 'iters', 'nprop', 'res', 'res_abs', 'maxn', 'rmsn', 'dT_rel', 'dC', 'nu',
        'nn_is_seed', 'T']
PLAT = {}
with open(f'{OUT}/S1_campaign_trials.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['seed_id', 'system', 'family', 'member', 'eps', 'trial'] + ['dx_' + c for c in COMP] +
               ['start_residual_scaled', 'method', 'status', 'outcome', 'iterations', 'propagations',
                'final_residual_scaled', 'final_residual_abs', 'max_dist_scaled', 'rms_dist_scaled',
                'rel_period_diff', 'jacobi_diff', 'stability_index', 'nearest_jpl_is_seed', 'T_converged',
                'platform'])
    for o in R:
        s = SEEDS[o['si']]
        head = [o['si'], s['sysname'], s['famname'], s['level'], o['eps'], o['k']] + \
               [g(v, 8) for v in o['dx']] + [g(o['pre'], 6)]
        for m in METHODS:
            d = o['m'][m]
            w.writerow(head + [m] + [g(d.get(k), 8) if k not in ('status', 'cls') else d.get(k, '')
                                     for k in keys] + [o.get('platform', '')])

# 3. outcome summary by perturbation size (Table 3, Figure 3a) ------------------
with open(f'{OUT}/S1_summary_by_size.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['method', 'eps', 'runs', 'recovered_pct', 'different_orbit_pct', 'failed_pct',
                'iterations_median_recovered', 'iterations_q1', 'iterations_q3'])
    for m in METHODS:
        for e in EPS:
            L = [o['m'][m] for o in R if o['eps'] == e]
            n = len(L)
            rec = [d for d in L if d['cls'] == 'recovered']
            oth = [d for d in L if d['cls'] == 'other']
            it = [d['iters'] for d in rec]
            w.writerow([m, e, n, g(100*len(rec)/n, 4) if n else '', g(100*len(oth)/n, 4) if n else '',
                        g(100*(n - len(rec) - len(oth))/n, 4) if n else '',
                        g(float(np.median(it)), 4) if it else '', g(float(np.percentile(it, 25)), 4) if it else '',
                        g(float(np.percentile(it, 75)), 4) if it else ''])

# 4. directional study -----------------------------------------------------------
D = json.load(open('results/directional.json'))
with open(f'{OUT}/S1_directional.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['seed_id', 'component', 'eps', 'trial', 'start_residual_scaled', 'status', 'outcome',
                'iterations', 'final_residual_scaled', 'max_dist_scaled'])
    for d in sorted(D, key=lambda d: (d['eps'], d['comp'], d['k'])):
        w.writerow([d['si'], COMP[d['comp']], d['eps'], d['k'], g(d['pre']), d['status'], d['cls'],
                    d['iters'], g(d['res']), g(d.get('maxn'))])

# 5. fixed absolute perturbation ----------------------------------------------------
A = json.load(open('results/absolute.json'))
with open(f'{OUT}/S1_absolute_perturbation.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['seed_id', 'system', 'family', 'member', 'A_r_LU', 'trial', 'method', 'status', 'outcome',
                'iterations', 'max_dist_scaled'])
    for a in sorted(A, key=lambda a: (a['si'], a['k'])):
        s = SEEDS[a['si']]
        for m, d in a['m'].items():
            w.writerow([a['si'], s['sysname'], s['famname'], s['level'], g(s['Ar']), a['k'], m,
                        d['status'], d['cls'], d['iters'], g(d.get('maxn'))])

# 6. integrator tolerance study ------------------------------------------------------
T = json.load(open('results/tolerance.json'))
with open(f'{OUT}/S1_tolerance.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['seed_id', 'system', 'family', 'member', 'RelTol_AbsTol', 'steps_per_period',
                'seed_closure_scaled', 'residual_floor_scaled', 'status', 'iterations',
                'final_residual_scaled', 'dist_to_reference_scaled', 'rel_period_diff_to_reference',
                'stability_index_rel_err', 'residual_at_5e-14'])
    for t in sorted(T, key=lambda t: t['si']):
        s = SEEDS[t['si']]
        for r in t['rows']:
            w.writerow([t['si'], s['sysname'], s['famname'], s['level'], r['tol'], r['steps'],
                        g(r['seed_closure']), g(r['floor']), r['status'], r['iters'], g(r['res']),
                        g(r.get('dx0')), g(r.get('dT')), g(r.get('nu_err')), g(r.get('true_res'))])

# 7. reproduction of the first version's runs (per-trial iteration counts) -------------
O = json.load(open('results/orig_repro.json'))
with open(f'{OUT}/S1_first_version_reproduction.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['seed', 'trial', 'amplitude_LU', 'sigma', 'original_status', 'original_iterations',
                'original_final_residual', 'original_max_dist_scaled', 'original_rel_period_diff',
                'bordered_status', 'bordered_iterations', 'bordered_max_dist_scaled'])
    for o in sorted(O, key=lambda o: (o['name'], o['k'])):
        w.writerow([o['name'], o['k'], g(o['span']), g(o['sigma']), o['orig_status'], o['orig_iters'],
                    g(o['orig_res']), g(o['orig_maxn']), g(o['orig_dT']), o['ba_status'], o['ba_iters'],
                    g(o['ba_maxn'])])

README = """Supplementary Data S1
An Augmented Newton Shooting Method for Refining Periodic Orbits in the
Circular Restricted Three-Body Problem Across Multiple Planetary Systems

All quantities are nondimensional CR3BP units (LU, TU, VU = LU/TU) unless a
column name says otherwise. "scaled" means positions divided by the orbit's
position amplitude A_r and velocities by its velocity amplitude A_v (largest
peak-to-peak extent of the seed orbit). Every random perturbation is
reproducible: numpy.random.default_rng([seed_id, eps_index, trial]).

S1_seeds.csv
  The sixty JPL catalog seeds: initial state, period, Jacobi constant,
  amplitudes, held (family-selection) coordinate, catalog and recomputed
  stability index, all six Floquet multipliers (modulus and argument),
  conditioning of the original (A) and bordered (B) matrices, kappa (largest
  singular value of the scaled M - I), the median measured residual growth
  per unit perturbation, the half-recovery perturbation eps50 of the bordered
  corrector and the linearization limit eps_lin (flag 'below'/'above' when
  outside the tested range). Three seeds start at the orbit's opposite
  xz-plane crossing (started_at_opposite_crossing = 1); their catalog
  crossing state is listed as well. jpl_index is the 0-based row of the
  family in the JPL API response (sys=..., family=lyapunov&libr=1 |
  halo&libr=2&branch=N | dro), retrieved 11-12 October 2026.

S1_campaign_trials.csv   (Table 3, Figures 2-5)
  One row per perturbed seed and corrector: 60 orbits x 7 sizes x 20 trials
  x 7 correctors. dx_* is the perturbation added to the seed state.
  outcome = recovered (converged, max scaled distance from the seed over one
  period < 1e-4) / other (converged to a different orbit) / failed.
  platform: computer that ran the record (x86_64 or arm64).
  Methods: BA bordered + line search; BA0 bordered without line search;
  ORIG original method of the first version; SDC standard (half-period
  symmetric) differential correction; CHORD chord method; FP fixed-point
  iteration; BAP bordered corrector started from the SDC projected guess.

S1_summary_by_size.csv   (Table 3, Figure 3a)
  Outcome percentages and iteration medians/quartiles per corrector and size.

S1_directional.csv       (Directional Sensitivity)
  One component perturbed at a time, Earth-Moon L2 Halo (seed_id 6, nu = 488),
  thirty trials per component at eps = 1e-4 and 1e-2.

S1_absolute_perturbation.csv
  Fixed absolute perturbation sigma = 1e-5 on every component (the first
  version's setting), all sixty orbits, twenty trials each.

S1_tolerance.csv         (Figure 6)
  Integrator tolerance study: steps per period, residual floor, outcome,
  and distance of the converged orbit from the 5e-14 reference.

S1_first_version_reproduction.csv
  The first version's runs reproduced with its seeds, perturbation
  (sigma = 1e-4 A, thirty trials per seed) and stopping rule: per-trial
  iteration counts of the original method (Must-fix comment 11), with the
  bordered corrector on the same starts for comparison.
"""
open(f'{OUT}/README.txt', 'w').write(README)
print('rows:', len(R) * len(METHODS), 'campaign;', len(D), 'directional;', len(A), 'absolute;',
      len(T), 'tolerance seeds;', len(O), 'reproduction')
