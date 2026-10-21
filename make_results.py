"""Turn the raw campaign output into every number, table and figure used in
the revised manuscript.  Writes results/numbers.json (all quoted numbers),
results/tables.json (LaTeX table bodies) and figs/*.png."""
import json, numpy as np, collections, math
from figstyle import *
from runlib import SEEDS, ID_THRESH
from cr3bp_core import prop_state
from metrics import pair_multipliers

EPS = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
MET = ['BA', 'BA0', 'SDC', 'BAP', 'ORIG', 'CHORD', 'FP']
R = [json.loads(l) for l in open('results/campaign_main.jsonl')]
NUM = {}
TAB = {}
NUM['n_records'] = len(R)
NUM['n_corrections'] = sum(len(o['m']) for o in R)


def pct(a, b):
    return 100.0*a/b if b else float('nan')


def fmt_e(v, d=1):
    if v is None or not np.isfinite(v):
        return '--'
    if v == 0:
        return '0'
    e = int(math.floor(math.log10(abs(v))))
    m = v/10**e
    if round(m, d) >= 10:
        m /= 10; e += 1
    return f'{m:.{d}f}\\times10^{{{e}}}'


# ---------------------------------------------------------------------------
# per-orbit quantities: kappa = largest singular value of the amplitude-
# scaled (M - I); lambda_max; nu
# ---------------------------------------------------------------------------
for si, s in enumerate(SEEDS):
    x0 = np.array(s['x0']); W = np.array([s['Ar']]*3 + [s['Av']]*3)
    _, M, _ = prop_state(x0, s['T'], s['mu'])
    K = (M - np.eye(6))*W[None, :]/W[:, None]
    s['kappa'] = float(np.linalg.svd(K, compute_uv=False)[0])
    s['M'] = M

# ---------------------------------------------------------------------------
# outcome counts by method x eps
# ---------------------------------------------------------------------------
cnt = collections.defaultdict(collections.Counter)
its = collections.defaultdict(list)
for o in R:
    for m, d in o['m'].items():
        cnt[(m, o['li'])][d['cls']] += 1
        if d['cls'] == 'recovered':
            its[(m, o['li'])].append(d['iters'])
for m in MET:
    for li, e in enumerate(EPS):
        c = cnt[(m, li)]; n = sum(c.values())
        NUM[f'rec_{m}_{li}'] = pct(c['recovered'], n)
        NUM[f'oth_{m}_{li}'] = pct(c['other'], n)
        NUM[f'fail_{m}_{li}'] = pct(c['failed'], n)
        NUM[f'n_{m}_{li}'] = n
        it = its[(m, li)]
        if it:
            NUM[f'itmed_{m}_{li}'] = float(np.median(it))
            NUM[f'itq1_{m}_{li}'] = float(np.percentile(it, 25))
            NUM[f'itq3_{m}_{li}'] = float(np.percentile(it, 75))
            NUM[f'itmin_{m}_{li}'] = int(np.min(it)); NUM[f'itmax_{m}_{li}'] = int(np.max(it))

# Table: method comparison at three perturbation sizes
rows = []
for m in MET:
    cells = [MSHORT[m]]
    for li in (0, 2, 4):
        c = cnt[(m, li)]; n = sum(c.values())
        it = its[(m, li)]
        itx = (f"{np.median(it):.0f} [{np.percentile(it,25):.0f}, {np.percentile(it,75):.0f}]"
               if it else '--')
        cells.append(f"{pct(c['recovered'], n):.0f} / {pct(c['other'], n):.0f} / {pct(c['failed'], n):.0f}")
        cells.append(itx)
    rows.append(' & '.join(cells) + r' \\')
TAB['methods'] = '\n'.join(rows)

# family breakdown for BA and SDC
for m in ('BA', 'SDC', 'ORIG', 'BAP'):
    for fam in FAMORDER:
        for li in range(7):
            L = [o['m'][m]['cls'] for o in R if o['li'] == li and SEEDS[o['si']]['family'] == fam]
            NUM[f'recfam_{m}_{fam}_{li}'] = pct(sum(x == 'recovered' for x in L), len(L))

# ---------------------------------------------------------------------------
# initial residual: recovery vs ||F0|| pooled over all orbits
# ---------------------------------------------------------------------------
bins = np.array([0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, np.inf])
centers = np.sqrt(np.array([3e-4, 3e-3, 1.7e-2, 5.5e-2, 1.7e-1, 5.5e-1, 1.7, 5.5, 20]))
centers = np.array([3e-4, 3e-3, 1.7e-2, 5.5e-2, 1.7e-1, 5.5e-1, 1.7, 5.5, 20])
recF = {}
for m in MET:
    p, nn = [], []
    for b in range(len(bins) - 1):
        L = [o['m'][m]['cls'] for o in R if np.isfinite(o['pre']) and bins[b] <= o['pre'] < bins[b+1]]
        p.append(pct(sum(x == 'recovered' for x in L), len(L))); nn.append(len(L))
    recF[m] = (np.array(p), np.array(nn))
    for b in range(len(bins) - 1):
        NUM[f'recF_{m}_{b}'] = p[b]
NUM['recF_bins'] = [float(b) for b in bins[:-1]]
# BA: share recovered when ||F0|| < 0.3 / < 0.1
for thr, tag in ((0.1, '01'), (0.3, '03'), (1.0, '1')):
    for m in MET:
        L = [o['m'][m]['cls'] for o in R if np.isfinite(o['pre']) and o['pre'] < thr]
        NUM[f'recbelow{tag}_{m}'] = pct(sum(x == 'recovered' for x in L), len(L))
    for m in MET:
        L = [o['m'][m]['cls'] for o in R if np.isfinite(o['pre']) and o['pre'] >= thr]
        NUM[f'recabove{tag}_{m}'] = pct(sum(x == 'recovered' for x in L), len(L))

# ---------------------------------------------------------------------------
# per-orbit basin: eps50 for BA and SDC (first crossing of 50 % recovery,
# interpolated in log eps); amplification kappa_emp = median(||F0||/eps)
# ---------------------------------------------------------------------------
def eps50(si, m):
    p = []
    for li in range(7):
        L = [o['m'][m]['cls'] for o in R if o['si'] == si and o['li'] == li]
        p.append(np.mean([x == 'recovered' for x in L]) if L else np.nan)
    p = np.array(p)
    le = np.log10(EPS)
    if np.isnan(p).any():
        return np.nan, p, 'na'
    if p[0] < 0.5:
        return EPS[0], p, 'below'
    for i in range(1, 7):
        if p[i] < 0.5:
            t = (p[i-1] - 0.5)/(p[i-1] - p[i])
            return 10**(le[i-1] + t*(le[i] - le[i-1])), p, 'ok'
    return EPS[-1], p, 'above'

basin = []
for si, s in enumerate(SEEDS):
    rec = dict(si=si)
    for m in ('BA', 'SDC', 'ORIG', 'BAP'):
        e50, p, flag = eps50(si, m)
        rec[f'e50_{m}'] = e50; rec[f'flag_{m}'] = flag; rec[f'p_{m}'] = p.tolist()
    amp = [o['pre']/o['eps'] for o in R if o['si'] == si and o['li'] <= 1 and np.isfinite(o['pre'])]
    rec['kemp'] = float(np.median(amp)) if amp else np.nan
    rec['kappa'] = s['kappa']; rec['Ar'] = s['Ar']; rec['lmax'] = s['lmax']; rec['nu'] = s['nu']
    basin.append(rec)
json.dump(basin, open('results/basin.json', 'w'), indent=1)
ok = [b for b in basin if b['flag_BA'] == 'ok']
le50 = np.log10([b['e50_BA'] for b in ok])
NUM['basin_n_ok_BA'] = len(ok)
NUM['basin_n_below_BA'] = sum(b['flag_BA'] == 'below' for b in basin)
NUM['basin_n_above_BA'] = sum(b['flag_BA'] == 'above' for b in basin)
if len(ok) > 3:
    for key, name in (('kappa', 'kappa'), ('Ar', 'Ar'), ('lmax', 'lmax'), ('kemp', 'kemp')):
        x = np.log10([b[key] for b in ok])
        NUM[f'corr_e50_{name}'] = float(np.corrcoef(x, le50)[0, 1])
    prod = np.array([b['e50_BA']*b['kappa'] for b in ok])
    NUM['e50kappa_med'] = float(np.median(prod))
    NUM['e50kappa_q1'] = float(np.percentile(prod, 25)); NUM['e50kappa_q3'] = float(np.percentile(prod, 75))
    # absolute basin delta50 = eps50 * Ar (LU) spread vs relative spread
    d50 = np.array([b['e50_BA']*b['Ar'] for b in ok])
    NUM['e50_span_dec'] = float(np.log10(max(b['e50_BA'] for b in ok)/min(b['e50_BA'] for b in ok)))
    NUM['d50_span_dec'] = float(np.log10(d50.max()/d50.min()))
    NUM['e50kappa_span_dec'] = float(np.log10(prod.max()/prod.min()))
kemp = np.array([b['kemp'] for b in basin]); kap = np.array([b['kappa'] for b in basin])
good = np.isfinite(kemp)
NUM['corr_kemp_kappa'] = float(np.corrcoef(np.log10(kemp[good]), np.log10(kap[good]))[0, 1])
NUM['ratio_kemp_kappa_med'] = float(np.median(kemp[good]/kap[good]))
NUM['kappa_min'] = float(kap.min()); NUM['kappa_max'] = float(kap.max())
for fam in FAMORDER:
    kk = [b['kappa'] for b, s in zip(basin, SEEDS) if s['family'] == fam]
    NUM[f'kappa_min_{fam}'] = float(min(kk)); NUM[f'kappa_max_{fam}'] = float(max(kk))

# ---------------------------------------------------------------------------
# orbit identity
# ---------------------------------------------------------------------------
def collect(m, li=None, cls=None):
    out = []
    for o in R:
        if li is not None and o['li'] != li:
            continue
        d = o['m'][m]
        if cls is not None and d['cls'] != cls:
            continue
        if d['status'] == 'converged' and 'maxn' in d:
            out.append((o, d))
    return out

for m in ('BA', 'SDC', 'ORIG'):
    conv = collect(m)
    rec = [d for o, d in conv if d['cls'] == 'recovered']
    oth = [d for o, d in conv if d['cls'] == 'other']
    NUM[f'id_nconv_{m}'] = len(conv)
    NUM[f'id_nrec_{m}'] = len(rec); NUM[f'id_noth_{m}'] = len(oth)
    if rec:
        NUM[f'id_rec_maxn_med_{m}'] = float(np.median([d['maxn'] for d in rec]))
        NUM[f'id_rec_maxn_max_{m}'] = float(np.max([d['maxn'] for d in rec]))
        NUM[f'id_rec_rmsn_med_{m}'] = float(np.median([d['rmsn'] for d in rec]))
        NUM[f'id_rec_dT_med_{m}'] = float(np.median([d['dT_rel'] for d in rec]))
        NUM[f'id_rec_dT_max_{m}'] = float(np.max([d['dT_rel'] for d in rec]))
        NUM[f'id_rec_dC_med_{m}'] = float(np.median([d['dC'] for d in rec]))
        NUM[f'id_rec_dC_max_{m}'] = float(np.max([d['dC'] for d in rec]))
        nn = [d for d in rec if 'nn_is_seed' in d]
        NUM[f'id_rec_nnseed_{m}'] = pct(sum(d['nn_is_seed'] for d in nn), len(nn))
        NUM[f'id_rec_nn_ratio_max_{m}'] = float(np.max([d['nn_d_seed']/d['nn_d2'] for d in nn]))
        NUM[f'id_rec_nn_ratio_med_{m}'] = float(np.median([d['nn_d_seed']/d['nn_d2'] for d in nn]))
    if oth:
        NUM[f'id_oth_maxn_min_{m}'] = float(np.min([d['maxn'] for d in oth]))
        NUM[f'id_oth_maxn_med_{m}'] = float(np.median([d['maxn'] for d in oth]))
        NUM[f'id_oth_dT_med_{m}'] = float(np.median([d['dT_rel'] for d in oth]))
        nn = [d for d in oth if 'nn_is_seed' in d]
        NUM[f'id_oth_nnseed_{m}'] = pct(sum(d['nn_is_seed'] for d in nn), len(nn))
    allm = [d['maxn'] for o, d in conv]
    gap_lo = max([d['maxn'] for d in rec]) if rec else np.nan
    gap_hi = min([d['maxn'] for d in oth]) if oth else np.nan
    NUM[f'id_gap_lo_{m}'] = float(gap_lo); NUM[f'id_gap_hi_{m}'] = float(gap_hi)
    # converged runs in the band between the closure-limited cluster and the threshold
    NUM[f'id_band_{m}'] = sum(ID_THRESH <= v < 1e-1 for v in allm)
    NUM[f'id_rec_above1e6_{m}'] = pct(sum(d['maxn'] > 1e-6 for d in rec), len(rec)) if rec else float('nan')

# stability index of corrected orbits vs JPL (BA, recovered)
nuerr = []
for o, d in collect('BA', cls='recovered'):
    if 'nu' in d:
        s = SEEDS[o['si']]
        nuerr.append(abs(d['nu'] - s['nu_jpl'])/s['nu_jpl'])
NUM['nu_corr_relerr_med'] = float(np.median(nuerr)); NUM['nu_corr_relerr_max'] = float(np.max(nuerr))
nus = [abs(s['nu'] - s['nu_jpl'])/s['nu_jpl'] for s in SEEDS]
NUM['nu_seed_relerr_max'] = float(np.max(nus)); NUM['nu_seed_relerr_med'] = float(np.median(nus))
NUM['seed_closure_max'] = float(max(s['closure'] for s in SEEDS))
NUM['seed_closure_med'] = float(np.median([s['closure'] for s in SEEDS]))

# unperturbed seeds: how many already satisfy the tolerance (zero iterations)
zero = 0
for s in SEEDS:
    x0 = np.array(s['x0']); W = np.array([s['Ar']]*3 + [s['Av']]*3)
    xT, _, _ = prop_state(x0, s['T'], s['mu'], stm=False)
    F = xT - x0
    if np.linalg.norm(F/W) < 1e-8 or np.linalg.norm(F) < 2e-12:
        zero += 1
NUM['seeds_zero_iter'] = zero

# conditioning summary
gA = [s['gapA'] for s in SEEDS]; cl = [s['cosL'] for s in SEEDS]; cB = [s['condB'] for s in SEEDS]
NUM['gapA_min'] = float(min(gA)); NUM['gapA_max'] = float(max(gA))
NUM['cosL_min'] = float(min(cl))
NUM['condB_min'] = float(min(cB)); NUM['condB_max'] = float(max(cB))
NUM['condB_med'] = float(np.median(cB))
NUM['rightnull_max'] = float(max(s['rightnull'] for s in SEEDS))
# near convergence: scaled condition number of B at the last BA iteration (recovered runs, eps=1e-4)
cbl = [d['condB_last'] for o, d in collect('BA', li=0, cls='recovered') if 'condB_last' in d]
svl = [min(d['sv']) for o, d in collect('BA', li=0, cls='recovered') if 'sv' in d]
NUM['condB_conv_min'] = float(min(cbl)); NUM['condB_conv_max'] = float(max(cbl))
NUM['svmin_conv_min'] = float(min(svl))

json.dump(NUM, open('results/numbers.json', 'w'), indent=1)
json.dump(TAB, open('results/tables.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in NUM.items() if not k.startswith(('rec_', 'oth_', 'fail_', 'n_', 'it', 'recfam', 'recF_'))}, indent=0)[:6000])

# ===========================================================================
# extra numbers and tables
# ===========================================================================
from correctors import backslash_basic
from cr3bp_core import f6
from metrics import perturb
NUM = json.load(open('results/numbers.json'))

# original method: first-step basic solution zeroes dT
nz = 0
for si, s in enumerate(SEEDS):
    x0 = np.array(s['x0']); rng = np.random.default_rng([si, 0, 0])
    xp = perturb(x0, 1e-4, s['Ar'], s['Av'], rng)
    xT, Phi, _ = prop_state(xp, s['T'], s['mu'])
    A = np.hstack([Phi - np.eye(6), f6(xT, s['mu'])[:, None]])
    d, r = backslash_basic(A, xp - xT)
    nz += int(d[6] == 0.0)
NUM['orig_dT0_first'] = nz
L = [o['m']['ORIG'] for o in R if o['m']['ORIG']['status'] == 'converged' and 'T' in o['m']['ORIG']]
NUM['orig_dTzero_pct'] = pct(sum(d.get('dT_rel', 1) == 0.0 for d in L), len(L))

# worked example
ex = json.load(open('results/example_info.json'))
NUM['ex_BA_maxn'] = ex['BA']['maxn']; NUM['ex_ORIG_dT'] = ex['ORIG']['dT']; NUM['ex_ORIG_maxn'] = ex['ORIG']['maxn']
NUM['ex_pre'] = ex['pre']; NUM['ex_nu'] = ex['nu']
for m_ in ('BA', 'BA0', 'SDC', 'BAP', 'ORIG', 'CHORD', 'FP'):
    NUM[f'ex_{m_}_iters'] = ex[m_]['iters']
NUM['ex_BA_nshort'] = ex['BA']['n_short']; NUM['ex_ORIG_warmup'] = ex['ORIG']['warmup']
NUM['ex_BA_nfull'] = ex['BA']['iters'] - ex['BA']['n_short']
NUM['ex_BA_amin'] = min(ex['BA']['alpha']); NUM['ex_BA_amax_short'] = max(a for a in ex['BA']['alpha'] if a < 1)
NUM['ex_FP_resmed'] = float(np.median(ex['FP']['hist'][1:]))

# nearest-neighbour ratio
NUM['nnratio_inv'] = 1.0/NUM['id_rec_nn_ratio_max_BA']

# mirror-image halos among BA 'other' outcomes
mir = 0
for o in R:
    d = o['m']['BA']
    s = SEEDS[o['si']]
    if d['cls'] == 'other' and s['family'] == 'halo' and d.get('dT_rel', 1) < 1e-9 and \
            np.sign(d['x0'][2]) != np.sign(s['x0'][2]):
        mir += 1
NUM['mirror_n'] = mir

# 'different' BA outcomes that lie close to the threshold (accuracy-limited, not new orbits)
near = [(o, o['m']['BA']) for o in R if o['m']['BA']['cls'] == 'other' and o['m']['BA'].get('maxn', 1) < 1e-3]
far = [o['m']['BA']['maxn'] for o in R if o['m']['BA']['cls'] == 'other' and o['m']['BA'].get('maxn', 0) >= 1e-3]
NUM['id_near_n_BA'] = len(near)
NUM['id_near_max_BA'] = max([d['maxn'] for o, d in near]) if near else 0.0
NUM['id_near_dT_max_BA'] = max([d['dT_rel'] for o, d in near]) if near else 0.0
NUM['id_near_seeds'] = sorted({o['si'] for o, d in near})
NUM['id_near_sv7'] = min(SEEDS[si]['svB'][-1] for si in NUM['id_near_seeds']) if near else 0.0
NUM['id_far_min_BA'] = float(min(far)) if far else float('nan')
NUM['svB_min_seed'] = float(min(s['svB'][-1] for s in SEEDS))

# up to which perturbation size is standard DC the most robust corrector?
EPSL = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
best = [all(NUM[f'rec_SDC_{li}'] >= NUM[f'rec_{m}_{li}'] for m in ('BA', 'BA0', 'ORIG', 'CHORD', 'FP', 'BAP'))
        for li in range(7)]
k = 0
while k < 7 and best[k]:
    k += 1
NUM['sdc_best_all'] = int(all(best))
NUM['sdc_best_upto_li'] = k - 1
NUM['sdc_best_upto'] = ['10^{-4}', '10^{-3}', '10^{-2}', '3\\times10^{-2}', '10^{-1}', '0.3', '1'][max(k - 1, 0)]
NUM['sdc_best_list'] = best

# stability index ranges by family
NUM['nu_lyap_min'] = min(s['nu'] for s in SEEDS if s['family'] == 'lyapunov')
NUM['nu_lyap_max'] = max(s['nu'] for s in SEEDS if s['family'] == 'lyapunov')
NUM['nu_halo_max'] = max(s['nu'] for s in SEEDS if s['family'] == 'halo')

# tolerance study
T = json.load(open('results/tolerance.json'))
tols = [r['tol'] for r in T[0]['rows']]
conv_ref = [t for t in T if t['rows'][-1]['status'] == 'converged']
NUM['tol_n_ref_conv'] = len(conv_ref)
ok_tols = [tol for k, tol in enumerate(tols) if all(t['rows'][k]['status'] == 'converged' for t in conv_ref)]
NUM['tol_conv_all'] = max(ok_tols)
for k, tol in enumerate(tols):
    dx = [t['rows'][k].get('dx0') for t in conv_ref if t['rows'][k].get('dx0') is not None]
    NUM[f'tol_dx_{tol:.0e}'] = float(max(dx)) if dx else float('nan')
    NUM[f'tol_steps_{tol:.0e}'] = float(np.median([t['rows'][k]['steps'] for t in T]))
NUM['tol_dx_1e10'] = NUM['tol_dx_1e-10']; NUM['tol_dx_1e8'] = NUM['tol_dx_1e-08']
NUM['tol_step_ratio'] = NUM['tol_steps_1e-10']/NUM['tol_steps_5e-14']

# absolute-perturbation study
A_ = json.load(open('results/absolute.json'))
def recp(m, sel):
    L = [a['m'][m]['cls'] for a in A_ if sel(SEEDS[a['si']])]
    return pct(sum(x == 'recovered' for x in L), len(L))
def othp(m, sel):
    L = [a['m'][m]['cls'] for a in A_ if sel(SEEDS[a['si']])]
    return pct(sum(x == 'other' for x in L), len(L))
small = lambda s: s['Ar'] < 1e-3
NUM['abs_rec_BA'] = recp('BA', lambda s: True)
NUM['abs_rec_BA_small'] = recp('BA', small)
NUM['abs_rec_BA_large'] = recp('BA', lambda s: not small(s))
NUM['abs_n_small'] = sum(small(s) for s in SEEDS)
NUM['abs_oth_ORIG'] = othp('ORIG', lambda s: True)
NUM['abs_rec_ORIG'] = recp('ORIG', lambda s: True)
NUM['abs_rec_SDC'] = recp('SDC', lambda s: True)
NUM['abs_rec_SDC_small'] = recp('SDC', small)
NUM['abs_oth_ORIG_small'] = othp('ORIG', small)
NUM['abs_rec_ORIG_small'] = recp('ORIG', small)

# cross-platform reproducibility: tasks run on both computers
import gzip
def _cls(d):
    if d.get('status') != 'converged':
        return 'failed'
    return 'recovered' if d.get('maxn', 1) < ID_THRESH else 'other'
Cc = {(o['si'], o['li'], o['k']): o for o in map(json.loads, open('results/campaign_container.jsonl'))}
Mc = {(o['si'], o['li'], o['k']): o for o in map(json.loads, gzip.open('results/campaign_mac.jsonl.gz', 'rt'))}
both = sorted(set(Cc) & set(Mc))
agree = [(_cls(Cc[k]['m'][m]) == _cls(Mc[k]['m'][m])) for k in both for m in Cc[k]['m']]
NUM['overlap_n'] = len(both); NUM['overlap_ncorr'] = len(agree); NUM['overlap_agree'] = pct(sum(agree), len(agree))

# linearization limit (linearity.py)
LIN = json.load(open('results/linearity.json'))
okl = [(l, b) for l, b in zip(LIN, json.load(open('results/basin.json'))) if l['flag_lin'] == 'ok' and b['flag_BA'] == 'ok']
xl = np.log10([l['eps_lin'] for l, b in okl]); yl = np.log10([b['e50_BA'] for l, b in okl])
NUM['lin_n'] = len(okl)
NUM['corr_e50_elin'] = float(np.corrcoef(xl, yl)[0, 1])
NUM['ratio_e50_elin_med'] = float(np.median(10**(yl - xl)))
NUM['ratio_e50_elin_q1'] = float(np.percentile(10**(yl - xl), 25)); NUM['ratio_e50_elin_q3'] = float(np.percentile(10**(yl - xl), 75))
NUM['lin_flags_match'] = int(sum((l['flag_lin'] == b['flag_BA']) for l, b in zip(LIN, json.load(open('results/basin.json'))) if l['flag_lin'] != 'ok'))
NUM['lin_flags_n'] = int(sum(l['flag_lin'] != 'ok' for l in LIN))
# eps50*kappa outside the near-linear small Lyapunov orbits
B_ = json.load(open('results/basin.json'))
big = [b['e50_BA']*b['kappa'] for b, s in zip(B_, SEEDS) if b['flag_BA'] == 'ok' and b['e50_BA']*b['kappa'] < 20]
NUM['e50kappa_n_lt20'] = len(big)
NUM['e50kappa_big_n'] = sum(1 for b in B_ if b['flag_BA'] == 'ok' and b['e50_BA']*b['kappa'] >= 20)
NUM['e50kappa_big_max'] = max(b['e50_BA']*b['kappa'] for b in B_)
lyap_big = [s for b, s in zip(B_, SEEDS) if b['e50_BA']*b['kappa'] >= 20]
NUM['e50kappa_big_fams'] = sorted({s['family'] for s in lyap_big})
NUM['e50kappa_big_Armax'] = max(s['Ar'] for s in lyap_big)
from scipy.optimize import brentq
def _gamma1(mu):
    return brentq(lambda g: (1 - mu)/(1 - g)**2 - mu/g**2 - (1 - mu - g), 1e-7, 0.9)
NUM['e50kappa_big_rmax'] = max(s['Ar']/_gamma1(s['mu']) for s in lyap_big)
x = np.log10([b['kappa'] for b in B_ if b['flag_BA'] == 'ok' and b['e50_BA']*b['kappa'] < 20])
y = np.log10([b['e50_BA'] for b in B_ if b['flag_BA'] == 'ok' and b['e50_BA']*b['kappa'] < 20])
NUM['corr_e50_kappa_excl'] = float(np.corrcoef(x, y)[0, 1])
NUM['e50kappa_excl_med'] = float(np.median(10**(x + y)))
NUM['e50kappa_excl_q1'] = float(np.percentile(10**(x + y), 25)); NUM['e50kappa_excl_q3'] = float(np.percentile(10**(x + y), 75))

# tolerance study: per-tolerance convergence counts and accuracy
T_ = json.load(open('results/tolerance.json'))
tl = [r['tol'] for r in T_[0]['rows']]
refc = [t for t in T_ if t['rows'][-1]['status'] == 'converged']
for k, tol in enumerate(tl):
    rows = [t['rows'][k] for t in refc]
    cv = [r for r in rows if r['status'] == 'converged']
    tag = f'{tol:.0e}'
    NUM[f'tol_nconv_{tag}'] = len(cv)
    NUM[f'tol_floor_med_{tag}'] = float(np.median([r['floor'] for r in rows]))
    NUM[f'tol_trueres_max_{tag}'] = float(max(r['true_res'] for r in cv)) if cv else float('nan')
NUM['tol_nref'] = len(refc)
NUM['Ar_min_LU'] = min(s['Ar'] for s in SEEDS); NUM['Ar_max_LU'] = max(s['Ar'] for s in SEEDS)
_smin = min(SEEDS, key=lambda s: s['Ar']); _smax = max(SEEDS, key=lambda s: s['Ar'])
NUM['Ar_min_km'] = _smin['Ar']*_smin['lunit']; NUM['Ar_min_name'] = f"{_smin['sysname']} {_smin['famname']}"
NUM['Ar_max_name'] = f"{_smax['sysname']} {_smax['famname']}"
NUM['n_catalog'] = 33860

# reproduction of the first submission
O = json.load(open('results/orig_repro.json'))
by = collections.defaultdict(list)
for o in O:
    by[o['name']].append(o)
emh = [o['orig_iters'] for o in by['Earth-Moon Halo'] if o['orig_status'] == 'converged']
NUM['orep_EMH_min'] = int(min(emh)); NUM['orep_EMH_max'] = int(max(emh)); NUM['orep_EMH_med'] = float(np.median(emh))
oth = [o['orig_iters'] for n, L in by.items() if n != 'Earth-Moon Halo' for o in L if o['orig_status'] == 'converged']
NUM['orep_other_range'] = f"{min(oth)} to {max(oth)}"
allo = [o for o in O if o['orig_maxn'] is not None]
NUM['orep_other_pct'] = pct(sum(o['orig_maxn'] > ID_THRESH for o in allo), len(allo))
NUM['orep_n_maxn'] = len(allo)
NUM['orep_EMH_span'] = by['Earth-Moon Halo'][0]['span']
NUM['orep_ratio'] = by['Earth-Moon Halo'][0]['span']/by['Earth-Moon Lyapunov'][0]['span']
NUM['orep_conv'] = pct(sum(o['orig_status'] == 'converged' for o in O), len(O))
NUM['orep_ba_med'] = float(np.median([o['ba_iters'] for o in O if o['ba_status'] == 'converged']))
for n, L in by.items():
    tag = ''.join(w[0] for w in n.replace('-', ' ').split())
    it = [o['orig_iters'] for o in L if o['orig_status'] == 'converged']
    NUM[f'orep_{tag}_med'] = float(np.median(it)); NUM[f'orep_{tag}_rng'] = f'{min(it)}--{max(it)}'

# ---- tables -------------------------------------------------------------
TAB = json.load(open('results/tables.json'))
# Table: systems
rows = []
for sy in SYSORDER:
    ss = [s for s in SEEDS if s['sys'] == sy]
    LU = ss[0]['lunit']; TU = ss[0]['tunit']; VU = LU/TU*1e3   # m/s
    Ar = sorted(s['Ar']*LU for s in ss)
    rows.append(f"{SYSNAME[sy]} & ${fmt_e(ss[0]['mu'], 2)}$ & {LU:,.0f} & {TU/86400:.3g} & "
                f"${fmt_e(Ar[0], 1)}$ -- ${fmt_e(Ar[-1], 1)}$ & ${fmt_e(1e-8*Ar[0]*1e3, 1)}$ -- ${fmt_e(1e-8*Ar[-1]*1e3, 1)}$ & "
                f"${fmt_e(1e-8*min(s['Av'] for s in ss)*VU*1e3, 1)}$ -- ${fmt_e(1e-8*max(s['Av'] for s in ss)*VU*1e3, 1)}$ \\\\")
TAB['systems'] = ('\\small\n\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccccc}\n\\toprule\n'
                  '\\textbf{System} & $\\mu$ & \\textbf{LU (km)} & \\textbf{TU (days)} & '
                  '\\textbf{$A_r$ (km)} & \\textbf{Position (m)} & \\textbf{Velocity (mm/s)} \\\\\n\\midrule\n'
                  + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}}')
# Table: conditioning (ranges by family)
rows = []
for fa in FAMORDER:
    ss = [s for s in SEEDS if s['family'] == fa]
    g = [s['gapA'] for s in ss]; c = [s['cosL'] for s in ss]; b = [s['condB'] for s in ss]
    sB = [s['svB'][-1] for s in ss]
    rA = sorted({int(np.sum(np.array(s['svA']) > 1e-8*s['svA'][0])) for s in ss})
    rB = sorted({int(np.sum(np.array(s['svB']) > 1e-8*s['svB'][0])) for s in ss})
    rows.append(f"{FAMNAME[fa]} & ${fmt_e(min(g))}$ -- ${fmt_e(max(g))}$ & {min(c):.6f} & {'/'.join(map(str, rA))} & "
                f"${fmt_e(min(b))}$ -- ${fmt_e(max(b))}$ & {'/'.join(map(str, rB))} \\\\")
TAB['cond'] = ('\\small\n\\begin{tabular}{lccccc}\n\\toprule\n'
               '& \\multicolumn{3}{c}{\\textbf{Original} $\\mathbf{A}$ ($6\\times7$)} & '
               '\\multicolumn{2}{c}{\\textbf{Bordered} $\\mathbf{B}$ ($8\\times7$)} \\\\\n'
               '\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n'
               '\\textbf{Family} & $\\sigma_5/\\sigma_6$ & $\\min|\\cos(\\mathbf{u}_6, \\nabla C)|$ & \\textbf{Rank} & '
               '$\\sigma_1/\\sigma_7$ & \\textbf{Rank} \\\\\n\\midrule\n' + '\n'.join(rows) +
               '\n\\bottomrule\n\\end{tabular}')
# Table: directional
D = json.load(open('results/directional.json'))
names = ['$x_0$', '$y_0$', '$z_0$', '$\\dot{x}_0$', '$\\dot{y}_0$', '$\\dot{z}_0$']
rows = []
for c in range(6):
    cells = [names[c]]
    for e in (1e-4, 1e-2):
        L = [d for d in D if d['comp'] == c and d['eps'] == e]
        rec = [d for d in L if d['cls'] == 'recovered']
        cells += [f"${fmt_e(np.mean([d['pre'] for d in L]))}$",
                  f"{np.mean([d['iters'] for d in rec]):.1f}" if rec else '--',
                  f"{len(rec)}/{len(L)}"]
    rows.append(' & '.join(cells) + ' \\\\')
TAB['dir'] = '\n'.join(rows)
sdir = SEEDS[D[0]['si']]
NUM['dir_nu'] = sdir['nu']; NUM['dir_level'] = sdir['level']
Wd = np.array([sdir['Ar']]*3 + [sdir['Av']]*3)
_, Md, _ = prop_state(np.array(sdir['x0']), sdir['T'], sdir['mu'])
lamL, UL = np.linalg.eig((Md*Wd[None, :]/Wd[:, None]).T)
uL = np.abs(np.real(UL[:, np.argmax(np.abs(lamL))]))
pre = [np.mean([d['pre'] for d in D if d['comp'] == c and d['eps'] == 1e-2]) for c in range(6)]
NUM['dir_order_match'] = int(list(np.argsort(uL)) == list(np.argsort(pre)))
NUM['dir_rec_1e4'] = sum(d['cls'] == 'recovered' for d in D if d['eps'] == 1e-4)
NUM['dir_n_1e4'] = sum(1 for d in D if d['eps'] == 1e-4)
NUM['dir_pre2_max'] = max(pre); NUM['dir_pre2_min'] = min(pre)
recs = [sum(d['cls'] == 'recovered' for d in D if d['comp'] == c and d['eps'] == 1e-2) for c in range(6)]
NUM['dir_rec2_x'] = recs[0]; NUM['dir_rec2_vx'] = recs[3]; NUM['dir_rec2_vz'] = recs[5]
NUM['dir_it4_mean'] = float(np.mean([d['iters'] for d in D if d['eps'] == 1e-4 and d['cls'] == 'recovered']))
# original method: converged onto an equilibrium point (f(x0) ~ 0) -- possible
# only because dT was never corrected
from cr3bp_core import f6 as _f6
eq = 0; eqtot = 0
for o in R:
    d = o['m']['ORIG']
    if d['status'] == 'converged':
        eqtot += 1
        if np.linalg.norm(_f6(np.array(d['x0']), SEEDS[o['si']]['mu'])) < 1e-6:
            eq += 1
NUM['orig_equilibrium_n'] = eq
NUM['orig_equilibrium_pct'] = pct(eq, eqtot)
NUM['id_thresh'] = ID_THRESH
# ORIG converged runs between the threshold and 1e-1, and NN agreement there
L = [o['m']['ORIG'] for o in R if o['m']['ORIG']['status'] == 'converged' and ID_THRESH <= o['m']['ORIG']['maxn'] < 1e-1]
NUM['orig_mid_n'] = len(L)
NUM['orig_mid_nnseed'] = pct(sum(d.get('nn_is_seed', False) for d in L), len(L))
json.dump(NUM, open('results/numbers.json', 'w'), indent=1)
json.dump(TAB, open('results/tables.json', 'w'), indent=1)
print('extra numbers done')
