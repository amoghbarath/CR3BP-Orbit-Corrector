"""
metrics.py -- orbit amplitude, perturbation model, orbit-identity metrics,
Floquet analysis and conditioning diagnostics.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from cr3bp_core import (trajectory, prop_state, prop_crossing, propagate_to_crossing,
                        f6, jacobi, grad_jacobi, aug, TOL)


def amplitudes(x0, T, mu, n=4000):
    """A_r = largest peak-to-peak extent of (x, y, z) over one period;
    A_v = largest peak-to-peak extent of (vx, vy, vz)."""
    _, X = trajectory(x0, T, mu, n)
    span = X.max(axis=0) - X.min(axis=0)
    return float(span[:3].max()), float(span[3:6].max())


def perturb(x0, eps, Ar, Av, rng, comps=None):
    """Independent Gaussian perturbation: position by eps*Ar, velocity by
    eps*Av, xi ~ N(0, I).  comps = list of component indices to perturb
    (None = all six)."""
    xi = rng.standard_normal(6)
    scale = np.array([Ar]*3 + [Av]*3)*eps
    d = xi*scale
    if comps is not None:
        mask = np.zeros(6)
        mask[comps] = 1.0
        d = d*mask
    return np.asarray(x0, float) + d


# ---------------------------------------------------------------------------
# Orbit identity
# ---------------------------------------------------------------------------
def _state_at(x0, t, mu):
    if t <= 0:
        return np.array(x0, float)
    xt, _, st = prop_state(x0, t, mu, stm=False)
    return xt


def phase_align(xc, Tc, xs, mu, W=None, n=2000):
    """Time tau in [0, Tc) at which the corrected orbit passes closest to the
    seed's initial state xs (6-D distance, scaled by W when given);
    returns (tau, x_c(tau))."""
    W = np.ones(6) if W is None else np.asarray(W, float)
    t, X = trajectory(xc, Tc, mu, n)
    d = np.linalg.norm((X[:, :6] - xs[None, :])/W[None, :], axis=1)
    k = int(np.argmin(d))
    dt = Tc/(n - 1)
    lo, hi = t[k] - dt, t[k] + dt

    def obj(tau):
        tt = tau % Tc
        return np.linalg.norm((_state_at(xc, tt, mu) - xs)/W)
    r = minimize_scalar(obj, bounds=(lo, hi), method='bounded',
                        options=dict(xatol=1e-13*max(Tc, 1), maxiter=200))
    tau = r.x % Tc
    return tau, _state_at(xc, tau, mu)


def orbit_identity(xs, Ts, xc, Tc, mu, Ar, Av, n=1001):
    """Compare a corrected orbit (xc, Tc) with the seed (xs, Ts) over one
    period after phase alignment.  Distances are nondimensional; *_rel are
    divided by the amplitude (Ar for position, Av for velocity)."""
    W = np.array([Ar]*3 + [Av]*3)
    tau, xca = phase_align(xc, Tc, np.asarray(xs, float), mu, W)
    _, Xs = trajectory(xs, Ts, mu, n)
    _, Xc = trajectory(xca, Tc, mu, n)
    D = Xc[:, :6] - Xs[:, :6]
    d6 = np.linalg.norm(D, axis=1)
    dr = np.linalg.norm(D[:, :3], axis=1)
    dv = np.linalg.norm(D[:, 3:], axis=1)
    Dn = np.hstack([D[:, :3]/Ar, D[:, 3:]/Av])
    dn = np.linalg.norm(Dn, axis=1)
    return dict(tau=tau,
                max6=float(d6.max()), rms6=float(np.sqrt(np.mean(d6**2))),
                maxr=float(dr.max()), maxv=float(dv.max()),
                rmsr=float(np.sqrt(np.mean(dr**2))), rmsv=float(np.sqrt(np.mean(dv**2))),
                maxn=float(dn.max()), rmsn=float(np.sqrt(np.mean(dn**2))),
                dT_rel=float(abs(Tc - Ts)/Ts),
                dC=float(abs(jacobi(xc, mu) - jacobi(xs, mu))))


def section_state(x0, T, mu, vysign):
    """State at the first xz-plane crossing (y = 0) with sign(vy) == vysign,
    searched over (0, 1.05 T]."""
    direction = 1.0 if vysign > 0 else -1.0
    # y increases through 0 iff vy > 0 at the crossing
    t, Y, st = propagate_to_crossing(np.asarray(x0, float).copy(), 1e-9*T, 1.05*T,
                                     direction, mu, TOL, TOL)
    if st != 0:
        return None
    return Y[:6].copy()


def nearest_neighbor(xc, Tc, mu, family_states, seed_index):
    """Rank JPL family members by 6-D distance to the corrected orbit's state
    on the same xz-plane crossing as the catalogue states."""
    vysign = np.sign(family_states[seed_index][4])
    xs = section_state(xc, Tc, mu, vysign)
    if xs is None:
        return None
    d = np.linalg.norm(family_states - xs[None, :], axis=1)
    order = np.argsort(d)
    return dict(nn=int(order[0]), d1=float(d[order[0]]),
                d2=float(d[order[1]]) if len(order) > 1 else np.nan,
                is_seed=bool(order[0] == seed_index),
                d_seed=float(d[seed_index]))


# ---------------------------------------------------------------------------
# Floquet analysis
# ---------------------------------------------------------------------------
def floquet(x0, T, mu):
    _, M, _ = prop_state(x0, T, mu)
    lam, V = np.linalg.eig(M)
    lmax = np.max(np.abs(lam))
    nu = 0.5*(lmax + 1/lmax)
    sv = np.linalg.svd(M, compute_uv=False)
    return dict(M=M, lam=lam, V=V, lmax=float(lmax), nu=float(nu),
                smax=float(sv[0]), sv=sv)


def pair_multipliers(lam):
    """Group the six multipliers into the trivial pair and two reciprocal
    pairs; return list of (pair, stability index of the pair)."""
    lam = list(lam)
    # trivial pair: the two closest to 1
    idx = np.argsort(np.abs(np.array(lam) - 1.0))
    triv = [lam[idx[0]], lam[idx[1]]]
    rest = [lam[i] for i in idx[2:]]
    pairs = []
    used = [False]*4
    for i in range(4):
        if used[i]:
            continue
        best, bj = np.inf, -1
        for j in range(4):
            if j == i or used[j]:
                continue
            err = abs(rest[i]*rest[j] - 1.0)
            if err < best:
                best, bj = err, j
        used[i] = used[bj] = True
        p = (rest[i], rest[bj])
        nu = 0.5*abs(p[0] + p[1])
        pairs.append((p, nu))
    return triv, pairs


# ---------------------------------------------------------------------------
# Conditioning of the original and bordered systems at an orbit
# ---------------------------------------------------------------------------
def conditioning(x0, T, mu, fix, Ar, Av):
    """Rank and conditioning of the original 6x7 matrix A and the 8x7
    bordered matrix B at an orbit, both scaled by the orbit amplitudes
    (rows by 1/W, columns by [W, T]).  Returns the singular values, the gap
    sigma5/sigma6 of A, |cos| between A's sixth left singular vector and the
    (scaled) Jacobi gradient, the relative residual of the time-shift
    direction [f(x0); 0], and the condition number of B."""
    W = np.array([Ar]*3 + [Av]*3)
    xT, Phi, _ = prop_state(x0, T, mu)
    fT = f6(xT, mu)
    A = np.hstack([Phi - np.eye(6), fT[:, None]])
    Dc = np.concatenate([W, [T]])
    As = A/W[:, None]*Dc[None, :]
    U, s, Vt = np.linalg.svd(As)
    g = grad_jacobi(x0, mu)*W
    g = g/np.linalg.norm(g)
    cos_left = abs(U[:, 5] @ g)
    f0 = f6(x0, mu)
    v = np.concatenate([f0, [0.0]])/Dc
    v = v/np.linalg.norm(v)
    right = np.linalg.norm(As @ v)/s[0]
    B = np.zeros((8, 7))
    B[:6] = A
    B[6, 1] = 1.0
    B[7, fix] = 1.0
    Dr = np.concatenate([1/W, [1/W[0], 1/W[fix]]])
    Bs = Dr[:, None]*B*Dc[None, :]
    sB = np.linalg.svd(Bs, compute_uv=False)
    rankA = int(np.sum(s > max(As.shape)*np.spacing(s[0])))
    rankB = int(np.sum(sB > max(Bs.shape)*np.spacing(sB[0])))
    return dict(svA=s, condA=float(s[0]/s[-1]), gapA=float(s[4]/s[5]),
                rankA=rankA, cos_left=float(cos_left), right_null=float(right),
                svB=sB, condB=float(sB[0]/sB[-1]), rankB=rankB)
