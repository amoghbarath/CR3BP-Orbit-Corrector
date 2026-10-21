"""
correctors.py -- the seven correctors compared in the revised manuscript.

All correctors are judged by the SAME quantity and the SAME test: the
full-period periodicity residual F(x0, T) = x(T; x0) - x0, scaled by the
orbit amplitudes (positions by A_r, velocities by A_v),

    converged  <=>  ( ||F ./ W|| < tol   or   ||F|| < ABSFLOOR )

(and, for the bordered methods, both constraint residuals below tol in the
same scaled units).  W = [A_r A_r A_r A_v A_v A_v].

  BA      bordered augmented Newton + backtracking line search  (revised method)
  BA0     bordered augmented Newton, full steps (no line search)
  CHORD   bordered system with the matrix frozen at the first iterate
  BAP     BA started from the symmetric projection used by SDC (diagnostic)
  ORIG    the first version's method: 6x7 system, MATLAB-backslash basic
          solution, warm-up line search (alpha0 = 1e-3, x1.5 per accepted step)
  SDC     standard symmetric (half-period) differential correction with the
          moving-crossing-time Jacobian, full Newton steps
  FP      fixed-point (Picard) iteration on the period-T flow map

"""
import numpy as np
import scipy.linalg as sla
from cr3bp_core import prop_state, prop_crossing, f6, TOL

ABSFLOOR = 2e-12         # accept when the unscaled closure reaches integration noise
EPSM = np.finfo(float).eps


class Result:
    __slots__ = ('method', 'status', 'iters', 'hist', 'alpha', 'x0', 'T',
                 'nprop', 'sv', 'condB', 'Phi', 'F', 'g_hist')

    def __init__(self, method):
        self.method = method
        self.status = 'running'
        self.iters = 0
        self.hist = []          # scaled residual ||F./W|| per iterate
        self.alpha = []
        self.x0 = None
        self.T = None
        self.nprop = 0
        self.sv = None
        self.condB = []
        self.Phi = None
        self.F = None
        self.g_hist = []

    @property
    def converged(self):
        return self.status == 'converged'


def _bad(x, T, Tseed):
    return (not np.all(np.isfinite(x))) or (not np.isfinite(T)) or T < 0.3*Tseed \
        or T > 3*Tseed or np.max(np.abs(x[:3])) > 5


def _diverging(hist):
    return hist[-1] > max(10.0, 10.0*hist[0])


def _ok(F, W, tol, absfloor=ABSFLOOR):
    return np.linalg.norm(F/W) < tol or np.linalg.norm(F) < absfloor


# ---------------------------------------------------------------------------
# Bordered augmented system
# ---------------------------------------------------------------------------
def bordered_matrix(xT, Phi, fix, mu):
    """8x7 bordered matrix B = [Phi - I, f(xT); e_y^T, 0; e_fix^T, 0]."""
    B = np.zeros((8, 7))
    B[:6, :6] = Phi - np.eye(6)
    B[:6, 6] = f6(xT, mu)
    B[6, 1] = 1.0
    B[7, fix] = 1.0
    return B


def scaled_sv(B, W, fix, T):
    """Singular values of the amplitude-scaled bordered matrix."""
    Dr = np.concatenate([1/W, [1/W[0], 1/W[fix]]])
    Dc = np.concatenate([W, [T]])
    return np.linalg.svd(Dr[:, None]*B*Dc[None, :], compute_uv=False)


def constrained_step(B, F, h, W):
    """min || W^-1 (B[:6] d + F) ||  subject to  B[6:8] d = -h  exactly
    (null-space method).  Unique when B has full column rank."""
    M = B[:6, :]/W[:, None]
    Fs = F/W
    C = B[6:8, :]
    Q, R = np.linalg.qr(C.T, mode='complete')       # C^T = Q R, Q is 7x7
    yp = sla.solve_triangular(R[:2, :2].T, -h, lower=True)
    dp = Q[:, :2] @ yp
    N = Q[:, 2:]                                      # null space of C (7x5)
    w, *_ = np.linalg.lstsq(M @ N, -Fs - M @ dp, rcond=None)
    return dp + N @ w


def correct_ba(x0, T, mu, fix, fixval, tol, W, maxit=50, linesearch=True,
               chord=False, max_back=20, c_armijo=1e-4, rtol=TOL, atol=TOL, name=None,
               absfloor=ABSFLOOR):
    """Bordered augmented Newton.  fix = index of the held coordinate (0..5),
    fixval = its value on the intended orbit, W = residual scales (6,)."""
    res = Result(name or ('CHORD' if chord else ('BA' if linesearch else 'BA0')))
    W = np.asarray(W, float)
    hs = np.array([W[0], W[fix]])
    Tseed = T
    x = np.array(x0, float)
    xT, Phi, st = prop_state(x, T, mu, rtol, atol)
    res.nprop += 1
    if st != 0:
        res.status = 'diverged'
        res.x0, res.T = x, T
        return res
    F = xT - x
    h = np.array([x[1], x[fix] - fixval])
    merit = np.sqrt(np.sum((F/W)**2) + np.sum((h/hs)**2))
    res.hist.append(float(np.linalg.norm(F/W)))
    res.g_hist.append(float(np.linalg.norm(h/hs)))
    Bfrozen = None
    while True:
        if _ok(F, W, tol, absfloor) and np.linalg.norm(h/hs) < tol:
            res.status = 'converged'
            break
        if res.iters >= maxit:
            res.status = 'maxiter'
            break
        if chord and Bfrozen is not None:
            B = Bfrozen
        else:
            B = bordered_matrix(xT, Phi, fix, mu)
            if chord:
                Bfrozen = B
        sv = scaled_sv(B, W, fix, Tseed)
        res.sv = sv
        res.condB.append(float(sv[0]/sv[-1]))
        d = constrained_step(B, F, h, W)
        alpha = 1.0
        ok = False
        nb = 0
        while True:
            xn = x + alpha*d[:6]
            Tn = T + alpha*d[6]
            st = 9
            if not _bad(xn, Tn, Tseed):
                # trial points are propagated without the STM; the STM is
                # computed only once a point has been accepted
                xTn, _, st = prop_state(xn, Tn, mu, rtol, atol, stm=False)
                res.nprop += 1
            if st == 0:
                Fn = xTn - xn
                hn = np.array([xn[1], xn[fix] - fixval])
                mn = np.sqrt(np.sum((Fn/W)**2) + np.sum((hn/hs)**2))
                if (not linesearch) or mn <= (1 - c_armijo*alpha)*merit:
                    ok = True
                    break
            elif not linesearch:
                break
            nb += 1
            if nb > max_back:
                break
            alpha *= 0.5
        if ok:
            xTn, Phin, st = prop_state(xn, Tn, mu, rtol, atol)
            res.nprop += 1
            if st != 0:
                ok = False
            else:
                Fn = xTn - xn
                mn = np.sqrt(np.sum((Fn/W)**2) + np.sum((hn/hs)**2))
        if not ok:
            res.status = 'ls_fail' if linesearch else 'diverged'
            break
        x, T, xT, Phi, F, h, merit = xn, Tn, xTn, Phin, Fn, hn, mn
        res.iters += 1
        res.alpha.append(alpha)
        res.hist.append(float(np.linalg.norm(F/W)))
        res.g_hist.append(float(np.linalg.norm(h/hs)))
        if _diverging(res.hist):
            res.status = 'diverged'
            break
    res.x0, res.T, res.Phi, res.F = x, T, Phi, F
    return res


# ---------------------------------------------------------------------------
# Original method of the first submission
# ---------------------------------------------------------------------------
def backslash_basic(A, b):
    """MATLAB mldivide for an underdetermined system: QR with column
    pivoting, basic solution with at most rank(A) nonzeros."""
    Q, R, P = sla.qr(A, pivoting=True, mode='economic')
    diagR = np.abs(np.diag(R))
    tolr = max(A.shape)*EPSM*diagR[0]
    r = int(np.sum(diagR > tolr))
    z = sla.solve_triangular(R[:r, :r], Q[:, :r].T @ b)
    x = np.zeros(A.shape[1])
    x[P[:r]] = z
    return x, r


def correct_orig(x0, T, mu, tol, W, maxit=200, alpha0=1e-3, max_back=40,
                 rtol=TOL, atol=TOL):
    """The first version's corrector (line_search.m): err = x0 - x(T),
    [Phi - I, f(x(T))] delta = err solved by backslash, x0 <- x0 + alpha*dx,
    alpha starts at alpha0, is halved until ||err|| does not increase, and is
    multiplied by 1.5 (capped at 1) after every accepted step."""
    res = Result('ORIG')
    W = np.asarray(W, float)
    Tseed = T
    x = np.array(x0, float)
    xT, Phi, st = prop_state(x, T, mu, rtol, atol)
    res.nprop += 1
    if st != 0:
        res.status = 'diverged'
        res.x0, res.T = x, T
        return res
    err = x - xT
    res.hist.append(float(np.linalg.norm(err/W)))
    alpha = alpha0
    while True:
        if _ok(-err, W, tol):
            res.status = 'converged'
            break
        if res.iters >= maxit:
            res.status = 'maxiter'
            break
        A = np.hstack([Phi - np.eye(6), f6(xT, mu)[:, None]])
        res.condB.append(float(np.linalg.cond(A)))
        delta, _ = backslash_basic(A, err)
        nb = 0
        errn = None
        while True:
            xn = x + alpha*delta[:6]
            Tn = T + alpha*delta[6]
            errn = None
            if not _bad(xn, Tn, Tseed):
                xTn, _, st = prop_state(xn, Tn, mu, rtol, atol, stm=False)
                res.nprop += 1
                if st == 0:
                    errn = xn - xTn
            if errn is not None and np.linalg.norm(errn) <= np.linalg.norm(err):
                break
            if nb >= max_back:
                errn = None
                break
            alpha /= 2
            nb += 1
        if errn is not None:
            xTn, Phin, st = prop_state(xn, Tn, mu, rtol, atol)
            res.nprop += 1
            errn = (xn - xTn) if st == 0 else None
        if errn is None:
            res.status = 'ls_fail'
            break
        x, T, xT, Phi, err = xn, Tn, xTn, Phin, errn
        res.iters += 1
        res.alpha.append(alpha)
        res.hist.append(float(np.linalg.norm(err/W)))
        if _diverging(res.hist):
            res.status = 'diverged'
            break
        alpha = min(alpha*1.5, 1.0)
    res.x0, res.T, res.Phi, res.F = x, T, Phi, -err
    return res


# ---------------------------------------------------------------------------
# Standard symmetric differential correction (half period)
# ---------------------------------------------------------------------------
def project_to_section(x0, fix, fixval, planar):
    """Map an arbitrary state onto the symmetric form [x,0,z,0,vy,0] used by
    the standard corrector, with the held coordinate set to its target."""
    s = np.array(x0, float).copy()
    s[1] = 0.0
    s[3] = 0.0
    s[5] = 0.0
    if planar:
        s[2] = 0.0
    s[fix] = fixval
    return s


def correct_sdc(x0, T, mu, fix, fixval, planar, tol, W, maxit=50,
                rtol=TOL, atol=TOL):
    """Free variables: {x0, z0, vy0} minus the held coordinate (planar orbits:
    {x0, vy0} minus it).  Targets at the next xz-plane crossing: vx = 0
    (and vz = 0 for spatial orbits).  Jacobian accounts for the moving
    crossing time.  Full Newton steps.  Convergence is judged on the
    full-period residual like every other method."""
    res = Result('SDC')
    W = np.asarray(W, float)
    s = project_to_section(x0, fix, fixval, planar)
    cand = [0, 4] if planar else [0, 2, 4]
    free = [c for c in cand if c != fix]
    targ = [3] if planar else [3, 5]
    Tseed = T

    def half(sv, Tg):
        th, xh, Ph, st = prop_crossing(sv, Tg, mu, rtol, atol)
        res.nprop += 1
        return th, xh, Ph, st

    def full_res(sv, Tf):
        xT, _, st = prop_state(sv, Tf, mu, rtol, atol, stm=False)
        res.nprop += 1
        return (xT - sv) if st == 0 else None

    th, xh, Ph, st = half(s, T)
    if st != 0:
        res.status = 'diverged'
        res.x0, res.T = s, T
        return res
    Fh = xh[targ]
    Ff = full_res(s, 2*th)
    if Ff is None:
        res.status = 'diverged'
        res.x0, res.T = s, 2*th
        return res
    res.hist.append(float(np.linalg.norm(Ff/W)))
    while True:
        if _ok(Ff, W, tol):
            res.status = 'converged'
            break
        if res.iters >= maxit:
            res.status = 'maxiter'
            break
        acc = f6(xh, mu)
        vyc = xh[4]
        if abs(vyc) < 1e-14:
            res.status = 'diverged'
            break
        J = Ph[np.ix_(targ, free)] - np.outer(acc[targ]/vyc, Ph[1, free])
        sv = np.linalg.svd(J, compute_uv=False)
        res.sv = sv
        res.condB.append(float(sv[0]/sv[-1]))
        try:
            dq = np.linalg.solve(J, -Fh)
        except np.linalg.LinAlgError:
            res.status = 'diverged'
            break
        sn = s.copy()
        sn[free] += dq
        if _bad(sn, 2*th, Tseed):
            res.status = 'diverged'
            break
        tht, xht, Pht, stt = half(sn, 2*th)
        if stt != 0 or not (0.3*Tseed < 2*tht < 3*Tseed):
            res.status = 'diverged'
            break
        s, th, xh, Ph, Fh = sn, tht, xht, Pht, xht[targ]
        res.iters += 1
        res.alpha.append(1.0)
        Ff = full_res(s, 2*th)
        if Ff is None:
            res.status = 'diverged'
            break
        res.hist.append(float(np.linalg.norm(Ff/W)))
        if _diverging(res.hist):
            res.status = 'diverged'
            break
    res.x0, res.T = s, 2*th
    if res.converged:
        _, Phi, _ = prop_state(s, 2*th, mu, rtol, atol)
        res.Phi = Phi
        res.F = Ff
    return res


# ---------------------------------------------------------------------------
# Fixed-point (Picard) iteration on the period-T map
# ---------------------------------------------------------------------------
def correct_fp(x0, T, mu, tol, W, maxit=50, rtol=TOL, atol=TOL):
    res = Result('FP')
    W = np.asarray(W, float)
    x = np.array(x0, float)
    for k in range(maxit + 1):
        xT, _, st = prop_state(x, T, mu, rtol, atol, stm=False)
        res.nprop += 1
        if st != 0:
            res.status = 'diverged'
            break
        F = xT - x
        res.hist.append(float(np.linalg.norm(F/W)))
        if _ok(F, W, tol):
            res.status = 'converged'
            res.F = F
            break
        if _diverging(res.hist) or _bad(xT, T, T):
            res.status = 'diverged'
            break
        if k == maxit:
            res.status = 'maxiter'
            break
        x = xT
        res.iters += 1
    res.x0, res.T = x, T
    if res.converged:
        _, Phi, _ = prop_state(x, T, mu, rtol, atol)
        res.Phi = Phi
    return res
