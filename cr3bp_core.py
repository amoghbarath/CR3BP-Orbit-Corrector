"""
cr3bp_core.py -- CR3BP dynamics, variational equations and an adaptive
Dormand-Prince 5(4) integrator that reproduces MATLAB ode45's step-size
controller (same Butcher tableau, same error estimate, same infinity-norm
error test with threshold AbsTol/RelTol, same step-growth rules, same default
MaxStep = 0.1*|tf - t0|).

All quantities are nondimensional (LU, TU).  State x = [x y z vx vy vz].
The augmented state used for sensitivities is the 42-vector [x; Phi(:)],
with Phi stored column-major (Fortran order) exactly as MATLAB's Phi(:).
"""
import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Dormand-Prince 5(4) coefficients (identical to ode45.m)
# ---------------------------------------------------------------------------
A2, A3, A4, A5 = 1/5, 3/10, 4/5, 8/9
B11 = 1/5
B21, B22 = 3/40, 9/40
B31, B32, B33 = 44/45, -56/15, 32/9
B41, B42, B43, B44 = 19372/6561, -25360/2187, 64448/6561, -212/729
B51, B52, B53, B54, B55 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
B61, B63, B64, B65, B66 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
E1, E3, E4, E5, E6, E7 = 71/57600, -71/16695, 71/1920, -17253/339200, 22/525, -1/40

# ntrp45 interpolation matrix (ode45 dense output), rows = stages 1..7
BI = np.array([
    [1.0, -183/64, 37/12, -145/128],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1500/371, -1000/159, 1000/371],
    [0.0, -125/32, 125/12, -375/64],
    [0.0, 9477/3392, -729/106, 25515/6784],
    [0.0, -11/7, 11/3, -55/28],
    [0.0, 3/2, -4.0, 5/2]])

EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
@njit(cache=True)
def accel(x, y, z, vx, vy, vz, mu):
    d1 = np.sqrt((x + mu)**2 + y*y + z*z)
    d2 = np.sqrt((x - 1 + mu)**2 + y*y + z*z)
    d13 = d1**3
    d23 = d2**3
    ax = -(1 - mu)*(x + mu)/d13 - mu*(x - 1 + mu)/d23 + 2*vy + x
    ay = -(1 - mu)*y/d13 - mu*y/d23 - 2*vx + y
    az = -(1 - mu)*z/d13 - mu*z/d23
    return ax, ay, az


@njit(cache=True)
def hessian_U(x, y, z, mu):
    """Second derivatives of the pseudo-potential Omega."""
    r1s = (x + mu)**2 + y*y + z*z
    r2s = (x - 1 + mu)**2 + y*y + z*z
    r13 = r1s**1.5
    r15 = r1s**2.5
    r23 = r2s**1.5
    r25 = r2s**2.5
    m1 = 1 - mu
    xa = x + mu
    xb = x - 1 + mu
    Uxx = 1 - m1/r13 - mu/r23 + 3*m1*xa*xa/r15 + 3*mu*xb*xb/r25
    Uyy = 1 - m1/r13 - mu/r23 + 3*m1*y*y/r15 + 3*mu*y*y/r25
    Uzz = -m1/r13 - mu/r23 + 3*m1*z*z/r15 + 3*mu*z*z/r25
    Uxy = 3*m1*xa*y/r15 + 3*mu*xb*y/r25
    Uxz = 3*m1*xa*z/r15 + 3*mu*xb*z/r25
    Uyz = 3*m1*y*z/r15 + 3*mu*y*z/r25
    return Uxx, Uyy, Uzz, Uxy, Uxz, Uyz


@njit(cache=True)
def rhs(Y, mu, out):
    """Right-hand side of the 6- or 42-dimensional system (in place)."""
    x, y, z, vx, vy, vz = Y[0], Y[1], Y[2], Y[3], Y[4], Y[5]
    ax, ay, az = accel(x, y, z, vx, vy, vz, mu)
    out[0] = vx; out[1] = vy; out[2] = vz
    out[3] = ax; out[4] = ay; out[5] = az
    if Y.shape[0] == 42:
        Uxx, Uyy, Uzz, Uxy, Uxz, Uyz = hessian_U(x, y, z, mu)
        # Phi column-major: Phi[i, j] = Y[6 + i + 6*j]
        for j in range(6):
            b = 6 + 6*j
            p0 = Y[b]; p1 = Y[b+1]; p2 = Y[b+2]
            p3 = Y[b+3]; p4 = Y[b+4]; p5 = Y[b+5]
            out[b] = p3
            out[b+1] = p4
            out[b+2] = p5
            out[b+3] = Uxx*p0 + Uxy*p1 + Uxz*p2 + 2*p4
            out[b+4] = Uxy*p0 + Uyy*p1 + Uyz*p2 - 2*p3
            out[b+5] = Uxz*p0 + Uyz*p1 + Uzz*p2


def f6(x, mu):
    """Vector field f(x) for a 6-state (numpy convenience)."""
    out = np.empty(6)
    rhs(np.ascontiguousarray(x[:6], dtype=float), mu, out)
    return out


def jacobi(x, mu):
    X, Y, Z, VX, VY, VZ = x[:6]
    d1 = np.sqrt((X + mu)**2 + Y**2 + Z**2)
    d2 = np.sqrt((X - 1 + mu)**2 + Y**2 + Z**2)
    U = 0.5*(X**2 + Y**2) + (1 - mu)/d1 + mu/d2
    return 2*U - (VX**2 + VY**2 + VZ**2)


def grad_jacobi(x, mu):
    X, Y, Z, VX, VY, VZ = x[:6]
    d1 = np.sqrt((X + mu)**2 + Y**2 + Z**2)
    d2 = np.sqrt((X - 1 + mu)**2 + Y**2 + Z**2)
    Ux = X - (1 - mu)*(X + mu)/d1**3 - mu*(X - 1 + mu)/d2**3
    Uy = Y - (1 - mu)*Y/d1**3 - mu*Y/d2**3
    Uz = -(1 - mu)*Z/d1**3 - mu*Z/d2**3
    return np.array([2*Ux, 2*Uy, 2*Uz, -2*VX, -2*VY, -2*VZ])


# ---------------------------------------------------------------------------
# One Dormand-Prince step (all 7 stage derivatives returned for dense output)
# ---------------------------------------------------------------------------
@njit(cache=True)
def dp_step(y, f1, h, mu, K, ytmp, ynew):
    n = y.shape[0]
    for i in range(n):
        K[0, i] = f1[i]
    for i in range(n):
        ytmp[i] = y[i] + h*(B11*K[0, i])
    rhs(ytmp, mu, K[1])
    for i in range(n):
        ytmp[i] = y[i] + h*(B21*K[0, i] + B22*K[1, i])
    rhs(ytmp, mu, K[2])
    for i in range(n):
        ytmp[i] = y[i] + h*(B31*K[0, i] + B32*K[1, i] + B33*K[2, i])
    rhs(ytmp, mu, K[3])
    for i in range(n):
        ytmp[i] = y[i] + h*(B41*K[0, i] + B42*K[1, i] + B43*K[2, i] + B44*K[3, i])
    rhs(ytmp, mu, K[4])
    for i in range(n):
        ytmp[i] = y[i] + h*(B51*K[0, i] + B52*K[1, i] + B53*K[2, i]
                            + B54*K[3, i] + B55*K[4, i])
    rhs(ytmp, mu, K[5])
    for i in range(n):
        ynew[i] = y[i] + h*(B61*K[0, i] + B63*K[2, i] + B64*K[3, i]
                            + B65*K[4, i] + B66*K[5, i])
    rhs(ynew, mu, K[6])


@njit(cache=True)
def dp_err(y, ynew, K, absh, threshold):
    n = y.shape[0]
    e = 0.0
    for i in range(n):
        fe = (E1*K[0, i] + E3*K[2, i] + E4*K[3, i] + E5*K[4, i]
              + E6*K[5, i] + E7*K[6, i])
        sc = max(max(abs(y[i]), abs(ynew[i])), threshold)
        v = abs(fe)/sc
        if v > e:
            e = v
    return absh*e


@njit(cache=True)
def _initial_step(y0, f0, t0, tf, rtol, threshold, hmax):
    pow_ = 1.0/5.0
    htspan = abs(tf - t0)
    absh = min(hmax, htspan)
    rh = 0.0
    for i in range(y0.shape[0]):
        v = abs(f0[i])/max(abs(y0[i]), threshold)
        if v > rh:
            rh = v
    rh = rh/(0.8*rtol**pow_)
    if absh*rh > 1:
        absh = 1.0/rh
    hmin = 16*np.spacing(abs(t0)) if t0 != 0 else 16*np.spacing(1.0)*1e-300
    return max(absh, hmin)


# ---------------------------------------------------------------------------
# Fixed-endpoint propagation
# ---------------------------------------------------------------------------
@njit(cache=True)
def propagate(y0, tf, mu, rtol, atol, maxsteps=2000000):
    """Integrate from t=0 to t=tf.  Returns (y(tf), status, nsteps).
    status 0 = ok, 1 = step size underflow, 2 = too many steps, 3 = non-finite."""
    n = y0.shape[0]
    y = y0.copy()
    f = np.empty(n)
    rhs(y, mu, f)
    K = np.empty((7, n))
    ytmp = np.empty(n)
    ynew = np.empty(n)
    t = 0.0
    tdir = 1.0 if tf >= 0 else -1.0
    threshold = atol/rtol
    hmax = 0.1*abs(tf)
    pow_ = 1.0/5.0
    absh = _initial_step(y, f, 0.0, tf, rtol, threshold, hmax)
    nsteps = 0
    done = False
    while not done:
        hmin = 16*np.spacing(abs(t)) if t != 0 else 1e-300
        absh = min(hmax, max(hmin, absh))
        h = tdir*absh
        if 1.1*absh >= abs(tf - t):
            h = tf - t
            absh = abs(h)
            done = True
        nofailed = True
        while True:
            dp_step(y, f, h, mu, K, ytmp, ynew)
            err = dp_err(y, ynew, K, absh, threshold)
            if not np.isfinite(err):
                return y, 3, nsteps
            if err > rtol:
                if absh <= hmin:
                    return y, 1, nsteps
                if nofailed:
                    nofailed = False
                    absh = max(hmin, absh*max(0.1, 0.8*(rtol/err)**pow_))
                else:
                    absh = max(hmin, 0.5*absh)
                h = tdir*absh
                done = False
            else:
                break
        nsteps += 1
        if nsteps > maxsteps:
            return y, 2, nsteps
        t = tf if done else t + h
        for i in range(n):
            y[i] = ynew[i]
            f[i] = K[6, i]
        if nofailed:
            temp = 1.25*(err/rtol)**pow_
            if temp > 0.2:
                absh = absh/temp
            else:
                absh = 5.0*absh
    return y, 0, nsteps


@njit(cache=True)
def _interp(y, K, h, s, out):
    """ntrp45 dense output at fraction s of the step."""
    s1 = s; s2 = s*s; s3 = s2*s; s4 = s3*s
    n = y.shape[0]
    for i in range(n):
        acc = 0.0
        for k in range(7):
            acc += K[k, i]*(BI[k, 0]*s1 + BI[k, 1]*s2 + BI[k, 2]*s3 + BI[k, 3]*s4)
        out[i] = y[i] + h*acc


# ---------------------------------------------------------------------------
# Propagation to a y = 0 crossing (event detection + exact refinement)
# ---------------------------------------------------------------------------
@njit(cache=True)
def propagate_to_crossing(y0, tmin, tmax, direction, mu, rtol, atol, maxsteps=50000):
    """Integrate until y (component 1) crosses zero with sign(dy/dt) == direction
    at a time t > tmin.  The crossing is bracketed on the accepted step, located
    on the ntrp45 interpolant, and then refined by re-integrating exactly to the
    crossing time (one DP step, shorter than the accepted step) followed by
    Newton corrections dt = -y/vy.  Returns (t*, y(t*), status)."""
    n = y0.shape[0]
    y = y0.copy()
    f = np.empty(n)
    rhs(y, mu, f)
    K = np.empty((7, n))
    ytmp = np.empty(n)
    ynew = np.empty(n)
    yev = np.empty(n)
    K2 = np.empty((7, n))
    t = 0.0
    threshold = atol/rtol
    hmax = 0.1*abs(tmax)
    pow_ = 1.0/5.0
    absh = _initial_step(y, f, 0.0, tmax, rtol, threshold, hmax)
    nsteps = 0
    while t < tmax:
        hmin = 16*np.spacing(abs(t)) if t != 0 else 1e-300
        absh = min(hmax, max(hmin, absh))
        h = absh
        if 1.1*absh >= abs(tmax - t):
            h = tmax - t
            absh = h
        nofailed = True
        while True:
            dp_step(y, f, h, mu, K, ytmp, ynew)
            err = dp_err(y, ynew, K, absh, threshold)
            if not np.isfinite(err):
                return 0.0, y, 3
            if err > rtol:
                if absh <= hmin:
                    return 0.0, y, 1
                if nofailed:
                    nofailed = False
                    absh = max(hmin, absh*max(0.1, 0.8*(rtol/err)**pow_))
                else:
                    absh = max(hmin, 0.5*absh)
                h = absh
            else:
                break
        nsteps += 1
        if nsteps > maxsteps:
            return 0.0, y, 2
        tnew = t + h
        # event check
        g0 = y[1]
        g1 = ynew[1]
        crossed = False
        if tnew > tmin:
            if direction > 0 and g0 < 0.0 and g1 >= 0.0:
                crossed = True
            elif direction < 0 and g0 > 0.0 and g1 <= 0.0:
                crossed = True
        if crossed:
            # locate on interpolant (Illinois)
            sa = 0.0; sb = 1.0
            ga = g0; gb = g1
            side = 0
            for it in range(200):
                sc = (sa*gb - sb*ga)/(gb - ga)
                if not (sc > sa and sc < sb):
                    sc = 0.5*(sa + sb)
                _interp(y, K, h, sc, yev)
                gc = yev[1]
                if gc == 0.0:
                    sa = sc; sb = sc
                    break
                if (gc > 0) == (gb > 0):
                    sb = sc; gb = gc
                    if side == 1:
                        ga *= 0.5
                    side = 1
                else:
                    sa = sc; ga = gc
                    if side == -1:
                        gb *= 0.5
                    side = -1
                if (sb - sa)*abs(h) < 4*np.spacing(tnew):
                    break
            sstar = 0.5*(sa + sb)
            if sstar*abs(h) < tmin - t:
                sstar = (tmin - t)/h
            # exact re-integration to the event time from the step start
            hs = sstar*h
            fy = np.empty(n)
            rhs(y, mu, fy)
            dp_step(y, fy, hs, mu, K2, ytmp, yev)
            tev = t + hs
            # Newton refinement on t: dt = -y/vy (small single steps)
            for it in range(3):
                vyv = yev[4]
                if vyv == 0.0:
                    break
                dt = -yev[1]/vyv
                if abs(dt) < 1e-17:
                    break
                rhs(yev, mu, fy)
                for i in range(n):
                    ytmp[i] = yev[i]
                dp_step(ytmp, fy, dt, mu, K2, ynew, yev)
                tev = tev + dt
            return tev, yev, 0
        t = tnew
        for i in range(n):
            y[i] = ynew[i]
            f[i] = K[6, i]
        if nofailed:
            temp = 1.25*(err/rtol)**pow_
            if temp > 0.2:
                absh = absh/temp
            else:
                absh = 5.0*absh
    return t, y, 4   # no crossing found before tmax


# ---------------------------------------------------------------------------
# Dense output for plotting / orbit comparison
# ---------------------------------------------------------------------------
@njit(cache=True)
def propagate_dense(y0, tout, mu, rtol, atol):
    """Integrate and return the state at the (increasing) times in tout,
    using the ntrp45 interpolant exactly as ode45 does when given a tspan
    vector.  tout[0] must be 0."""
    n = y0.shape[0]
    m = tout.shape[0]
    Yout = np.empty((m, n))
    y = y0.copy()
    f = np.empty(n)
    rhs(y, mu, f)
    K = np.empty((7, n))
    ytmp = np.empty(n)
    ynew = np.empty(n)
    yi = np.empty(n)
    tf = tout[m - 1]
    t = 0.0
    threshold = atol/rtol
    hmax = 0.1*abs(tf)
    pow_ = 1.0/5.0
    absh = _initial_step(y, f, 0.0, tf, rtol, threshold, hmax)
    for i in range(n):
        Yout[0, i] = y[i]
    k = 1
    done = False
    while not done:
        hmin = 16*np.spacing(abs(t)) if t != 0 else 1e-300
        absh = min(hmax, max(hmin, absh))
        h = absh
        if 1.1*absh >= abs(tf - t):
            h = tf - t
            absh = h
            done = True
        nofailed = True
        while True:
            dp_step(y, f, h, mu, K, ytmp, ynew)
            err = dp_err(y, ynew, K, absh, threshold)
            if not np.isfinite(err):
                for kk in range(k, m):
                    for i in range(n):
                        Yout[kk, i] = np.nan
                return Yout
            if err > rtol:
                if absh <= hmin:
                    for kk in range(k, m):
                        for i in range(n):
                            Yout[kk, i] = np.nan
                    return Yout
                if nofailed:
                    nofailed = False
                    absh = max(hmin, absh*max(0.1, 0.8*(rtol/err)**pow_))
                else:
                    absh = max(hmin, 0.5*absh)
                h = absh
                done = False
            else:
                break
        tnew = tf if done else t + h
        while k < m and tout[k] <= tnew:
            if tout[k] == tnew:
                for i in range(n):
                    Yout[k, i] = ynew[i]
            else:
                _interp(y, K, h, (tout[k] - t)/h, yi)
                for i in range(n):
                    Yout[k, i] = yi[i]
            k += 1
        t = tnew
        for i in range(n):
            y[i] = ynew[i]
            f[i] = K[6, i]
        if nofailed:
            temp = 1.25*(err/rtol)**pow_
            if temp > 0.2:
                absh = absh/temp
            else:
                absh = 5.0*absh
    return Yout


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
TOL = 5e-14   # RelTol = AbsTol used for every reported result
MAXSTEPS = 50000   # a period needing more steps than this (ten times the most
                   # demanding seed) passes too close to a primary: treated as
                   # a failed propagation


def aug(x0):
    Y = np.zeros(42)
    Y[:6] = x0
    Y[6::7] = 1.0          # identity, column-major: indices 6, 13, 20, ...
    return Y


def prop_state(x0, T, mu, rtol=TOL, atol=TOL, stm=True):
    """State (and STM) after time T.  Returns (xT, Phi, status)."""
    if stm:
        Y, st, _ = propagate(aug(np.asarray(x0, float)), float(T), mu, rtol, atol, MAXSTEPS)
        Phi = Y[6:].reshape(6, 6, order='F')
        return Y[:6].copy(), Phi, st
    Y, st, _ = propagate(np.asarray(x0, float).copy(), float(T), mu, rtol, atol, MAXSTEPS)
    return Y[:6].copy(), None, st


def prop_crossing(x0, Tguess, mu, rtol=TOL, atol=TOL, stm=True,
                  tmin_frac=0.25, tmax_frac=0.75):
    """Propagate to the first y=0 crossing in (tmin, tmax) whose direction is
    opposite to sign(vy0): the half-period crossing of an xz-symmetric orbit."""
    direction = -1.0 if x0[4] >= 0 else 1.0
    Y0 = aug(np.asarray(x0, float)) if stm else np.asarray(x0, float).copy()
    t, Y, st = propagate_to_crossing(Y0, tmin_frac*Tguess, tmax_frac*Tguess,
                                     direction, mu, rtol, atol)
    Phi = Y[6:].reshape(6, 6, order='F') if stm else None
    return t, Y[:6].copy(), Phi, st


def trajectory(x0, T, mu, n=2000, rtol=TOL, atol=TOL):
    tout = np.linspace(0.0, T, n)
    Y = propagate_dense(np.asarray(x0, float).copy(), tout, mu, rtol, atol)
    return tout, Y
