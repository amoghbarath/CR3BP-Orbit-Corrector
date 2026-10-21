"""runlib.py -- seed list, method dispatch and outcome classification shared by
every experiment script.

  SEEDS     the sixty JPL catalog seeds (seeds/seeds60.json, built by build_seeds.py)
  METHODS   the seven correctors, in the order used in tables and figures
  TAU       convergence tolerance on the amplitude-scaled residual
  run_method(m, s, xp)   run corrector m on perturbed state xp of seed s
  classify(s, r)         'recovered' / 'other' / 'failed' (+ identity metrics)
"""
import json, os
import numpy as np
from correctors import (correct_ba, correct_orig, correct_sdc, correct_fp,
                        project_to_section)
from metrics import orbit_identity

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = json.load(open(os.path.join(HERE, 'seeds', 'seeds60.json')))
METHODS = ['BA', 'BA0', 'SDC', 'BAP', 'ORIG', 'CHORD', 'FP']
TAU = 1e-8            # ||F ./ W|| < TAU  (or ||F|| < 2e-12)
ID_THRESH = 1e-4      # max scaled distance from the seed over one period (see Methods)


def W_of(s):
    return np.array([s['Ar']]*3 + [s['Av']]*3)


def run_method(m, s, xp):
    x0 = np.array(s['x0'])
    mu, T, fix = s['mu'], s['T'], s['fix']
    W = W_of(s)
    planar = s['family'] != 'halo'
    if m == 'BA':
        return correct_ba(xp, T, mu, fix, x0[fix], TAU, W)
    if m == 'BA0':
        return correct_ba(xp, T, mu, fix, x0[fix], TAU, W, linesearch=False)
    if m == 'CHORD':
        return correct_ba(xp, T, mu, fix, x0[fix], TAU, W, chord=True)
    if m == 'BAP':
        xs = project_to_section(xp, fix, x0[fix], planar)
        return correct_ba(xs, T, mu, fix, x0[fix], TAU, W, name='BAP')
    if m == 'ORIG':
        return correct_orig(xp, T, mu, TAU, W)
    if m == 'SDC':
        return correct_sdc(xp, T, mu, fix, x0[fix], planar, TAU, W)
    if m == 'FP':
        return correct_fp(xp, T, mu, TAU, W)
    raise KeyError(m)


def classify(s, r):
    if not r.converged:
        return 'failed', None
    mm = orbit_identity(np.array(s['x0']), s['T'], r.x0, r.T, s['mu'], s['Ar'], s['Av'])
    return ('recovered' if mm['maxn'] < ID_THRESH else 'other'), mm
