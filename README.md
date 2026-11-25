# CR3BP-Orbit-Corrector
An orbit correction algorithm for the Circular Restricted Three Body Problem implemented in Python.

This repository holds the code and data for the paper *An Augmented Newton
Shooting Method for Refining Periodic Orbits in the Circular Restricted
Three-Body Problem Across Multiple Planetary Systems* (A. Barath).

The code builds a bordered augmented Newton corrector (phase condition and
family-selection constraint, null-space solve, backtracking line search),
compares it with six baseline correctors on identical perturbed initial
conditions, and checks whether every converged orbit is the intended one.
It covers sixty JPL catalog orbits: four systems (Earth-Moon, Saturn-Titan,
Sun-Earth, Mars-Phobos), three families (L1 Lyapunov, northern L2 Halo,
distant retrograde) and five members of each.

Every number, table and figure in the paper was produced by this code.

## Data

`data/` holds the per-trial results behind every figure and table in the
paper as CSV files: the sixty seed orbits (with their Floquet multipliers and
stability indices), every perturbed start and its outcome for all seven
correctors, and the directional, absolute-perturbation, tolerance and
five-seed studies. `data/README.txt` describes every file and column.

## Requirements

Python 3.10 or later, with the packages in `requirements.txt`:

    pip install -r requirements.txt

The integrator is compiled with Numba; the first run takes a few seconds longer
while it compiles.

## Reproducing the results

    python campaign.py main              # 8,400 perturbed seeds x 7 correctors (several hours)
    python small_chunk.py tol            # integrator tolerance study    (resumable chunks;
    python small_chunk.py dir            # directional study              repeat each until
    python small_chunk.py abs            # fixed absolute perturbation    it prints
    python small_chunk.py orig           # reproduction of the first      "remaining 0")
                                         # version's runs
    python small_chunk.py finalize       # -> results/tolerance.json, directional.json, ...
    python linearity.py                  # linearization limit of every orbit
    python make_results.py               # results/numbers.json, tables.json, basin.json
    python fig_seeds.py; python fig_example.py; python make_figs.py   # Figures 1-6
    python supplement.py                 # Supplementary Data S1 (CSV files)

`results/` already holds the raw campaign records and the summaries the paper
was built from, so the figures and tables can be regenerated without rerunning
the campaign. Decompress the campaign file first:

    gunzip -k results/campaign_main.jsonl.gz
    python make_results.py

The sixty seeds are in `seeds/seeds60.json`, built by `build_seeds.py` from
`seeds/jpl_selection.json` (catalog rows from the
[JPL Three-Body Periodic Orbit API](https://ssd.jpl.nasa.gov/tools/periodic_orbits.html),
retrieved 11-12 October 2026). `seeds/jpl_raw.tgz` holds the raw API responses.

## Files

| File | Purpose |
|---|---|
| `cr3bp_core.py` | equations of motion, state transition matrix, Dormand-Prince 5(4) integrator, xz-plane crossings |
| `correctors.py` | bordered augmented Newton (with and without line search, chord), original method, standard differential correction, fixed-point iteration |
| `metrics.py` | amplitudes, perturbations, phase-aligned orbit identity, nearest JPL neighbour, Floquet multipliers, conditioning |
| `runlib.py` | seed list, corrector dispatch, recovered / different-orbit / failed classification |
| `campaign.py`, `experiments_small.py`, `small_chunk.py` | the experiments |
| `linearity.py` | linearization limit of each orbit |
| `make_results.py`, `make_figs.py`, `fig_*.py`, `figstyle.py`, `supplement.py` | numbers, figures and supplementary data |
| `build_seeds.py` | seed selection from the JPL catalog |
| `reclassify.py` | re-applies the orbit-identity classification to stored records |

## Implementation details that matter for reproducing the results

* **Integrator**: adaptive Dormand-Prince 5(4) pair with the same Butcher
  tableau, error estimate and step-size rules as MATLAB's `ode45`,
  RelTol = AbsTol = 5e-14 unless stated otherwise.
* **Bordered step**: the two constraint rows (phase condition y0 = 0 and the
  held coordinate) are imposed exactly by the null-space method; the six
  periodicity rows are solved by least squares, scaled by the orbit
  amplitudes (`constrained_step` in `correctors.py`).
* **Convergence**: amplitude-scaled residual `||F ./ W|| < 1e-8`, with
  `W = [Ar Ar Ar Av Av Av]`, or unscaled `||F|| < 2e-12` (integration-noise floor).
* **Line search**: Armijo backtracking on the scaled residual, full step first,
  at most 20 halvings. Trial points are propagated without the STM.
* **Propagation cap**: a propagation is abandoned after 50,000 integration
  steps (near-collision trial trajectories) and treated as a failed trial point.
* **Orbit identity**: a converged orbit counts as the seed when the largest
  phase-aligned, amplitude-scaled distance over one period is below 1e-4.
* **Random perturbations**: each perturbation is drawn from a generator seeded
  by (orbit, perturbation size, trial), so every corrector starts from the
  same states.

## License

MIT; see `LICENSE`.
