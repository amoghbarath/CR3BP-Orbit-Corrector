Supplementary Data S1
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
  conditioning of the unconstrained (A) and bordered (B) matrices, kappa (largest
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
  UA unconstrained augmented method (6 x 7 system, no phase condition or
  family constraint); SDC standard (half-period
  symmetric) differential correction; CHORD chord method; FP fixed-point
  iteration; BAP bordered corrector started from the SDC projected guess.

S1_summary_by_size.csv   (Table 3, Figure 3a)
  Outcome percentages and iteration medians/quartiles per corrector and size.

S1_directional.csv       (Directional Sensitivity)
  One component perturbed at a time, Earth-Moon L2 Halo (seed_id 6, nu = 488),
  thirty trials per component at eps = 1e-4 and 1e-2.

S1_absolute_perturbation.csv
  Fixed absolute perturbation sigma = 1e-5 on every component, all sixty
  orbits, twenty trials each.

S1_tolerance.csv         (Figure 6)
  Integrator tolerance study: steps per period, residual floor, outcome,
  and distance of the converged orbit from the 5e-14 reference.

S1_five_seed_reproduction.csv
  Five additional seed orbits corrected from perturbations of sigma = 1e-4 A
  (thirty trials per seed): per-trial status, iteration count and distance
  from the seed for the unconstrained augmented method, with the bordered
  corrector on the same starts for comparison.
