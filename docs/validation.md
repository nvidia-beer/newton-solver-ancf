# Validation

The tests exercise the installed package without another solver checkout.
Run the checks from the repository root:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m unittest discover -s tests -v
NEWTON_TEST_DEVICE=cuda:0 uv run --locked python -m unittest discover -s tests -v
```

## Coverage

- Generalized cable, both beam formulations, shell and cubic tetrahedron:
  energy/force/Hessian consistency, mass assembly, implicit material and state
  derivatives, multistep force histories, prescribed boundaries and floor contact.
- Shared HHT predictors, correctors, history weights and consistent-start order.
- Sparse CSR operations and true residual checks, including indefinite adjoints
  and rejection of singular or insufficiently converged solves.
- Tire static equilibrium: pressure/stiffness finite differences, EAS condensation,
  follower-pressure nonsymmetry, cavity rank-one corrections, transpose solves
  and multiple right-hand sides.
- Rigid attachment reactions and frame transforms, articulation, asset discovery,
  example timesteps and package exports. CUDA-only dynamics checks skip on CPU.

Tests retain the synthetic cable stiffness fit in `tests/cable_fit.py`; it is a
small gradient-validation fixture rather than another menu example.

## Recorded GPU scene checks

These are historical bounded runs from 2026-10-27/28 on Linux ARM64, NVIDIA GB10,
Newton 1.7.0 and Warp 1.17.0. They are correctness observations, not measurements
of current viewer throughput or guarantees for arbitrary slider settings.

| Scene/check | Recorded result |
| --- | --- |
| Generalized elements | Dense mathematical baseline agrees with sparse mass, residual, Hessian and energy; ten-frame position differences below $10^{-13}$ m |
| Shared tire HHT | 60 captured arrays agree bitwise over three CPU/CUDA steps with nonzero histories and prescribed nodes |
| `elements` | 120 CUDA frames passed at four substeps per specimen |
| `baymax_drop` | 180 CUDA frames passed at resolution 6; a 60-frame sparse-path check had minimum sampled $\det F=0.949109$ and lowest cage node −0.654 mm |
| `ancf_shell_drop` | 180 frames passed; maximum penetration 26.92 mm and positive sampled shell Jacobians |
| `ancf_rigid_mujoco_tires` (previous simple-tire default) | 180 frames passed; a 60-frame check had zero bead drift and floor load within 0.2% of rigid weight |
| `vehicle_ancf_tires` | 180 Warthog frames passed using the conditional-graph fallback; a 60-frame check had zero bead drift and approximately 617 N per tire |
| `diffsim_ancf_tire_lift` | 960 frames passed; ten optimization updates; final live rim-height error approximately 0.019 mm |
| Superjeep / Sherp simple | Rig and vehicle each passed 60 frames; Superjeep vehicle used eight substeps |
| Sherp detailed | Vehicle passed 60 frames; wheel rig became non-finite at frame 9 with 6/2/10 substep/Newton/PCG settings |
| Warthog detailed | Rig and vehicle passed a ten-frame launch smoke check |
| Package | 74 Python modules, 27 public exports, six discovered examples and 13 USD assets validated |

On 2026-10-28, the cleaned package passed 52 tests on CPU (two CUDA-only
checks skipped; 64.985 s) and 52 tests on CUDA (111.541 s, including compilation).
All six example clips were rendered at their native resolution and solver
budgets. The Docker image passed package imports, USD loading and discovery
without host mounts or network access; the launcher passed a two-frame CPU
`elements --test` smoke run. These are validation timings, not simulation
throughput benchmarks.

The updated example defaults were checked on the same GPU on 2026-10-28:

- The 62-node/60-quad ball completed 240 frames with `--test` at the unchanged
  6-substep, 3-Newton, 200-PCG budget.
- Baymax completed 180 frames with its cage visible; minimum sampled
  $\det F=0.916657$, lowest cage node −0.955 mm.
- Four detailed Warthog tires (2,128 nodes / 2,016 quads each) completed a
  720-frame recording at 6 substeps / 2 Newton / 10 PCG. Nominal pressure changed
  30 → 45 → 15 → 30 kPa while the spindles rolled forward and backward at up to
  ±8 RPM through the existing command ramps. Final bead drift was 0.000 mm and
  contact load 126.6 N matched the rigid weight. These scripted controls run only
  in the recorder; ordinary launches retain manual pressure and motion controls.
- The Warthog vehicle completed 720 frames with left/right steering and a stop,
  using a wheel-speed command up to 6 rad/s at the unchanged 6/2/10 budget.
  Chassis orientation measured from its quaternion reached approximately +81°
  and then −30°; the fixed camera shows the actual turns. All four tires passed
  the final checks with 0.00 mm bead drift.
- The 52-test CPU suite passed with two CUDA-only checks skipped. The four
  recordings also passed their GPU `test_final` checks; this change did not
  rerun the complete CUDA unit suite.

## Limits

The generalized adjoint assumes a converged locally nonsingular residual with
fixed geometry, topology, mass and contact parameters. Tire fitting differentiates
static equilibrium with fixed contact and EAS-clipping branches; the live MuJoCo
trajectory is a forward validation only. A passing static pressure gradient does
not validate dynamic contact gradients.

The shell-drop default uses the authored contact stiffness and 3 Newton / 200
PCG iterations. The rigs and vehicles use 2 / 10; tire-lift uses 2 / 20 in its
soft-material demonstration and 2 / 10 for telemetry. These finite budgets and
penalty laws do not guarantee nonpenetration or stability for every asset.
No updated throughput benchmark is included in these correctness results.
