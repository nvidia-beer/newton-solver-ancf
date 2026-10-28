# Example controls and dependencies

Install the `examples` extra to run the bundled scenes. It includes USD loading,
the Newton viewer, and MuJoCo simulation dependencies. No sibling solver package,
geometry package, asset generator, or external checkout is required. Python 3.12,
Newton 1.7.0, and Warp 1.17.0 are pinned; NumPy and SciPy provide host geometry
and sparse static-equilibrium solves. USD on Linux ARM64 additionally needs the
Python shared library (`libpython3.12.so.1.0`), installed by the Docker image.

`elements` and `baymax_drop` support CPU or CUDA. The tire rigs, vehicle, and
live tire optimization use CUDA. Example and asset menus discover packaged
files automatically. `--help` on a selected example lists its controls.

| Example | What it demonstrates |
| --- | --- |
| `elements` | Four clamped cable, beam, shell and cubic-tetrahedron specimens under gravity; per-specimen flexibility sliders, grayscale surfaces and optional director arrows |
| `baymax_drop` | A filled elastic Baymax solid drops onto a floor; its tetrahedral cage is visible around the detailed surface |
| `ancf_shell_drop` | The small 60-quad shell ball falls onto a frictional plane by default; choose other geometry with `--tire-asset` |
| `ancf_rigid_mujoco_tires` | Four detailed Warthog tires (2,128 nodes / 2,016 quads each) on independent spindles, with manual pressure and motion controls |
| `vehicle_ancf_tires` | A rigid vehicle on deformable tires; vehicle, compatible spindle tires and pressure controls come from USD metadata |
| `diffsim_ancf_tire_lift` | Fit tire pressure to a target static rim height, then check the fitted pressure in a live wheel rig; optional telemetry fitting |

The default Baymax mechanics resolution is 6 (84 nodes and 204 tetrahedra); its
9,882-triangle display mesh does not add mechanical DOFs. It is an elastic solid
with no gas inflation. Contact samples cage nodes, so the detailed display
surface can cross the floor between those samples.
The cage is shown by default; hide it with `--no-show-cage` or the viewer checkbox.

The wheel-rig asset menu defaults to the detailed Warthog tire and matching spindle.
Use `--tire-asset warthog_ancf_tire_simple.usda` for the smaller mesh. Pressure stays
under manual control in normal runs. The README recorder cycles tire pressure
while rolling the rigs forward and backward. Its vehicle preview drives, turns
left and right using Warthog's differential wheel braking, then stops. These
sequences use the existing manual controls without changing timesteps or
iteration budgets. Ordinary launches retain manual controls and the vehicle and
tire-fitting examples retain their own defaults.

All scenes advance 1/60 simulation second per frame. Default substeps are 4 for
`elements`, 1 for `baymax_drop`, 6 for the drop/rig/Warthog vehicle, 8 for the
Superjeep vehicle, and 10 for tire-lift preview. Increasing movie playback speed
or replaying saved states does not measure live simulation throughput.

The detailed Sherp wheel rig is experimental: its default 6-substep, 2-Newton,
10-PCG configuration has a recorded non-finite failure at frame 9. The simple
tire is available in the asset menu. Per-tire density overrides in batched tires do not create
independent masses: the implementation assembles the first tire's mass and
tiles it across environments.

The vehicle uses an adaptive fallback when a whole-frame conditional CUDA graph
cannot capture allocations in the installed MuJoCo Warp backend. This preserves
execution but can affect speed. See [validation](validation.md) for the tested
scope; real-time rates depend on scene, resolution, device and viewer cost.
