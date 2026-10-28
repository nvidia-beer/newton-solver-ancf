# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Original acceptance checks for the selected tire examples."""

import numpy as np

from . import _vehicle_config as vehicle_config


def check_vehicle_step(example) -> None:
    x_np = example.ancf_solver.node_x.numpy()
    if not np.all(np.isfinite(x_np)):
        raise AssertionError(f"non-finite ANCF node_x at frame {example._frame}, t={example._t:.3f}s")


def check_vehicle_final(example, *, wheel_loads=None) -> None:
    x_np = example.ancf_solver.node_x.numpy()
    xd_np = example.ancf_solver.node_xd.numpy()
    assert np.all(np.isfinite(x_np)), "non-finite node_x at test_final"
    assert np.all(np.isfinite(xd_np)), "non-finite node_xd at test_final"

    if wheel_loads is None:
        wheel_loads = [example.ancf_solver.wheel_wrenches[w].numpy()[0, 4] for w in range(vehicle_config.N_TIRES)]
    # MuJoCo's cached xpos/xquat precede integration; body_q is at t_{n+1}.
    spindle_poses = example.state_0.body_q.numpy()[example._spindle_body_indices_np]
    rest_np = example._bead_rest_np
    nn = example._n_nodes
    # Single-wheel rig metadata excludes the full articulated chassis load.
    fz_exp = float(example.model.body_mass.numpy().sum()) * vehicle_config.GRAVITY / vehicle_config.N_TIRES

    for e in range(vehicle_config.N_TIRES):
        x_e = x_np[e * nn : (e + 1) * nn]
        pos_zu = spindle_poses[e, :3]
        qx, qy, qz, qw = (float(value) for value in spindle_poses[e, 3:])
        u = np.array([qx, qy, qz])
        r_mj = np.stack([rest_np[:, 2], rest_np[:, 0], rest_np[:, 1]], axis=1)
        r_rot = (
            r_mj * (qw * qw - u @ u)
            + 2.0 * (r_mj @ u)[:, None] * u[None, :]
            + 2.0 * qw * np.cross(np.broadcast_to(u, r_mj.shape), r_mj)
        )
        p_mj = r_rot + pos_zu
        expected = np.stack([p_mj[:, 1], p_mj[:, 2], p_mj[:, 0]], axis=1)
        max_drift = float(np.max(np.linalg.norm(x_e[example._bead_idx_np] - expected, axis=1)))
        assert max_drift < 1e-3, f"FAIL tire {e}: bead drift {max_drift * 1e3:.2f} mm > 1 mm"
        fz = abs(float(wheel_loads[e]))
        rel = abs(fz - fz_exp) / max(fz_exp, 1.0)
        assert rel < 0.50, f"FAIL tire {e}: F_z={fz:.1f} N  expected~{fz_exp:.1f} N  err={rel * 100:.1f}% > 50%"

    assert np.all(np.isfinite(example.ancf_solver.global_f_int.numpy())), "non-finite global_f_int at test_final"
    print(f"[PASS] frame={example._frame}  t={example._t:.1f}s  all 4 tires OK")


def check_tire_lift_forward(example) -> None:
    """Check the coupled forward state independently of the training objective."""
    rig = example.rig
    for name in ("node_x", "node_D", "node_xd", "node_Dd", "global_f_int"):
        if not np.isfinite(getattr(rig.ancf_solver, name).numpy()).all():
            raise AssertionError(f"Non-finite {name} in tire-lift preview")
    if not np.isfinite(rig.state_0.body_q.numpy()).all():
        raise AssertionError("Non-finite rigid state in tire-lift preview")
    if not np.isfinite([example.height, example.loss]).all() or example.height <= 0.0:
        raise AssertionError("Invalid rim height/loss in tire-lift preview")
    if rig._n_envs != 1 or rig._current_rpm != [0.0]:
        raise AssertionError("The tire-lift experiment requires one non-spinning tire")


def check_tire_lift_result(example) -> None:
    """A full training run must reach its physical target, not just stay finite."""
    check_tire_lift_forward(example)
    if example._training_requested and example._measurements and example.rig._frame >= 120:
        if not example._fit_done:
            raise AssertionError(f"Static stiffness fitting failed: {example.training_status}")
    if example._training_requested and example.rig._frame >= 960 and not example._measurements:
        if example.training_status != "Converged" or abs(example.height_error) >= 2.5e-4:
            raise AssertionError(f"Pressure learning failed: {example.training_status}, error={example.height_error} m")
