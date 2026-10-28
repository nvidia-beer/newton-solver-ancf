# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""HHT kinematics and residuals shared by generalized elements and tire shells.

The scalar type follows ``dt`` or ``alpha``. The same device functions support
flat float64 generalized coordinates and float32 position/director vectors;
array layout and nonlinear iteration policy remain with each solver.
"""

from typing import Any

import warp as wp

HHT_ALPHA = -0.2
HHT_BETA = (1.0 - HHT_ALPHA) ** 2 / 4.0
HHT_GAMMA = (1.0 - 2.0 * HHT_ALPHA) / 2.0


@wp.func
def predict_position(q: Any, velocity: Any, acceleration: Any, dt: Any, beta: Any):
    return q + dt * velocity + (dt * dt * (type(dt)(0.5) - beta)) * acceleration


@wp.func
def predict_velocity(velocity: Any, acceleration: Any, dt: Any, gamma: Any):
    return velocity + (dt * (type(dt)(1.0) - gamma)) * acceleration


@wp.func
def position_increment(acceleration: Any, dt: Any, beta: Any):
    return (beta * dt * dt) * acceleration


@wp.func
def velocity_increment(acceleration: Any, dt: Any, gamma: Any):
    return (gamma * dt) * acceleration


@wp.func
def correct_position(predicted: Any, acceleration: Any, dt: Any, beta: Any):
    return predicted + position_increment(acceleration, dt, beta)


@wp.func
def correct_velocity(predicted: Any, acceleration: Any, dt: Any, gamma: Any):
    return predicted + velocity_increment(acceleration, dt, gamma)


@wp.func
def acceleration_from_position(q: Any, predicted: Any, dt: Any, beta: Any):
    return (q - predicted) / (beta * dt * dt)


@wp.func
def weighted_residual(inertia: Any, internal: Any, external: Any, old_internal: Any, old_external: Any, alpha: Any):
    return inertia + (type(alpha)(1.0) + alpha) * (internal - external) - alpha * (old_internal - old_external)


def mass_scale(dt: Any, damping: Any, alpha: Any, beta: Any, gamma: Any):
    """Displacement tangent coefficient for consistent mass and mass damping."""
    return type(dt)(1.0) / (beta * dt * dt) + (type(dt)(1.0) + alpha) * damping * gamma / (beta * dt)


def weighted_stiffness(alpha: Any, damping_increment: Any):
    return (type(alpha)(1.0) + alpha) * (type(alpha)(1.0) + damping_increment)


hht_weighted_stiffness = wp.func(weighted_stiffness)


def stiffness_scale(dt: Any, damping: Any, alpha: Any, beta: Any, gamma: Any):
    """Elastic tangent coefficient including stiffness-proportional damping."""
    return weighted_stiffness(alpha, damping * gamma / (beta * dt))


def stiffness_scale_from_velocity(damping: Any, alpha: Any, velocity_factor: Any):
    """Use an existing velocity factor without regrouping its multiplications."""
    return weighted_stiffness(alpha, damping * velocity_factor)


# Use the same arithmetic in host launch preparation and device assembly.
hht_mass_scale = wp.func(mass_scale)


@wp.func
def hht_stiffness_scale(dt: Any, damping: Any, alpha: Any, beta: Any, gamma: Any):
    return hht_weighted_stiffness(alpha, damping * gamma / (beta * dt))


@wp.kernel(enable_backward=False)
def hht_predict(
    x_n: wp.array[wp.vec3],
    xd_n: wp.array[wp.vec3],
    xdd_n: wp.array[wp.vec3],
    dt: float,
    beta: float,
    gamma: float,
    x_pred: wp.array[wp.vec3],
    xd_pred: wp.array[wp.vec3],
):
    """Vector-array adapter for the shared Newmark predictor."""
    i = wp.tid()
    x_pred[i] = predict_position(x_n[i], xd_n[i], xdd_n[i], dt, beta)
    xd_pred[i] = predict_velocity(xd_n[i], xdd_n[i], dt, gamma)


@wp.kernel(enable_backward=False)
def hht_kinematic(
    x_pred: wp.array[wp.vec3],
    xd_pred: wp.array[wp.vec3],
    xdd_new: wp.array[wp.vec3],
    dt: float,
    beta: float,
    gamma: float,
    x_new: wp.array[wp.vec3],
    xd_new: wp.array[wp.vec3],
):
    """Vector-array adapter for the acceleration-form corrector."""
    i = wp.tid()
    x_new[i] = correct_position(x_pred[i], xdd_new[i], dt, beta)
    xd_new[i] = correct_velocity(xd_pred[i], xdd_new[i], dt, gamma)


@wp.kernel(enable_backward=False)
def apply_acceleration_update(
    da_flat: wp.array[float],
    inv_bdt2: float,
    node_xdd: wp.array[wp.vec3],
    node_Ddd: wp.array[wp.vec3],
):
    """Convert displacement corrections to position/director accelerations."""
    i = wp.tid()
    base = i * 6
    du = wp.vec3(da_flat[base], da_flat[base + 1], da_flat[base + 2])
    dD = wp.vec3(da_flat[base + 3], da_flat[base + 4], da_flat[base + 5])
    node_xdd[i] = node_xdd[i] + inv_bdt2 * du
    node_Ddd[i] = node_Ddd[i] + inv_bdt2 * dD


@wp.kernel(enable_backward=False)
def dirichlet_pred_override(
    dirichlet_idx: wp.array[wp.int32],
    node_x: wp.array[wp.vec3],
    node_xd: wp.array[wp.vec3],
    node_D: wp.array[wp.vec3],
    node_Dd: wp.array[wp.vec3],
    node_xdd: wp.array[wp.vec3],
    node_Ddd: wp.array[wp.vec3],
    dt: float,
    beta: float,
    gamma: float,
    x_pred: wp.array[wp.vec3],
    xd_pred: wp.array[wp.vec3],
    D_pred: wp.array[wp.vec3],
    Dd_pred: wp.array[wp.vec3],
):
    """Invert Newmark at prescribed nodes, retaining acceleration reactions."""
    idx = dirichlet_idx[wp.tid()]
    x_pred[idx] = node_x[idx] - position_increment(node_xdd[idx], dt, beta)
    xd_pred[idx] = node_xd[idx] - velocity_increment(node_xdd[idx], dt, gamma)
    D_pred[idx] = node_D[idx] - position_increment(node_Ddd[idx], dt, beta)
    Dd_pred[idx] = node_Dd[idx] - velocity_increment(node_Ddd[idx], dt, gamma)


@wp.kernel(enable_backward=False)
def build_residual(
    M_a: wp.array[float],
    f_int_new: wp.array[float],
    f_int_old: wp.array[float],
    f_ext: wp.array[float],
    f_ext_old: wp.array[float],
    alpha: float,
    residual: wp.array[float],
):
    """Flat-array adapter for the shared old/new force-weighted residual."""
    i = wp.tid()
    residual[i] = weighted_residual(M_a[i], f_int_new[i], f_ext[i], f_int_old[i], f_ext_old[i], alpha)
