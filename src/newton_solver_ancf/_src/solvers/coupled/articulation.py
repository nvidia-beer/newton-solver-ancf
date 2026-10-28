# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Articulation kinematics and inertia kernels for rigid-deformable coupling.

Joint definitions use Newton's public JointType.
"""

import warp as wp
from newton import JointType


@wp.kernel
def compute_body_spatial_inertia(
    body_inertia: wp.array[wp.mat33],
    body_mass: wp.array[float],
    body_q: wp.array[wp.transform],
    # outputs
    body_I_s: wp.array[wp.spatial_matrix],
):
    """Compute COM-referenced spatial inertia for each body in world frame."""
    tid = wp.tid()

    I_local = body_inertia[tid]
    m = body_mass[tid]
    X_wb = body_q[tid]
    q = wp.transform_get_rotation(X_wb)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
    R = wp.matrix_from_cols(r1, r2, r3)
    I_world = R * I_local * wp.transpose(R)

    # Spatial inertia about the COM in world coordinates.
    # fmt: off
    body_I_s[tid] = wp.spatial_matrix(
        m,   0.0, 0.0, 0.0,           0.0,           0.0,
        0.0, m,   0.0, 0.0,           0.0,           0.0,
        0.0, 0.0, m,   0.0,           0.0,           0.0,
        0.0, 0.0, 0.0, I_world[0, 0], I_world[0, 1], I_world[0, 2],
        0.0, 0.0, 0.0, I_world[1, 0], I_world[1, 1], I_world[1, 2],
        0.0, 0.0, 0.0, I_world[2, 0], I_world[2, 1], I_world[2, 2],
    )


@wp.func
def jcalc_integrate(
    parent: int,
    joint_X_c: wp.transform,
    body_com_child: wp.vec3,
    type: int,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    dt: float,
    # outputs
    joint_q_new: wp.array[float],
    joint_qd_new: wp.array[float],
):
    if type == JointType.FIXED:
        return

    # prismatic / revolute
    if type == JointType.PRISMATIC or type == JointType.REVOLUTE:
        qdd = joint_qdd[dof_start]
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        qd_new = qd + qdd * dt
        q_new = q + qd_new * dt

        joint_qd_new[dof_start] = qd_new
        joint_q_new[coord_start] = q_new

        return

    # ball
    if type == JointType.BALL:
        m_j = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        w_j = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        r_j = wp.quat(
            joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2], joint_q[coord_start + 3]
        )

        # symplectic Euler
        w_j_new = w_j + m_j * dt

        drdt_j = wp.quat(w_j_new, 0.0) * r_j * 0.5

        # new orientation (normalized)
        r_j_new = wp.normalize(r_j + drdt_j * dt)

        # update joint coords
        joint_q_new[coord_start + 0] = r_j_new[0]
        joint_q_new[coord_start + 1] = r_j_new[1]
        joint_q_new[coord_start + 2] = r_j_new[2]
        joint_q_new[coord_start + 3] = r_j_new[3]

        # update joint vel
        joint_qd_new[dof_start + 0] = w_j_new[0]
        joint_qd_new[dof_start + 1] = w_j_new[1]
        joint_qd_new[dof_start + 2] = w_j_new[2]

        return

    if type == JointType.FREE or type == JointType.DISTANCE:
        if parent < 0:
            a_parent = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
            alpha = wp.vec3(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5])

            v_parent = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])
            omega = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

            p = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])
            r = wp.quat(
                joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
            )

            r_com_joint = wp.transform_point(wp.transform_inverse(joint_X_c), body_com_child)
            x_com = p + wp.quat_rotate(r, r_com_joint)
            v_com = v_parent + wp.cross(omega, x_com)
            a_com = a_parent + wp.cross(alpha, x_com) + wp.cross(omega, v_com)

            omega_new = omega + alpha * dt
            v_com_new = v_com + a_com * dt

            drdt = wp.quat(omega_new, 0.0) * r * 0.5
            r_new = wp.normalize(r + drdt * dt)
            x_com_new = x_com + v_com_new * dt
            p_new = x_com_new - wp.quat_rotate(r_new, r_com_joint)
            v_parent_new = v_com_new - wp.cross(omega_new, x_com_new)

            joint_q_new[coord_start + 0] = p_new[0]
            joint_q_new[coord_start + 1] = p_new[1]
            joint_q_new[coord_start + 2] = p_new[2]

            joint_q_new[coord_start + 3] = r_new[0]
            joint_q_new[coord_start + 4] = r_new[1]
            joint_q_new[coord_start + 5] = r_new[2]
            joint_q_new[coord_start + 6] = r_new[3]

            joint_qd_new[dof_start + 0] = v_parent_new[0]
            joint_qd_new[dof_start + 1] = v_parent_new[1]
            joint_qd_new[dof_start + 2] = v_parent_new[2]
            joint_qd_new[dof_start + 3] = omega_new[0]
            joint_qd_new[dof_start + 4] = omega_new[1]
            joint_qd_new[dof_start + 5] = omega_new[2]
            return

        a_s = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        m_s = wp.vec3(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5])

        v_s = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])
        w_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # Descendants stay in Featherstone's internal parent-origin coordinates
        # during the integrator step. The public COM convention is restored at
        # the solver boundary once the end-step parent pose is known.
        w_s = w_s + m_s * dt
        v_s = v_s + a_s * dt

        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        dpdt_s = v_s + wp.cross(w_s, p_s)
        r_s = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )

        drdt_s = wp.quat(w_s, 0.0) * r_s * 0.5

        p_s_new = p_s + dpdt_s * dt
        r_s_new = wp.normalize(r_s + drdt_s * dt)

        joint_q_new[coord_start + 0] = p_s_new[0]
        joint_q_new[coord_start + 1] = p_s_new[1]
        joint_q_new[coord_start + 2] = p_s_new[2]

        joint_q_new[coord_start + 3] = r_s_new[0]
        joint_q_new[coord_start + 4] = r_s_new[1]
        joint_q_new[coord_start + 5] = r_s_new[2]
        joint_q_new[coord_start + 6] = r_s_new[3]

        joint_qd_new[dof_start + 0] = v_s[0]
        joint_qd_new[dof_start + 1] = v_s[1]
        joint_qd_new[dof_start + 2] = v_s[2]
        joint_qd_new[dof_start + 3] = w_s[0]
        joint_qd_new[dof_start + 4] = w_s[1]
        joint_qd_new[dof_start + 5] = w_s[2]

        return

    # other joint types (compound, universal, D6)
    if type == JointType.D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            qdd = joint_qdd[dof_start + i]
            qd = joint_qd[dof_start + i]
            q = joint_q[coord_start + i]

            qd_new = qd + qdd * dt
            q_new = q + qd_new * dt

            joint_qd_new[dof_start + i] = qd_new
            joint_q_new[coord_start + i] = q_new

        return
