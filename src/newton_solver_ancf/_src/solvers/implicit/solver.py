# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit integration and equilibrium adjoints shared by nonlinear FEM elements."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverBase
from warp._src import context as wp_context
from warp.sparse import bsr_copy

from .hht import (
    HHT_ALPHA,
    HHT_BETA,
    HHT_GAMMA,
    acceleration_from_position,
    correct_velocity,
    hht_mass_scale,
    predict_position,
    predict_velocity,
    weighted_residual,
)
from .model import ImplicitModel
from .sparse import assembly_indices, solve
from .workspace import EvaluationWorkspace

# Mildly indefinite contact/buckling tangents need a nearby positive shift.
# Jumping from 0.01 to 1.0 can reduce Newton to hundreds of tiny descent steps.
_REGULARIZATION_SCALES = (0.0, 1.0e-4, *(1.0e-2 * 2**i for i in range(14)), 100.0)


@dataclass(frozen=True)
class ImplicitState:
    """An owned HHT state; retain every array until a recorded rollout's backward.

    Attributes:
        q: Generalized positions [m, dimensionless position gradients].
        qd: Generalized velocities [m/s, 1/s].
        qdd: Generalized accelerations [m/s², 1/s²].
        internal: Accepted elastic and damping forces [N, N m].
        external: Accepted applied, gravity and floor forces [N, N m].
    """

    q: wp.array[wp.float64]
    qd: wp.array[wp.float64]
    qdd: wp.array[wp.float64]
    internal: wp.array[wp.float64]
    external: wp.array[wp.float64]


@dataclass(frozen=True)
class ImplicitStepInfo:
    """Convergence information from the most recent nonlinear solve.

    Attributes:
        iterations: Number of accepted Newton updates.
        residual: Maximum free residual, with gradient-coordinate loads scaled
            by the reference element length [N].
        objective: HHT incremental potential [J].
    """

    iterations: int
    residual: float
    objective: float


@wp.kernel
def _assemble_force(dofs: wp.array2d[int], local: wp.array2d[wp.float64], force: wp.array[wp.float64]):
    e, i = wp.tid()
    wp.atomic_add(force, dofs[e, i], local[e, i])


@wp.kernel(enable_backward=False)
def _assemble_tangent(slots: wp.array3d[int], local: wp.array3d[wp.float64], matrix: wp.array[wp.float64]):
    e, i, j = wp.tid()
    wp.atomic_add(matrix, slots[e, i, j], local[e, i, j])


@wp.func
def _ground_terms(height: wp.float64, z: wp.float64, stiffness: wp.float64, smoothing: wp.float64):
    """Smooth nodal floor potential and its first two derivatives with respect to z."""
    depth = height - z
    root = wp.sqrt(depth * depth + smoothing * smoothing)
    penetration = wp.float64(0.5) * (depth + root)
    first = wp.float64(0.5) * (wp.float64(1.0) + depth / root)
    second = wp.float64(0.5) * smoothing * smoothing / (root * root * root)
    return wp.vec3d(
        wp.float64(0.5) * stiffness * penetration * penetration,
        -stiffness * penetration * first,
        stiffness * (first * first + penetration * second),
    )


@wp.kernel
def _kinematics(
    q: wp.array[wp.float64],
    previous: wp.array[wp.float64],
    velocity: wp.array[wp.float64],
    acceleration: wp.array[wp.float64],
    dt: wp.float64,
    next_velocity: wp.array[wp.float64],
    next_acceleration: wp.array[wp.float64],
):
    i = wp.tid()
    predicted_q = predict_position(previous[i], velocity[i], acceleration[i], dt, wp.float64(HHT_BETA))
    predicted_v = predict_velocity(velocity[i], acceleration[i], dt, wp.float64(HHT_GAMMA))
    a = acceleration_from_position(q[i], predicted_q, dt, wp.float64(HHT_BETA))
    next_acceleration[i] = a
    next_velocity[i] = correct_velocity(predicted_v, a, dt, wp.float64(HHT_GAMMA))


@wp.kernel
def _force_history(
    q: wp.array[wp.float64],
    velocity: wp.array[wp.float64],
    mass: wp.array[wp.float64],
    mass_rows: wp.array[int],
    mass_columns: wp.array[int],
    elastic: wp.array[wp.float64],
    applied: wp.array[wp.float64],
    gravity: wp.vec3d,
    damping: wp.float64,
    dofs_per_node: int,
    ground: bool,
    height: wp.float64,
    stiffness: wp.float64,
    smoothing: wp.float64,
    internal: wp.array[wp.float64],
    external: wp.array[wp.float64],
):
    i = wp.tid()
    drag, weight = wp.float64(0.0), wp.float64(0.0)
    for entry in range(mass_rows[i], mass_rows[i + 1]):
        j = mass_columns[entry]
        drag += damping * mass[entry] * velocity[j]
        axis = j % dofs_per_node
        if axis < 3:
            weight += mass[entry] * gravity[axis]
    internal[i] = elastic[i] + drag
    force = applied[i] + weight
    if ground and i % dofs_per_node == 2:
        force -= _ground_terms(height, q[i], stiffness, smoothing)[1]
    external[i] = force


@wp.kernel
def _dynamic_residual(
    q: wp.array[wp.float64],
    previous: wp.array[wp.float64],
    velocity: wp.array[wp.float64],
    acceleration: wp.array[wp.float64],
    old_internal: wp.array[wp.float64],
    old_external: wp.array[wp.float64],
    mass: wp.array[wp.float64],
    mass_rows: wp.array[int],
    mass_columns: wp.array[int],
    elastic: wp.array[wp.float64],
    applied: wp.array[wp.float64],
    gravity: wp.vec3d,
    dt: wp.float64,
    damping: wp.float64,
    dofs_per_node: int,
    ground: bool,
    height: wp.float64,
    stiffness: wp.float64,
    smoothing: wp.float64,
    residual: wp.array[wp.float64],
    potential: wp.array[wp.float64],
):
    i = wp.tid()
    inertial, drag, predicted_drag, weight = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    for entry in range(mass_rows[i], mass_rows[i + 1]):
        j = mass_columns[entry]
        coefficient = mass[entry]
        predicted_q = predict_position(previous[j], velocity[j], acceleration[j], dt, wp.float64(HHT_BETA))
        predicted_v = predict_velocity(velocity[j], acceleration[j], dt, wp.float64(HHT_GAMMA))
        a = acceleration_from_position(q[j], predicted_q, dt, wp.float64(HHT_BETA))
        inertial += coefficient * a
        drag += damping * coefficient * (correct_velocity(predicted_v, a, dt, wp.float64(HHT_GAMMA)))
        predicted_drag += damping * coefficient * predicted_v
        axis = j % dofs_per_node
        if axis < 3:
            weight += coefficient * gravity[axis]
    force = applied[i] + weight
    history = old_internal[i] - old_external[i]
    current_weight = wp.float64(1.0) + wp.float64(HHT_ALPHA)
    value = weighted_residual(
        inertial, elastic[i] + drag, force, old_internal[i], old_external[i], wp.float64(HHT_ALPHA)
    )
    predicted_q_i = predict_position(previous[i], velocity[i], acceleration[i], dt, wp.float64(HHT_BETA))
    increment = q[i] - predicted_q_i
    # Gradient with respect to q is the HHT residual. Old forces are independent
    # state inputs; elastic energy is weighted separately by (1 + alpha).
    energy = (
        wp.float64(0.5) * increment * inertial
        + current_weight * (wp.float64(0.5) * increment * (drag + predicted_drag) - q[i] * force)
        - wp.float64(HHT_ALPHA) * q[i] * history
    )
    if ground and i % dofs_per_node == 2:
        terms = _ground_terms(height, q[i], stiffness, smoothing)
        energy += current_weight * terms[0]
        value += current_weight * terms[1]
    residual[i] = value
    potential[i] = energy


@wp.kernel(enable_backward=False)
def _effective_tangent(
    rows: wp.array[int],
    columns: wp.array[int],
    elastic: wp.array[wp.float64],
    mass: wp.array[wp.float64],
    q: wp.array[wp.float64],
    dt: wp.float64,
    damping: wp.float64,
    dofs_per_node: int,
    ground: bool,
    height: wp.float64,
    stiffness: wp.float64,
    smoothing: wp.float64,
):
    entry = wp.tid()
    i, j = rows[entry], columns[entry]
    current_weight = wp.float64(1.0) + wp.float64(HHT_ALPHA)
    value = current_weight * elastic[entry]
    value += (
        hht_mass_scale(dt, damping, wp.float64(HHT_ALPHA), wp.float64(HHT_BETA), wp.float64(HHT_GAMMA)) * mass[entry]
    )
    if ground and i == j and i % dofs_per_node == 2:
        value += current_weight * _ground_terms(height, q[i], stiffness, smoothing)[2]
    elastic[entry] = value


@wp.kernel
def _instantaneous_reaction(
    acceleration: wp.array[wp.float64],
    internal: wp.array[wp.float64],
    external: wp.array[wp.float64],
    mass: wp.array[wp.float64],
    mass_rows: wp.array[int],
    mass_columns: wp.array[int],
    reaction: wp.array[wp.float64],
):
    i = wp.tid()
    value = internal[i] - external[i]
    for entry in range(mass_rows[i], mass_rows[i + 1]):
        j = mass_columns[entry]
        value += mass[entry] * acceleration[j]
    reaction[i] = value


@wp.kernel
def _prescribe(
    values: wp.array[wp.float64], boundary: wp.array[wp.float64], fixed: wp.array[int], out: wp.array[wp.float64]
):
    i = wp.tid()
    out[i] = boundary[i] if fixed[i] else values[i]


@wp.kernel(enable_backward=False)
def _trial(q: wp.array[wp.float64], direction: wp.array[wp.float64], alpha: wp.float64, out: wp.array[wp.float64]):
    i = wp.tid()
    out[i] = q[i] + alpha * direction[i]


@wp.kernel(enable_backward=False)
def _residual_norm(
    residual: wp.array[wp.float64], fixed: wp.array[int], scale: wp.array[wp.float64], metrics: wp.array[wp.float64]
):
    i = wp.tid()
    value = wp.float64(0.0)
    if not wp.isfinite(residual[i]):
        value = wp.float64(wp.inf)
    elif fixed[i] == 0:
        value = wp.abs(residual[i]) / scale[i]
    wp.atomic_max(metrics, 0, value)


@wp.kernel(enable_backward=False)
def _accumulate(source: wp.array[Any], target: wp.array[Any]):
    i = wp.tid()
    target[i] += source[i]


@wp.kernel(enable_backward=False)
def _accumulate_boundary(seed: wp.array[wp.float64], fixed: wp.array[int], target: wp.array[wp.float64]):
    i = wp.tid()
    if fixed[i]:
        target[i] += seed[i]


def _clone_materials(fem, pairs):
    """Isolate the explicitly supported material arrays for the residual VJP."""
    parameters = {}
    for name, original in fem.parameters.items():
        if original.requires_grad:
            clone = wp.clone(original, requires_grad=True)
            clone.grad.zero_()
            pairs.append((original, clone))
            parameters[name] = clone
    return replace(fem, element_data=replace(fem.element_data, **parameters))


class SolverImplicit(SolverBase, ABC):
    """Shared converged HHT-alpha and implicit adjoint for concrete elements.

    Hilber-Hughes-Taylor uses alpha=-0.2, beta=0.36 and gamma=0.7 for every
    element. Consistent mass, exact element Hessians and an Armijo Newton solve define the
    forward step. A custom implicit adjoint differentiates the converged residual
    with respect to state, applied loads, prescribed coordinates and enabled
    material arrays. Geometry, mass, timestep and gravity are fixed. Arrays from a
    recorded rollout must remain unchanged until backward. Consistent mass and
    tangents use scalar CSR storage. Checked PCG rejects nonpositive encountered
    curvature before Newton regularization; the adjoint solves the unshifted
    transpose, with checked sparse GMRES available for indefinite tangents.

    ``ground_height`` optionally enables a frictionless Z-up floor at position
    nodes. ``ground_stiffness`` is the penalty stiffness per node [N/m], and
    ``contact_smoothing`` is the smooth positive-part width [m]. These settings
    are fixed during differentiation. Contact has no thickness or friction.
    """

    element_type: type
    _orientation_constrained = False

    @staticmethod
    @abstractmethod
    def _evaluate(model, q, *, tangent):
        """Return local energy, gradient and optional Hessian for this element."""
        raise NotImplementedError

    @staticmethod
    def _feasibility(model, q):
        """Optional orientation constraint evaluated with the physical energy."""
        return None

    def __init__(
        self,
        model: newton.Model,
        fem: ImplicitModel,
        *,
        fixed_dofs: Sequence[int] = (),
        gravity=None,
        mass_damping=0.0,
        max_iterations=30,
        tolerance=1.0e-8,
        ground_height: float | None = None,
        ground_stiffness: float = 1.0e3,
        contact_smoothing: float = 1.0e-5,
    ):
        if not isinstance(fem.element_data, self.element_type):
            raise ValueError(
                f"{type(self).__name__} requires {self.element_type.__name__}, got {type(fem.element_data).__name__}"
            )
        super().__init__(model)
        if model.device != fem.device:
            raise ValueError("Newton model and FEM data must share a device")
        if model.world_count > 1 or model.body_count or model.particle_count:
            raise ValueError("Use one FEM-only Newton model; rigid bodies belong to the explicit attachment adapter")
        if model.custom_frequency_counts.get(f"{fem.namespace}:dof", 0) != fem.dof_count:
            raise ValueError("Call fem.register(builder) before builder.finalize()")
        self.fem = fem
        self._default_control = self.model.control()
        self._gravity_from_model = gravity is None
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if not np.isfinite(mass_damping) or mass_damping < 0:
            raise ValueError("mass_damping must be finite and nonnegative")
        if ground_height is not None and not np.isfinite(ground_height):
            raise ValueError("ground_height must be finite")
        if not np.isfinite(ground_stiffness) or ground_stiffness < 0:
            raise ValueError("ground_stiffness must be finite and nonnegative")
        if not np.isfinite(contact_smoothing) or contact_smoothing <= 0:
            raise ValueError("contact_smoothing must be finite and positive")
        self._ground = ground_height is not None
        self._height = 0.0 if ground_height is None else float(ground_height)
        self._ground_stiffness = float(ground_stiffness)
        self._smoothing = float(contact_smoothing)
        gravity = np.asarray(self.model.gravity.numpy()[0] if gravity is None else gravity, dtype=np.float64)
        if gravity.shape != (3,) or not np.isfinite(gravity).all():
            raise ValueError("gravity must be a finite three-vector")
        fixed = np.asarray(fixed_dofs)
        if fixed.size and (fixed.ndim != 1 or not np.issubdtype(fixed.dtype, np.integer)):
            raise ValueError("fixed_dofs must be integer indices")
        if np.any(fixed < 0) or np.any(fixed >= fem.dof_count):
            raise ValueError("fixed_dofs is out of range")
        self.fixed_dofs = np.unique(fixed).astype(np.int32)
        mask = np.zeros(fem.dof_count, dtype=np.int32)
        mask[self.fixed_dofs] = 1
        self.fixed_mask = wp.array(mask, dtype=int, device=self.device)
        self._free = mask == 0
        self.gravity = wp.vec3d(gravity)
        self.mass_damping, self.max_iterations, self.tolerance = mass_damping, max_iterations, tolerance
        self._mass_rows = fem.mass_matrix.offsets
        self._mass_columns = fem.mass_matrix.columns
        self._assembly_slots, self._matrix_rows = assembly_indices(fem.mass_matrix, fem.element_dofs.numpy())
        scale = np.ones((fem.node_count, fem.dofs_per_node), dtype=np.float64)
        scale[:, 3:] = float(np.mean(fem.lengths.numpy()))
        self._force_scale = scale.ravel()
        self._force_scale_wp = wp.array(self._force_scale, dtype=wp.float64, device=self.device)
        self._zero_force = wp.zeros(fem.dof_count, dtype=wp.float64, device=self.device)
        self.last_step_info = None

    def _inputs(self, state, dt, external_force, boundary_q):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        for field in fields(state):
            self._array(getattr(state, field.name), f"state.{field.name}")
        applied = self._zero_force if external_force is None else external_force
        boundary = self.fem.rest_q if boundary_q is None else boundary_q
        self._array(applied, "external_force")
        self._array(boundary, "boundary_q")
        return applied, boundary

    def _residual(self, model, q, state, dt, applied, *, tangent):
        energy, gradient, hessian = self._evaluate(model, q, tangent=tangent)
        energies = [energy]
        feasibility = self._feasibility(model, q)
        if feasibility is not None:
            energies.append(feasibility)
        n = model.dof_count
        requires_grad = any(a.requires_grad for a in (q, *self._state_arrays(state), applied, gradient))
        buffers = EvaluationWorkspace(self.device, requires_grad=requires_grad)
        elastic = buffers.array("elastic", n, zero=True)
        wp.launch(_assemble_force, gradient.shape, inputs=[model.element_dofs, gradient, elastic], device=self.device)
        residual = buffers.array("residual", n)
        potential = buffers.array("potential", n)
        contact = [model.dofs_per_node, self._ground, self._height, self._ground_stiffness, self._smoothing]
        wp.launch(
            _dynamic_residual,
            n,
            inputs=[
                q,
                state.q,
                state.qd,
                state.qdd,
                state.internal,
                state.external,
                model.mass_matrix.values,
                self._mass_rows,
                self._mass_columns,
                elastic,
                applied,
                self.gravity,
                dt,
                self.mass_damping,
                *contact,
                residual,
                potential,
            ],
            device=self.device,
        )
        matrix = None
        if tangent:
            matrix = bsr_copy(model.mass_matrix)
            matrix.values.zero_()
            wp.launch(
                _assemble_tangent,
                hessian.shape,
                inputs=[self._assembly_slots, hessian, matrix.values],
                device=self.device,
            )
            wp.launch(
                _effective_tangent,
                self._matrix_rows.size,
                inputs=[
                    self._matrix_rows,
                    matrix.columns,
                    matrix.values,
                    model.mass_matrix.values,
                    q,
                    dt,
                    self.mass_damping,
                    *contact,
                ],
                device=self.device,
            )
        return residual, matrix, energies, potential

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
        *,
        observables=None,
    ) -> None:
        """Standard Newton step using the model's registered State/Control namespace.

        Use distinct retained states for recorded rollouts. This minimal solver
        accepts no external collision manifold or requested rigid observables.
        """
        if state_in is state_out:
            raise ValueError("Provide distinct input and output states")
        if contacts is not None or observables is not None:
            raise ValueError("External contacts and observables are not implemented")
        if control is None:
            control = self._default_control
        incoming = getattr(state_in, self.fem.namespace)
        outgoing = getattr(state_out, self.fem.namespace)
        result = self.advance(
            ImplicitState(*(getattr(incoming, field.name) for field in fields(ImplicitState))),
            dt,
            external_force=getattr(control, self.fem.namespace).load,
            boundary_q=getattr(control, self.fem.namespace).boundary_q,
        )
        for field in fields(ImplicitState):
            wp.copy(getattr(outgoing, field.name), getattr(result, field.name))

    def notify_model_changed(self, flags: newton.ModelFlags | int) -> None:
        """Refresh model gravity; FEM material arrays are read on every step."""
        if self._gravity_from_model:
            self.gravity = wp.vec3d(self.model.gravity.numpy()[0])

    def reset(self, state: newton.State, world_mask=None, flags=None) -> None:
        """Reset the registered generalized state for this single FEM world."""
        if world_mask is not None and not np.asarray(world_mask.numpy()).all():
            raise ValueError("Partial-world reset is not supported by this single FEM model")
        if flags is not None and int(flags) == 0:
            return
        if flags is not None and int(flags) != int(newton.StateFlags.ALL):
            raise ValueError("Reset the complete FEM state with flags=None or StateFlags.ALL")
        wp.copy(getattr(state, self.fem.namespace).q, self.fem.rest_q)
        for name in ("qd", "qdd", "internal", "external"):
            getattr(getattr(state, self.fem.namespace), name).zero_()

    def advance(
        self, state: ImplicitState, dt: float, *, external_force=None, boundary_q=None, initial_guess=None
    ) -> ImplicitState:
        """Return an owned, converged state. Generalized loads use N and N m."""
        applied, boundary = self._inputs(state, dt, external_force, boundary_q)
        if initial_guess is not None:
            self._array(initial_guess, "initial_guess")
        if self.device.is_cuda and self.device.is_capturing:
            raise RuntimeError("Convergence checks do not support external CUDA graph capture")
        tape = wp_context.runtime.tape
        requires_grad = tape is not None or any(
            a.requires_grad for a in (*self._state_arrays(state), applied, boundary)
        )
        wp_context.runtime.tape = None
        try:
            q, tangent = self._forward(state, dt, applied, boundary, initial_guess, requires_grad)
        finally:
            wp_context.runtime.tape = tape
        if tape is not None:
            self._record_adjoint(tape, q, tangent, state, dt, applied, boundary)
        velocity, acceleration = (wp.empty_like(q, requires_grad=requires_grad) for _ in range(2))
        wp.launch(
            _kinematics,
            q.size,
            inputs=[q, state.q, state.qd, state.qdd, dt, velocity, acceleration],
            device=self.device,
        )
        result = ImplicitState(q, velocity, acceleration, state.internal, state.external)
        return self.initialize_history(result, external_force=applied)

    def reaction(self, previous: ImplicitState, current: ImplicitState, dt: float, *, external_force=None):
        """Instantaneous dynamic imbalance ``M a + internal - external`` [N, N m].

        Fixed entries are the support loads exerted on the FEM. Free entries need
        not vanish because HHT balances weighted forces. This is not the HHT
        residual and not a timestep-averaged impulse. Current force history must
        describe the accepted current state; a supplied load is reevaluated.
        """
        self._inputs(previous, dt, external_force, None)
        for field in fields(current):
            self._array(getattr(current, field.name), f"current.{field.name}")
        forces = current if external_force is None else self.initialize_history(current, external_force=external_force)
        reaction = wp.empty_like(current.q, requires_grad=any(a.requires_grad for a in self._state_arrays(forces)))
        wp.launch(
            _instantaneous_reaction,
            current.q.size,
            inputs=[
                current.qdd,
                forces.internal,
                forces.external,
                self.fem.mass_matrix.values,
                self._mass_rows,
                self._mass_columns,
                reaction,
            ],
            device=self.device,
        )
        return reaction

    @staticmethod
    def _state_arrays(state):
        return tuple(getattr(state, field.name) for field in fields(ImplicitState))

    def initialize_history(self, state: ImplicitState, *, external_force=None) -> ImplicitState:
        """Return this state with forces evaluated at its current q and qd.

        Use after prescribing the initial state and initial applied load. This
        leaves q, qd and qdd unchanged; it does not solve for a dynamically
        consistent initial acceleration. Call within Tape when initial forces
        depend on optimized states, loads or materials. Returned arrays must
        remain alive until backward, like the other HHT state arrays.
        """
        for field in fields(state):
            self._array(getattr(state, field.name), f"state.{field.name}")
        applied = self._zero_force if external_force is None else external_force
        self._array(applied, "external_force")
        _, gradient, _ = self._evaluate(self.fem, state.q, tangent=False)
        requires_grad = wp_context.runtime.tape is not None or any(
            a.requires_grad for a in (state.q, state.qd, applied, gradient)
        )
        elastic = wp.zeros(self.fem.dof_count, dtype=wp.float64, device=self.device, requires_grad=requires_grad)
        wp.launch(
            _assemble_force, gradient.shape, inputs=[self.fem.element_dofs, gradient, elastic], device=self.device
        )
        internal, external = (wp.empty_like(elastic, requires_grad=requires_grad) for _ in range(2))
        wp.launch(
            _force_history,
            elastic.size,
            inputs=[
                state.q,
                state.qd,
                self.fem.mass_matrix.values,
                self._mass_rows,
                self._mass_columns,
                elastic,
                applied,
                self.gravity,
                self.mass_damping,
                self.fem.dofs_per_node,
                self._ground,
                self._height,
                self._ground_stiffness,
                self._smoothing,
                internal,
                external,
            ],
            device=self.device,
        )
        return replace(state, internal=internal, external=external)

    def _device_metrics(self, residual, energies, potential):
        metrics = wp.zeros(2 + len(energies), dtype=wp.float64, device=self.device)
        wp.launch(
            _residual_norm,
            residual.size,
            inputs=[residual, self.fixed_mask, self._force_scale_wp, metrics],
            device=self.device,
        )
        for index, energy in enumerate((potential, *energies), start=1):
            wp.utils.array_sum(energy, axis=0, out=metrics[index : index + 1])
        return metrics

    def initial_state(self, *, requires_grad: bool = False) -> ImplicitState:
        """Reference q with zero rates, acceleration and force histories.

        This is an explicit cold start, not an equilibrium initialization under
        gravity. Use initialize_history after setting initial state/load; supply
        qdd separately when an acceleration-consistent initial condition is needed.
        """
        return ImplicitState(
            wp.clone(self.fem.rest_q, requires_grad=requires_grad),
            *(wp.zeros_like(self.fem.rest_q, requires_grad=requires_grad) for _ in range(4)),
        )

    def _array(self, values, name):
        if values.dtype != wp.float64 or values.shape != (self.fem.dof_count,) or values.device != self.device:
            raise ValueError(f"{name} must be a float64 array of shape ({self.fem.dof_count},) on {self.device}")

    def _metrics(self, residual, energies, potential):
        """Download only convergence scalars; retain full state arrays on-device."""
        if not self.device.is_cuda:
            r = residual.numpy()
            norm = np.max(np.abs(r[self._free]) / self._force_scale[self._free], initial=0.0)
            if not np.isfinite(r).all():
                norm = np.inf
            objective = float(potential.numpy().sum()) + (1.0 + HHT_ALPHA) * sum(
                float(energy.numpy().sum()) for energy in energies
            )
            return float(norm), objective
        values = self._device_metrics(residual, energies, potential).numpy()
        return float(values[0]), float(values[1] + (1.0 + HHT_ALPHA) * values[2:].sum())

    def _forward(self, state, dt, applied, boundary, initial_guess, requires_grad):
        def evaluate(q, *, tangent):
            residual, matrix, energies, potential = self._residual(self.fem, q, state, dt, applied, tangent=tangent)
            norm, objective = self._metrics(residual, energies, potential)
            return residual, matrix, norm, objective

        def evaluate_initial(q, *, tangent):
            result = evaluate(q, tangent=tangent)
            if self._orientation_constrained and initial_guess is not None and not np.isfinite(result[2:]).all():
                wp.launch(_prescribe, dim=q.size, inputs=[state.q, boundary, self.fixed_mask, q], device=self.device)
                result = evaluate(q, tangent=tangent)
            return result

        q = wp.empty_like(state.q, requires_grad=requires_grad)
        guess = state.q if initial_guess is None else initial_guess
        wp.launch(_prescribe, dim=q.size, inputs=[guess, boundary, self.fixed_mask, q], device=self.device)
        if not requires_grad and self.fixed_dofs.size == self.fem.dof_count:
            residual, _, norm, objective = evaluate_initial(q, tangent=False)
            if not np.isfinite(objective) or not np.isfinite(norm):
                raise RuntimeError("FEM encountered non-finite energy/residual or an inverted tetrahedral domain")
            self.last_step_info = ImplicitStepInfo(0, norm, objective)
            return q, None
        for iteration in range(self.max_iterations + 1):
            residual, matrix, norm, objective = (
                evaluate_initial(q, tangent=True) if iteration == 0 else evaluate(q, tangent=True)
            )
            if not np.isfinite(objective) or not np.isfinite(norm):
                raise RuntimeError("FEM encountered non-finite energy/residual or an inverted tetrahedral domain")
            if norm <= self.tolerance:
                self.last_step_info = ImplicitStepInfo(iteration, norm, objective)
                return q, matrix
            if iteration == self.max_iterations:
                break
            accepted = False
            for regularization in _REGULARIZATION_SCALES:
                direction = solve(matrix, residual, self.fixed_mask, regularization=regularization, multiplier=-1.0)
                if direction is None:
                    continue
                descent = float(wp.utils.array_inner(residual, direction))
                if not np.isfinite(descent) or descent >= 0.0:
                    continue
                alpha = 1.0
                for _ in range(18):
                    trial = wp.empty_like(q, requires_grad=requires_grad)
                    wp.launch(_trial, dim=q.size, inputs=[q, direction, alpha, trial], device=self.device)
                    _, _, trial_norm, value = evaluate(trial, tangent=False)
                    # Near stationarity, energy subtraction loses accuracy before
                    # the force residual does. Accept a strictly improving residual.
                    small_step = abs(alpha * descent) < 1.0e-12 * max(1.0, abs(objective))
                    improved = small_step and trial_norm < norm
                    if (
                        np.isfinite(value)
                        and np.isfinite(trial_norm)
                        and (value <= objective + 1.0e-4 * alpha * descent or improved)
                    ):
                        q, accepted = trial, True
                        break
                    alpha *= 0.5
                if accepted:
                    break
            if not accepted:
                raise RuntimeError(f"FEM line search failed; free residual={norm:.3e} N")
        self.last_step_info = ImplicitStepInfo(self.max_iterations, norm, objective)
        raise RuntimeError(f"FEM did not converge in {self.max_iterations} iterations; residual={norm:.3e} N")

    def _record_adjoint(self, tape, q, matrix, state, dt, applied, boundary):
        inputs = (*self._state_arrays(state), applied, boundary)

        def backward():
            with wp.ScopedDevice(self.device):
                negative_adjoint = solve(matrix, q.grad, self.fixed_mask, multiplier=-1.0, transpose=True)
                if negative_adjoint is None:
                    negative_adjoint = solve(
                        matrix, q.grad, self.fixed_mask, multiplier=-1.0, transpose=True, method="gmres"
                    )
                if negative_adjoint is None:
                    raise RuntimeError("The unshifted FEM transpose solve did not satisfy its residual tolerance")
                pairs = []
                copies = []
                for original in inputs:
                    clone = wp.clone(original, requires_grad=original.requires_grad)
                    if clone.grad is not None:
                        clone.grad.zero_()
                    pairs.append((original, clone))
                    copies.append(clone)
                local_model = _clone_materials(self.fem, pairs)
                # Rebuild constrained coordinates inside the local tape so their
                # cross derivatives enter the boundary VJP automatically.
                frozen_q = wp.clone(q, requires_grad=False)
                trial = wp.empty_like(q, requires_grad=True)
                with wp.Tape() as local_tape:
                    wp.launch(_prescribe, dim=q.size, inputs=[frozen_q, copies[6], self.fixed_mask, trial])
                    residual = self._residual(
                        local_model,
                        trial,
                        ImplicitState(*copies[:5]),
                        dt,
                        copies[5],
                        tangent=False,
                    )[0]
                local_tape.backward(grads={residual: negative_adjoint})
                for original, clone in pairs:
                    if original.grad is not None:
                        # Parameter arrays may have multiple dimensions.
                        wp.launch(
                            _accumulate, dim=original.size, inputs=[clone.grad.flatten(), original.grad.flatten()]
                        )
                if boundary.grad is not None:
                    wp.launch(_accumulate_boundary, dim=q.size, inputs=[q.grad, self.fixed_mask, boundary.grad])

        parameters = tuple(self.fem.parameters.values())
        tape.record_func(backward, [a for a in (*inputs, *parameters, q) if a.requires_grad])
