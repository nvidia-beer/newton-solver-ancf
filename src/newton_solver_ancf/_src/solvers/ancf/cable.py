# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Objective twelve-coordinate ANCF cable energy, gradient, and exact tangent.

The two-node cubic Hermite centerline carries axial Green strain and isotropic
curvature bending integrated over reference arc length. An intrinsic scalar
curvature preserves objectivity for curved reference shapes.
There is no material frame and consequently no torsional stiffness.

Straight reference elements use curvature squared directly; the exact tangent is
well defined at zero curvature. Curved reference elements use the smooth norm
``sqrt(curvature_squared + (1e-8 / length)**2)`` for intrinsic curvature. This
regularization permits differentiation through a temporarily straight centerline.
All kernels use float64 and run on CPU or CUDA.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..implicit import EvaluationWorkspace
from ..implicit.sparse import assemble_matrix
from .model import ANCFModel, _element_values
from .solver import SolverANCF


@dataclass
class CableData:
    """Cable section parameters: EA [N], EI [N m²], and fixed density [kg/m]."""

    axial_stiffness: wp.array[wp.float64]
    bending_stiffness: wp.array[wp.float64]
    line_density: wp.array[wp.float64]

    @property
    def parameters(self) -> dict[str, wp.array]:
        return {"axial_stiffness": self.axial_stiffness, "bending_stiffness": self.bending_stiffness}


def cable_shape_functions(xi: float, length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate cubic Hermite interpolation and its reference derivatives.

    Args:
        xi: Dimensionless coordinate in [0, 1].
        length: Reference interpolation length [m].

    Returns:
        Shape values [dimensionless, m], first derivatives [1/m, dimensionless],
        and second derivatives [1/m², 1/m], with position/slope entries alternating.
    """
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    x2, x3 = xi * xi, xi * xi * xi
    shape = np.array(
        [1.0 - 3.0 * x2 + 2.0 * x3, length * (xi - 2.0 * x2 + x3), 3.0 * x2 - 2.0 * x3, length * (-x2 + x3)]
    )
    first = np.array(
        [
            (6.0 * x2 - 6.0 * xi) / length,
            1.0 - 4.0 * xi + 3.0 * x2,
            (6.0 * xi - 6.0 * x2) / length,
            -2.0 * xi + 3.0 * x2,
        ]
    )
    second = np.array(
        [
            (12.0 * xi - 6.0) / length**2,
            (-4.0 + 6.0 * xi) / length,
            (6.0 - 12.0 * xi) / length**2,
            (-2.0 + 6.0 * xi) / length,
        ]
    )
    return shape, first, second


@wp.func
def cable_gauss_point(index: int) -> wp.float64:
    if index == 0:
        return wp.float64(0.0469100770306680036)
    if index == 1:
        return wp.float64(0.2307653449471584545)
    if index == 2:
        return wp.float64(0.5)
    if index == 3:
        return wp.float64(0.7692346550528415455)
    return wp.float64(0.9530899229693319964)


@wp.func
def cable_gauss_weight(index: int) -> wp.float64:
    if index == 0 or index == 4:
        return wp.float64(0.1184634425280945438)
    if index == 1 or index == 3:
        return wp.float64(0.2393143352496832340)
    return wp.float64(0.2844444444444444444)


@wp.func
def cable_shape(xi: wp.float64, length: wp.float64) -> wp.vec4d:
    x2 = xi * xi
    x3 = x2 * xi
    return wp.vec4d(
        wp.float64(1.0) - wp.float64(3.0) * x2 + wp.float64(2.0) * x3,
        length * (xi - wp.float64(2.0) * x2 + x3),
        wp.float64(3.0) * x2 - wp.float64(2.0) * x3,
        length * (-x2 + x3),
    )


@wp.func
def cable_shape_derivatives(xi: wp.float64, length: wp.float64):
    x2 = xi * xi
    first = wp.vec4d(
        (wp.float64(6.0) * x2 - wp.float64(6.0) * xi) / length,
        wp.float64(1.0) - wp.float64(4.0) * xi + wp.float64(3.0) * x2,
        (wp.float64(6.0) * xi - wp.float64(6.0) * x2) / length,
        -wp.float64(2.0) * xi + wp.float64(3.0) * x2,
    )
    second = wp.vec4d(
        (wp.float64(12.0) * xi - wp.float64(6.0)) / (length * length),
        (-wp.float64(4.0) + wp.float64(6.0) * xi) / length,
        (wp.float64(6.0) - wp.float64(12.0) * xi) / (length * length),
        (-wp.float64(2.0) + wp.float64(6.0) * xi) / length,
    )
    return first, second


@wp.func
def _centerline_derivatives(q: wp.array[wp.float64], node_a: int, node_b: int, first: wp.vec4d, second: wp.vec4d):
    position_delta = wp.vec3d(
        q[6 * node_b] - q[6 * node_a], q[6 * node_b + 1] - q[6 * node_a + 1], q[6 * node_b + 2] - q[6 * node_a + 2]
    )
    slope_a = wp.vec3d(q[6 * node_a + 3], q[6 * node_a + 4], q[6 * node_a + 5])
    slope_b = wp.vec3d(q[6 * node_b + 3], q[6 * node_b + 4], q[6 * node_b + 5])
    tangent = first[2] * position_delta + first[1] * slope_a + first[3] * slope_b
    second_derivative = second[2] * position_delta + second[1] * slope_a + second[3] * slope_b
    return tangent, second_derivative


@wp.func
def _curvature_coefficients(
    tangent: wp.vec3d, second: wp.vec3d, rest_a: wp.vec3d, rest_b: wp.vec3d, length: wp.float64
):
    metric = wp.dot(tangent, tangent) + wp.float64(1.0e-24)
    cross = wp.cross(tangent, second)
    curvature_squared = wp.dot(cross, cross) / (metric * metric * metric)
    rest_metric = wp.dot(rest_a, rest_a) + wp.float64(1.0e-24)
    rest_cross = wp.cross(rest_a, rest_b)
    rest_curvature_squared = wp.dot(rest_cross, rest_cross) / (rest_metric * rest_metric * rest_metric)
    energy = wp.float64(0.5) * curvature_squared
    first = wp.float64(0.5)
    second_coefficient = wp.float64(0.0)
    # Branch on fixed geometry: straight rest must retain quadratic bending energy.
    if rest_curvature_squared * length * length > wp.float64(1.0e-20):
        epsilon_squared = wp.float64(1.0e-16) / (length * length)
        curvature = wp.sqrt(curvature_squared + epsilon_squared)
        rest_curvature = wp.sqrt(rest_curvature_squared + epsilon_squared)
        delta = curvature - rest_curvature
        energy = wp.float64(0.5) * delta * delta
        first = wp.float64(0.5) * (wp.float64(1.0) - rest_curvature / curvature)
        second_coefficient = wp.float64(0.25) * rest_curvature / (curvature * curvature * curvature)
    return energy, first, second_coefficient


@wp.func
def _basis(axis: int) -> wp.vec3d:
    result = wp.vec3d(wp.float64(0.0))
    result[axis] = wp.float64(1.0)
    return result


@wp.func
def _point_gradient(
    tangent: wp.vec3d,
    second: wp.vec3d,
    rest_a: wp.vec3d,
    rest_b: wp.vec3d,
    first: wp.vec4d,
    second_shape: wp.vec4d,
    length: wp.float64,
    axial: wp.float64,
    bending: wp.float64,
    local_dof: int,
) -> wp.float64:
    block = local_dof // 3
    axis = local_dof % 3
    metric0 = wp.dot(rest_a, rest_a)
    metric = wp.dot(tangent, tangent) + wp.float64(1.0e-24)
    strain = wp.float64(0.5) * (wp.dot(tangent, tangent) / metric0 - wp.float64(1.0))
    strain_i = first[block] * tangent[axis] / metric0
    cross = wp.cross(tangent, second)
    cross_i = first[block] * wp.cross(_basis(axis), second) + second_shape[block] * wp.cross(tangent, _basis(axis))
    numerator = wp.dot(cross, cross)
    numerator_i = wp.float64(2.0) * wp.dot(cross, cross_i)
    metric_i = wp.float64(2.0) * first[block] * tangent[axis]
    inv_metric = wp.float64(1.0) / metric
    inv_cube = inv_metric * inv_metric * inv_metric
    curvature_i = inv_cube * (numerator_i - wp.float64(3.0) * numerator * metric_i * inv_metric)
    _, bending_first, _ = _curvature_coefficients(tangent, second, rest_a, rest_b, length)
    return axial * strain * strain_i + bending * bending_first * curvature_i


@wp.kernel
def evaluate_element_energy(
    q: wp.array[wp.float64],
    elements: wp.array2d[wp.int32],
    lengths: wp.array[wp.float64],
    axial_stiffness: wp.array[wp.float64],
    bending_stiffness: wp.array[wp.float64],
    rest_q: wp.array[wp.float64],
    energy: wp.array[wp.float64],
):
    """Evaluate elastic energy [J]; launch one thread per element."""
    element = wp.tid()
    length = lengths[element]
    a, b = elements[element, 0], elements[element, 1]
    result = wp.float64(0.0)
    for point in range(5):
        first, second = cable_shape_derivatives(cable_gauss_point(point), length)
        tangent, curvature_vector = _centerline_derivatives(q, a, b, first, second)
        rest_a, rest_b = _centerline_derivatives(rest_q, a, b, first, second)
        metric0 = wp.dot(rest_a, rest_a)
        strain = wp.float64(0.5) * (wp.dot(tangent, tangent) / metric0 - wp.float64(1.0))
        bending_energy, _, _ = _curvature_coefficients(tangent, curvature_vector, rest_a, rest_b, length)
        weight = length * cable_gauss_weight(point) * wp.sqrt(metric0)
        result += weight * (
            wp.float64(0.5) * axial_stiffness[element] * strain * strain + bending_stiffness[element] * bending_energy
        )
    energy[element] = result


@wp.kernel
def evaluate_element_gradient(
    q: wp.array[wp.float64],
    elements: wp.array2d[wp.int32],
    lengths: wp.array[wp.float64],
    axial_stiffness: wp.array[wp.float64],
    bending_stiffness: wp.array[wp.float64],
    rest_q: wp.array[wp.float64],
    gradient: wp.array2d[wp.float64],
):
    """Evaluate energy gradient [N, N m]; launch shape (element_count, 12).

    The physical internal force is the negative of this gradient. This kernel
    supports Warp reverse mode for parameter and state vector-Jacobian products.
    """
    element, local_dof = wp.tid()
    length = lengths[element]
    a, b = elements[element, 0], elements[element, 1]
    result = wp.float64(0.0)
    for point in range(5):
        first, second = cable_shape_derivatives(cable_gauss_point(point), length)
        tangent, curvature_vector = _centerline_derivatives(q, a, b, first, second)
        rest_a, rest_b = _centerline_derivatives(rest_q, a, b, first, second)
        weight = length * cable_gauss_weight(point) * wp.length(rest_a)
        result += weight * _point_gradient(
            tangent,
            curvature_vector,
            rest_a,
            rest_b,
            first,
            second,
            length,
            axial_stiffness[element],
            bending_stiffness[element],
            local_dof,
        )
    gradient[element, local_dof] = result


@wp.kernel(enable_backward=False)
def evaluate_elements(
    q: wp.array[wp.float64],
    elements: wp.array2d[wp.int32],
    lengths: wp.array[wp.float64],
    axial_stiffness: wp.array[wp.float64],
    bending_stiffness: wp.array[wp.float64],
    rest_q: wp.array[wp.float64],
    energy: wp.array[wp.float64],
    gradient: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
):
    """Evaluate energy, gradient, and exact tangent; launch (element_count, 12, 12).

    The tangent includes geometric stiffness and can be indefinite away from an
    energy minimum. Use :func:`evaluate_element_gradient` for reverse-mode products
    instead of differentiating this assembled second-derivative kernel.
    """
    element, i, j = wp.tid()
    length = lengths[element]
    node_a, node_b = elements[element, 0], elements[element, 1]
    block_i, axis_i = i // 3, i % 3
    block_j, axis_j = j // 3, j % 3
    basis_i, basis_j = _basis(axis_i), _basis(axis_j)
    delta_axis = wp.float64(0.0)
    if axis_i == axis_j:
        delta_axis = wp.float64(1.0)
    energy_sum = wp.float64(0.0)
    gradient_sum = wp.float64(0.0)
    hessian_sum = wp.float64(0.0)
    axial = wp.float64(axial_stiffness[element])
    bending = wp.float64(bending_stiffness[element])
    for point in range(5):
        first, second = cable_shape_derivatives(cable_gauss_point(point), length)
        tangent, curvature_vector = _centerline_derivatives(q, node_a, node_b, first, second)
        rest_a, rest_b = _centerline_derivatives(rest_q, node_a, node_b, first, second)
        metric0 = wp.dot(rest_a, rest_a)
        metric = wp.dot(tangent, tangent) + wp.float64(1.0e-24)
        inv_metric = wp.float64(1.0) / metric
        inv_cube = inv_metric * inv_metric * inv_metric
        metric_i = wp.float64(2.0) * first[block_i] * tangent[axis_i]
        metric_j = wp.float64(2.0) * first[block_j] * tangent[axis_j]
        metric_ij = wp.float64(2.0) * first[block_i] * first[block_j] * delta_axis
        strain = wp.float64(0.5) * (wp.dot(tangent, tangent) / metric0 - wp.float64(1.0))
        strain_i = wp.float64(0.5) * metric_i / metric0
        strain_j = wp.float64(0.5) * metric_j / metric0
        strain_ij = wp.float64(0.5) * metric_ij / metric0

        cross = wp.cross(tangent, curvature_vector)
        cross_i = first[block_i] * wp.cross(basis_i, curvature_vector) + second[block_i] * wp.cross(tangent, basis_i)
        cross_j = first[block_j] * wp.cross(basis_j, curvature_vector) + second[block_j] * wp.cross(tangent, basis_j)
        cross_ij = (first[block_i] * second[block_j] - second[block_i] * first[block_j]) * wp.cross(basis_i, basis_j)
        numerator = wp.dot(cross, cross)
        numerator_i = wp.float64(2.0) * wp.dot(cross, cross_i)
        numerator_j = wp.float64(2.0) * wp.dot(cross, cross_j)
        numerator_ij = wp.float64(2.0) * (wp.dot(cross_i, cross_j) + wp.dot(cross, cross_ij))
        curvature_i = inv_cube * (numerator_i - wp.float64(3.0) * numerator * metric_i * inv_metric)
        curvature_j = inv_cube * (numerator_j - wp.float64(3.0) * numerator * metric_j * inv_metric)
        curvature_ij = inv_cube * (
            numerator_ij
            - wp.float64(3.0) * inv_metric * (numerator_i * metric_j + numerator_j * metric_i + numerator * metric_ij)
            + wp.float64(12.0) * numerator * metric_i * metric_j * inv_metric * inv_metric
        )
        bending_energy, bending_first, bending_second = _curvature_coefficients(
            tangent, curvature_vector, rest_a, rest_b, length
        )
        weight = length * cable_gauss_weight(point) * wp.sqrt(metric0)
        energy_sum += weight * (wp.float64(0.5) * axial * strain * strain + bending * bending_energy)
        gradient_sum += weight * (axial * strain * strain_i + bending * bending_first * curvature_i)
        hessian_sum += weight * (
            axial * (strain_i * strain_j + strain * strain_ij)
            + bending * (bending_second * curvature_i * curvature_j + bending_first * curvature_ij)
        )
    hessian[element, i, j] = hessian_sum
    if j == 0:
        gradient[element, i] = gradient_sum
        if i == 0:
            energy[element] = energy_sum


class SolverANCFCable(SolverANCF):
    """Implicit cubic ANCF cable with axial strain and curvature bending."""

    element_type = CableData

    @staticmethod
    def create_model(
        positions: np.ndarray,
        *,
        slopes: np.ndarray | None = None,
        elements: np.ndarray | None = None,
        rest_lengths: float | np.ndarray | None = None,
        young_modulus: float | np.ndarray = 1.0e7,
        radius: float | np.ndarray = 0.002,
        density: float | np.ndarray = 1000.0,
        device: wp.DeviceLike = None,
        requires_grad: bool = False,
    ) -> ANCFModel:
        """Build two-node, twelve-coordinate ANCF cable elements.

        Cubic Hermite centerlines have axial Green strain and isotropic bending, but
        no cross-section twist or shear degrees of freedom. The input centerline is
        stress-free, including its intrinsic scalar curvature. Position gradients are
        derivatives with respect to each element's reference interpolation coordinate
        [m]. Adjacent elements therefore share the same slope vector.

        Args:
            positions: Reference node positions [m], shape (node_count, 3).
            slopes: Reference tangent vectors [dimensionless], shape (node_count, 3).
                Omitted tangents are inferred for a sequential polyline.
            elements: Two node indices per element; defaults to a sequential chain.
                Explicit topology requires explicit slopes.
            rest_lengths: Interpolation lengths [m]; defaults to endpoint distances.
            young_modulus: Young's modulus [Pa], scalar or one value per element.
            radius: Circular cross-section radius [m], scalar or one value per element.
            density: Material density [kg/m³], scalar or one value per element.
            device: Warp device; defaults to the current device.
            requires_grad: Allocate differentiable axial and bending rigidity arrays.
                Reference geometry and density are fixed because mass is precomputed.

        Returns:
            Reference model with consistent mass integrated along the curved centerline.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
            raise ValueError("positions must have shape (node_count >= 2, 3)")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must be finite")
        node_count = len(positions)
        if elements is None:
            topology = np.column_stack((np.arange(node_count - 1), np.arange(1, node_count))).astype(np.int32)
        else:
            raw_topology = np.asarray(elements)
            if raw_topology.ndim != 2 or raw_topology.shape[1] != 2 or len(raw_topology) == 0:
                raise ValueError("elements must have shape (element_count >= 1, 2)")
            if not np.issubdtype(raw_topology.dtype, np.integer):
                raise ValueError("elements must contain integer node indices")
            if np.any(raw_topology < 0) or np.any(raw_topology >= node_count):
                raise ValueError("element node indices are out of range")
            topology = raw_topology.astype(np.int32)
            if slopes is None:
                raise ValueError("explicit element topology requires explicit slopes")
        if np.any(topology[:, 0] == topology[:, 1]):
            raise ValueError("an element must connect two different nodes")
        if len(np.unique(topology)) != node_count:
            raise ValueError("every node must belong to at least one element")
        count = len(topology)
        chords = positions[topology[:, 1]] - positions[topology[:, 0]]
        chord_lengths = np.linalg.norm(chords, axis=1)
        if np.any(chord_lengths <= 0.0):
            raise ValueError("element endpoints must be distinct")
        lengths = _element_values(chord_lengths if rest_lengths is None else rest_lengths, count, "rest_lengths")
        if slopes is None:
            slopes = np.zeros_like(positions)
            directions = chords / chord_lengths[:, None]
            slopes[0], slopes[-1] = directions[0], directions[-1]
            slopes[1:-1] = directions[:-1] + directions[1:]
            norms = np.linalg.norm(slopes, axis=1)
            if np.any(norms < 1.0e-12):
                raise ValueError("a reversing polyline requires explicit slopes")
            slopes /= norms[:, None]
        else:
            slopes = np.asarray(slopes, dtype=np.float64)
            if slopes.shape != positions.shape or not np.all(np.isfinite(slopes)):
                raise ValueError("slopes must be finite and have the same shape as positions")
            if np.any(np.linalg.norm(slopes, axis=1) < 1.0e-12):
                raise ValueError("reference slopes must be nonzero")

        modulus = _element_values(young_modulus, count, "young_modulus")
        radii = _element_values(radius, count, "radius")
        densities = _element_values(density, count, "density")
        area = np.pi * radii**2
        inertia = 0.25 * np.pi * radii**4
        q = np.column_stack((positions, slopes)).reshape(-1)
        dofs = (6 * topology[:, :, None] + np.arange(6)[None, None, :]).reshape(count, 12).astype(np.int32)
        mass = np.empty((count, dofs.shape[1], dofs.shape[1]), dtype=np.float64)
        gauss_points, gauss_weights = np.polynomial.legendre.leggauss(5)
        for element, length in enumerate(lengths):
            local_q = q[dofs[element]].reshape(4, 3)
            local_mass = np.zeros((4, 4))
            for point, weight in zip(gauss_points, gauss_weights, strict=True):
                shape, first, _ = cable_shape_functions(0.5 * (point + 1.0), length)
                # Relative coordinates keep translations out of reference derivatives.
                tangent = first[2] * (local_q[2] - local_q[0]) + first[1] * local_q[1] + first[3] * local_q[3]
                metric = np.linalg.norm(tangent)
                if metric < 1.0e-10:
                    raise ValueError("reference centerline has a degenerate tangent")
                local_mass += np.outer(shape, shape) * (
                    0.5 * weight * length * metric * densities[element] * area[element]
                )
            mass[element] = np.kron(local_mass, np.eye(3))

        target = wp.get_device(device)

        def array(values: np.ndarray, dtype=wp.float64, grad=requires_grad):
            return wp.array(values, dtype=dtype, device=target, requires_grad=grad)

        return ANCFModel(
            formulation="cable",
            rest_q=array(q, grad=False),
            elements=array(topology, wp.int32, False),
            element_dofs=array(dofs, wp.int32, False),
            lengths=array(lengths, grad=False),
            mass_matrix=assemble_matrix(dofs, mass, len(q), target),
            node_count=node_count,
            element_count=count,
            dofs_per_node=6,
            nodes_per_element=2,
            device=target,
            element_data=CableData(
                array(modulus * area), array(modulus * inertia), array(densities * area, grad=False)
            ),
        )

    @staticmethod
    def _evaluate(
        model: ANCFModel, q: wp.array[wp.float64], *, tangent: bool = True
    ) -> tuple[wp.array[wp.float64], wp.array2d[wp.float64], wp.array3d[wp.float64] | None]:
        """Evaluate cable energy [J], local gradients [N, N m], and tangents.

        Args:
            model: Cable reference geometry and material.
            q: Flattened generalized coordinates [m, dimensionless].
            tangent: Compute the exact energy Hessian. When false, return ``None``
                for the Hessian and record differentiable energy/gradient kernels.

        Returns:
            Per-element energy, twelve-entry gradient, and optional 12-by-12 Hessian.
            Gradient signs correspond to the energy derivative, so physical internal
            forces are their negatives. Hessian entries have mixed generalized units
            [N/m, N, N m].
        """
        data = model.element_data
        if not isinstance(data, CableData):
            raise ValueError("SolverANCFCable evaluation requires a cable model")
        if q.dtype != wp.float64 or q.ndim != 1 or q.shape[0] != model.dof_count:
            raise ValueError("q must be a float64 vector with model.dof_count entries")
        requires_grad = q.requires_grad or data.axial_stiffness.requires_grad or data.bending_stiffness.requires_grad
        workspace = EvaluationWorkspace(model.device, requires_grad=requires_grad)
        energy = workspace.array("energy", model.element_count)
        gradient = workspace.array("gradient", (model.element_count, 12))
        args = [q, model.elements, model.lengths, data.axial_stiffness, data.bending_stiffness, model.rest_q]
        hessian = None
        if tangent:
            hessian = workspace.array("hessian", (model.element_count, 12, 12), requires_grad=False)
            wp.launch(
                evaluate_elements,
                dim=(model.element_count, 12, 12),
                inputs=args,
                outputs=[energy, gradient, hessian],
                device=model.device,
            )
        else:
            wp.launch(
                evaluate_element_energy, dim=model.element_count, inputs=args, outputs=[energy], device=model.device
            )
            wp.launch(
                evaluate_element_gradient,
                dim=(model.element_count, 12),
                inputs=args,
                outputs=[gradient],
                device=model.device,
            )
        return energy, gradient, hessian
