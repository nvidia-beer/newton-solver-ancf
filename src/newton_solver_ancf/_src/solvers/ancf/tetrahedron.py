# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Four-node cubic ANCF tetrahedra with Cartesian nodal gradients.

The FN interpolation is Eq. (33) and Appendix A.4 of Pappalardo, Wang, and
Shabana, ``Development of ANCF tetrahedral finite elements for the nonlinear
dynamics of flexible structures``. Each node stores position and three full
position gradients: 48 coordinates per element. Straight reference geometry,
finite-strain Saint-Venant--Kirchhoff elasticity, and full integration are used.
"""

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
import warp as wp

from ..implicit import EvaluationWorkspace
from ..implicit.sparse import assemble_matrix
from .beam import _deformation_gradient, _deviatoric_stress, _green_strain, _strain_derivative, _Vec6d
from .model import ANCFModel, _element_values
from .solver import SolverANCF

_Shape16d = wp.types.vector(length=16, dtype=wp.float64)


@wp.func
def _reference_corner(index: int, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d):
    result = a
    if index == 1:
        result = b
    elif index == 2:
        result = c
    elif index == 3:
        result = d
    return result


@wp.func
def tetrahedral_shape_values(
    barycentric: wp.vec4d, corner0: wp.vec3d, corner1: wp.vec3d, corner2: wp.vec3d, corner3: wp.vec3d
):
    """Evaluate sixteen cubic FN interpolation coefficients inside Warp kernels.

    Corners are the original positively oriented reference positions [m]. The
    caller supplies finite barycentrics summing to one. The returned float64
    vector follows ``[position, Dx, Dy, Dz]`` for each of four nodes; position
    coefficients are dimensionless and gradient coefficients have units [m].
    This is the device counterpart of :meth:`SolverANCFTetrahedron.shape_functions` with
    ``derivatives=False``; it performs no host validation or matrix inversion.
    """
    shape = _Shape16d(wp.float64(0.0))
    for node in range(4):
        value = barycentric[node]
        total, pairs = wp.float64(0.0), wp.float64(0.0)
        for a in range(4):
            if a != node:
                total += barycentric[a]
                for b in range(a + 1, 4):
                    if b != node:
                        pairs += barycentric[a] * barycentric[b]
        shape[4 * node] = value * (value * value + wp.float64(3.0) * value * total + wp.float64(2.0) * pairs)
        origin = _reference_corner(node, corner0, corner1, corner2, corner3)
        for other in range(4):
            if other != node:
                factor = wp.float64(3.0) * value - (total - barycentric[other])
                edge = _reference_corner(other, corner0, corner1, corner2, corner3) - origin
                coefficient = value * barycentric[other] * factor / wp.float64(3.0)
                for axis in range(3):
                    shape[4 * node + 1 + axis] += coefficient * edge[axis]
    return shape


@dataclass
class TetrahedralData:
    """Fixed reference quadrature and differentiable elastic moduli [Pa]."""

    derivatives: wp.array3d[wp.vec3d]
    gradient_products: wp.array4d[wp.float64]
    weights: wp.array2d[wp.float64]
    shear_modulus: wp.array[wp.float64]
    bulk_modulus: wp.array[wp.float64]
    active_elements: wp.array[int]

    @property
    def parameters(self) -> dict[str, wp.array]:
        return {"shear_modulus": self.shear_modulus, "bulk_modulus": self.bulk_modulus}


def _reference_coordinates(reference_positions: np.ndarray) -> tuple[np.ndarray, float]:
    corners = np.asarray(reference_positions, dtype=np.float64)
    if corners.shape != (4, 3) or not np.isfinite(corners).all():
        raise ValueError("reference_positions must be a finite (4, 3) array")
    jacobian = (corners[1:] - corners[0]).T
    determinant = np.linalg.det(jacobian)
    if not np.isfinite(determinant) or determinant <= 1.0e-14 * np.linalg.norm(np.ptp(corners, axis=0)) ** 3:
        raise ValueError("reference tetrahedron has a singular or inverted Jacobian")
    inverse = np.linalg.inv(jacobian)
    return np.vstack((-inverse.sum(axis=0), inverse)), determinant


def _shape_functions_batch(
    coordinates: np.ndarray, corners: np.ndarray, *, derivatives: bool = True
) -> tuple[np.ndarray, np.ndarray | None]:
    """Vectorized interpolation for reference quadrature."""
    leading = np.broadcast_shapes(coordinates.shape[:-1], corners.shape[:-2])
    shape = np.zeros((*leading, 16))
    derivative = np.zeros((*leading, 16, 4)) if derivatives else None
    for node in range(4):
        others = [other for other in range(4) if other != node]
        value = coordinates[..., node]
        total = coordinates[..., others].sum(axis=-1)
        pairs = sum(
            coordinates[..., a] * coordinates[..., b]
            for a, b in ((others[0], others[1]), (others[0], others[2]), (others[1], others[2]))
        )
        shape[..., 4 * node] = value * (value**2 + 3.0 * value * total + 2.0 * pairs)
        if derivatives:
            derivative[..., 4 * node, node] = 3.0 * value**2 + 6.0 * value * total + 2.0 * pairs
        for other in others:
            if derivatives:
                derivative[..., 4 * node, other] = 3.0 * value**2 + 2.0 * value * (total - coordinates[..., other])
            remaining = [index for index in others if index != other]
            factor = 3.0 * value - coordinates[..., remaining].sum(axis=-1)
            edge = corners[..., other, :] - corners[..., node, :]
            shape[..., 4 * node + 1 : 4 * node + 4] += (value * coordinates[..., other] * factor / 3.0)[
                ..., None
            ] * edge
            if derivatives:
                gradient = np.zeros((*leading, 4))
                gradient[..., node] = coordinates[..., other] * (factor + 3.0 * value) / 3.0
                gradient[..., other] = value * factor / 3.0
                gradient[..., remaining] = (-value * coordinates[..., other] / 3.0)[..., None]
                derivative[..., 4 * node + 1 : 4 * node + 4, :] += edge[..., :, None] * gradient[..., None, :]
    return shape, derivative


def _shape_functions(coordinates: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Differentiate in four independent volume coordinates, then restrict to sum(lambda)=1.
    shape = np.zeros(16)
    derivative = np.zeros((16, 4))
    for node in range(4):
        others = [other for other in range(4) if other != node]
        value = coordinates[node]
        total = coordinates[others].sum()
        pairs = sum(
            coordinates[a] * coordinates[b]
            for a, b in ((others[0], others[1]), (others[0], others[2]), (others[1], others[2]))
        )
        shape[4 * node] = value * (value**2 + 3.0 * value * total + 2.0 * pairs)
        derivative[4 * node, node] = 3.0 * value**2 + 6.0 * value * total + 2.0 * pairs
        for other in others:
            derivative[4 * node, other] = 3.0 * value**2 + 2.0 * value * (total - coordinates[other])
            remaining = [index for index in others if index != other]
            factor = 3.0 * value - coordinates[remaining].sum()
            edge = corners[other] - corners[node]
            shape[4 * node + 1 : 4 * node + 4] += value * coordinates[other] * factor / 3.0 * edge
            gradient = np.zeros(4)
            gradient[node] = coordinates[other] * (factor + 3.0 * value) / 3.0
            gradient[other] = value * factor / 3.0
            gradient[remaining] = -value * coordinates[other] / 3.0
            derivative[4 * node + 1 : 4 * node + 4] += np.outer(edge, gradient)
    return shape, derivative


def _weighted_gauss(order: int, exponent: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Jacobi rule on [0,1] for weight (1-x)**exponent."""
    index = np.arange(order, dtype=np.float64)
    denominator = 2.0 * index + exponent
    diagonal = np.zeros(order)
    nonzero = denominator != 0
    diagonal[nonzero] = -(exponent**2) / (denominator[nonzero] * (denominator[nonzero] + 2.0))
    index = index[1:]
    denominator = 2.0 * index + exponent
    off_diagonal = np.sqrt(4.0 * index**2 * (index + exponent) ** 2 / (denominator**2 * (denominator**2 - 1.0)))
    values, vectors = np.linalg.eigh(np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1))
    return (values + 1.0) / 2.0, vectors[0] ** 2 / (exponent + 1.0)


def _quadrature(order: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Positive degree-eight symmetric rule; other orders use conical products."""
    if order == 5:
        # Zhang, Cui & Liu, J. Computational Mathematics 27 (2009), §4.2.
        # These seven permutation orbits integrate the cubic FN element's
        # degree-eight energy exactly, with 46 points instead of 125.
        # https://www.researchgate.net/publication/228667586
        orbits = []
        for a, weight in (
            (0.039675423070389901, 0.0063971477799023213),
            (0.31448780069809631, 0.040190448020966172),
            (0.10198669306270330, 0.024307975504770321),
            (0.18420369694919151, 0.054858892413697440),
        ):
            orbits.append(((a, a, a, 1.0 - 3.0 * a), weight))
        a = 0.063436287754539892
        orbits.append(((a, a, 0.5 - a, 0.5 - a), 0.035719612234099182))
        for a, b, weight in (
            (0.021690162067728005, 0.71993192203946594, 0.0071831906978525394),
            (0.20448008063679571, 0.58057719012880922, 0.016372181945319118),
        ):
            orbits.append(((a, a, b, 1.0 - 2.0 * a - b), weight))
        points, weights = [], []
        for coordinates, weight in orbits:
            orbit = sorted(set(permutations(coordinates)))
            points.extend(orbit)
            weights.extend([weight / 6.0] * len(orbit))
        return np.asarray(points), np.asarray(weights)

    points_u, weights_u = _weighted_gauss(order, 2)
    points_v, weights_v = _weighted_gauss(order, 1)
    points_w, weights_w = _weighted_gauss(order, 0)
    points, weights = [], []
    for i, j, k in product(range(order), repeat=3):
        u, v, w = points_u[i], points_v[j], points_w[k]
        points.append([(1 - u) * (1 - v) * (1 - w), u, (1 - u) * v, (1 - u) * (1 - v) * w])
        weights.append(weights_u[i] * weights_v[j] * weights_w[k])
    return np.asarray(points), np.asarray(weights)


@wp.func
def _stress(strain: _Vec6d, shear: wp.float64, bulk: wp.float64):
    result = _deviatoric_stress(strain, shear)
    pressure = bulk * (strain[0] + strain[1] + strain[2])
    result[0] += pressure
    result[1] += pressure
    result[2] += pressure
    return result


@wp.func
def _kinematics_value(
    element: int,
    point: int,
    q: wp.array[wp.float64],
    dofs: wp.array2d[int],
    derivatives: wp.array3d[wp.vec3d],
    shear: wp.array[wp.float64],
    bulk: wp.array[wp.float64],
    deformation: wp.array2d[wp.mat33d],
    stress: wp.array2d[_Vec6d],
):
    value = _deformation_gradient(q, dofs, derivatives, element, point, 4)
    deformation[element, point] = value
    stress[element, point] = _stress(_green_strain(value), shear[element], bulk[element])


@wp.kernel
def _kinematics(
    q: wp.array[wp.float64],
    dofs: wp.array2d[int],
    derivatives: wp.array3d[wp.vec3d],
    shear: wp.array[wp.float64],
    bulk: wp.array[wp.float64],
    active: wp.array[int],
    deformation: wp.array2d[wp.mat33d],
    stress: wp.array2d[_Vec6d],
):
    element, point = wp.tid()
    if active.shape[0] > 0 and active[element] == 0:
        return
    _kinematics_value(element, point, q, dofs, derivatives, shear, bulk, deformation, stress)


@wp.kernel(enable_backward=False)
def _orientation_from_deformation(
    deformation: wp.array2d[wp.mat33d], active: wp.array[int], feasibility: wp.array[wp.float64]
):
    element, point = wp.tid()
    if active.shape[0] == 0 or active[element] != 0:
        determinant = wp.determinant(deformation[element, point])
        if not wp.isfinite(determinant) or determinant <= wp.float64(0.0):
            wp.atomic_max(feasibility, 0, wp.float64(wp.inf))


@wp.kernel(enable_backward=False)
def _orientation_from_coordinates(
    q: wp.array[wp.float64],
    dofs: wp.array2d[int],
    derivatives: wp.array3d[wp.vec3d],
    active: wp.array[int],
    feasibility: wp.array[wp.float64],
):
    element, point = wp.tid()
    if active.shape[0] == 0 or active[element] != 0:
        value = _deformation_gradient(q, dofs, derivatives, element, point, 4)
        determinant = wp.determinant(value)
        if not wp.isfinite(determinant) or determinant <= wp.float64(0.0):
            wp.atomic_max(feasibility, 0, wp.float64(wp.inf))


def _orientation_feasibility(model, q):
    """Reject inverted FN integration points without modifying the elastic energy."""
    data = model.element_data
    result = wp.zeros(1, dtype=wp.float64, device=model.device)
    wp.launch(
        _orientation_from_coordinates,
        data.weights.shape,
        inputs=[q, model.element_dofs, data.derivatives, data.active_elements, result],
        device=model.device,
        record_tape=False,
    )
    return result


@wp.func
def _energy_value(
    element: int,
    deformation: wp.array2d[wp.mat33d],
    stress: wp.array2d[_Vec6d],
    weights: wp.array2d[wp.float64],
    energy: wp.array[wp.float64],
):
    value = wp.float64(0.0)
    for point in range(weights.shape[1]):
        value += (
            wp.float64(0.5)
            * weights[element, point]
            * wp.dot(_green_strain(deformation[element, point]), stress[element, point])
        )
    energy[element] = value


@wp.kernel
def _energy(
    deformation: wp.array2d[wp.mat33d],
    stress: wp.array2d[_Vec6d],
    weights: wp.array2d[wp.float64],
    active: wp.array[int],
    energy: wp.array[wp.float64],
):
    element = wp.tid()
    if active.shape[0] > 0 and active[element] == 0:
        energy[element] = wp.float64(0.0)
        return
    _energy_value(element, deformation, stress, weights, energy)


@wp.func
def _gradient_value(
    element: int,
    i: int,
    deformation: wp.array2d[wp.mat33d],
    stress: wp.array2d[_Vec6d],
    derivatives: wp.array3d[wp.vec3d],
    weights: wp.array2d[wp.float64],
    gradient: wp.array2d[wp.float64],
):
    value = wp.float64(0.0)
    for point in range(weights.shape[1]):
        strain_i = _strain_derivative(deformation[element, point], derivatives[element, point, i // 3], i % 3)
        # Keep differentiable factors inside the reduction: Warp reverse replay
        # does not retain the final primal value of a dynamic-loop accumulator.
        value += weights[element, point] * wp.dot(strain_i, stress[element, point])
    gradient[element, i] = value


@wp.kernel
def _gradient(
    deformation: wp.array2d[wp.mat33d],
    stress: wp.array2d[_Vec6d],
    derivatives: wp.array3d[wp.vec3d],
    weights: wp.array2d[wp.float64],
    active: wp.array[int],
    gradient: wp.array2d[wp.float64],
):
    element, i = wp.tid()
    if active.shape[0] > 0 and active[element] == 0:
        gradient[element, i] = wp.float64(0.0)
        return
    _gradient_value(element, i, deformation, stress, derivatives, weights, gradient)


@wp.func
def _tangent_geometry_point(
    element: int,
    point: int,
    a: int,
    deformation: wp.array2d[wp.mat33d],
    stress: wp.array2d[_Vec6d],
    derivatives: wp.array3d[wp.vec3d],
    projected: wp.array3d[wp.vec3d],
    metric: wp.array2d[wp.mat33d],
    stress_gradients: wp.array3d[wp.vec3d],
):
    f = deformation[element, point]
    da = derivatives[element, point, a]
    projected[element, point, a] = f * da
    if a == 0:
        metric[element, point] = f * wp.transpose(f)
    sigma = stress[element, point]
    tensor = wp.mat33d(sigma[0], sigma[5], sigma[4], sigma[5], sigma[1], sigma[3], sigma[4], sigma[3], sigma[2])
    # Cache 16 stress-gradient vectors instead of a 16-by-16 pair table.
    stress_gradients[element, point, a] = tensor * da


@wp.kernel(enable_backward=False)
def _tangent_geometry(
    deformation: wp.array2d[wp.mat33d],
    stress: wp.array2d[_Vec6d],
    derivatives: wp.array3d[wp.vec3d],
    active: wp.array[int],
    projected: wp.array3d[wp.vec3d],
    metric: wp.array2d[wp.mat33d],
    stress_gradients: wp.array3d[wp.vec3d],
):
    element, point, a = wp.tid()
    if active.shape[0] == 0 or active[element] != 0:
        _tangent_geometry_point(
            element, point, a, deformation, stress, derivatives, projected, metric, stress_gradients
        )


@wp.func
def _hessian_block(
    element: int,
    pair: int,
    projected: wp.array3d[wp.vec3d],
    metric: wp.array2d[wp.mat33d],
    stress_gradients: wp.array3d[wp.vec3d],
    derivatives: wp.array3d[wp.vec3d],
    gradient_products: wp.array4d[wp.float64],
    weights: wp.array2d[wp.float64],
    shear: wp.array[wp.float64],
    bulk: wp.array[wp.float64],
    hessian: wp.array3d[wp.float64],
):
    # Compact upper-triangular indexing keeps every launched thread useful.
    a = wp.int32((wp.float64(33.0) - wp.sqrt(wp.float64(1089 - 8 * pair))) * wp.float64(0.5))
    b = pair - a * (33 - a) // 2 + a
    mu = shear[element]
    lame = bulk[element] - wp.float64(2.0) * mu / wp.float64(3.0)
    value = wp.mat33d(0.0)
    for point in range(weights.shape[1]):
        pa, pb = projected[element, point, a], projected[element, point, b]
        weight = weights[element, point]
        weighted_mu, weighted_lame = weight * mu, weight * lame
        gradient_product = wp.float64(0.0)
        if gradient_products.shape[0] != 0:
            gradient_product = gradient_products[element, point, a, b]
        else:
            gradient_product = wp.dot(derivatives[element, point, a], derivatives[element, point, b])
        metric_scale = weighted_mu * gradient_product
        geometric = weight * wp.dot(stress_gradients[element, point, a], derivatives[element, point, b])
        spatial_metric = metric[element, point]
        # The two material outer products are transposes. Reuse their scalar
        # products and apply geometric stiffness only on the three diagonals,
        # avoiding full-matrix scaling/addition of known zero entries. This is
        # the same exact tangent, with the same quadrature and accumulation
        # order; no sampled material or constitutive term is dropped.
        for i in range(3):
            value[i, i] += (
                (weighted_lame + weighted_mu) * pa[i] * pb[i] + metric_scale * spatial_metric[i, i] + geometric
            )
            for j in range(3):
                if j > i:
                    first, second = pa[i] * pb[j], pb[i] * pa[j]
                    value[i, j] += weighted_lame * first + weighted_mu * second + metric_scale * spatial_metric[i, j]
                    value[j, i] += weighted_lame * second + weighted_mu * first + metric_scale * spatial_metric[j, i]
    for i in range(3):
        for j in range(3):
            entry = value[i, j]
            if a == b and j < i:
                # CUDA may contract the equivalent off-diagonal expressions
                # differently. Mirror one value inside a diagonal block too,
                # preserving exact energy-Hessian symmetry after rounding.
                entry = value[j, i]
            hessian[element, 3 * a + i, 3 * b + j] = entry
            # This is an energy Hessian. Its opposite block is the transpose,
            # including geometric stiffness, so integrate each pair only once.
            if a != b:
                hessian[element, 3 * b + j, 3 * a + i] = entry


@wp.kernel(enable_backward=False)
def _hessian(
    projected: wp.array3d[wp.vec3d],
    metric: wp.array2d[wp.mat33d],
    stress_gradients: wp.array3d[wp.vec3d],
    derivatives: wp.array3d[wp.vec3d],
    gradient_products: wp.array4d[wp.float64],
    weights: wp.array2d[wp.float64],
    shear: wp.array[wp.float64],
    bulk: wp.array[wp.float64],
    active: wp.array[int],
    clear_inactive: bool,
    hessian: wp.array3d[wp.float64],
):
    element, pair = wp.tid()
    if active.shape[0] > 0 and active[element] == 0:
        if clear_inactive:
            a = wp.int32((wp.float64(33.0) - wp.sqrt(wp.float64(1089 - 8 * pair))) * wp.float64(0.5))
            b = pair - a * (33 - a) // 2 + a
            for i in range(3):
                for j in range(3):
                    hessian[element, 3 * a + i, 3 * b + j] = wp.float64(0.0)
                    if a != b:
                        hessian[element, 3 * b + j, 3 * a + i] = wp.float64(0.0)
        return
    _hessian_block(
        element,
        pair,
        projected,
        metric,
        stress_gradients,
        derivatives,
        gradient_products,
        weights,
        shear,
        bulk,
        hessian,
    )


class SolverANCFTetrahedron(SolverANCF):
    """Implicit cubic four-node, 48-DOF ANCF/FN tetrahedron."""

    element_type = TetrahedralData
    _orientation_constrained = True
    _feasibility = staticmethod(_orientation_feasibility)

    @staticmethod
    def create_model(
        positions: np.ndarray,
        elements: np.ndarray,
        *,
        young_modulus: float | np.ndarray = 1.0e7,
        poisson_ratio: float | np.ndarray = 0.3,
        density: float | np.ndarray = 1000.0,
        device: str | wp.Device | None = None,
        requires_grad: bool = False,
        cache_gradient_products: bool = True,
    ) -> ANCFModel:
        """Build cubic four-node, 48-DOF ANCF/FN tetrahedral solid elements.

        Each node stores ``[position, Dx, Dy, Dz]``. The reference tetrahedra are
        straight, so their Cartesian gradients are the identity. The deformed field
        is cubic and supports nonconstant curvature. A positive 46-point symmetric
        rule exactly integrates the degree-eight StVK energy and degree-six
        consistent mass on these straight reference cells. No reduced integration,
        enhanced assumed strain, or nearly incompressible correction is applied.

        Args:
            positions: Reference corner positions [m], shape (node_count, 3).
            elements: Integer vertex indices, shape (element_count, 4). Vertices must
                have positive det([X1-X0, X2-X0, X3-X0]); every node must be referenced.
            young_modulus: Young's modulus [Pa], scalar or per element.
            poisson_ratio: Poisson ratio, scalar or per element in (-1, 0.5).
            density: Reference mass density [kg/m³], scalar or per element.
            device: Warp device.
            requires_grad: Enable modulus gradients. Geometry and mass remain fixed
                when these parameters are optimized.
            cache_gradient_products: Cache reference derivative dot products for
                tangent evaluation. Disabling this recomputes the same products
                and reduces memory without changing quadrature or the formulation.

        Returns:
            A ``tetra_fn`` model for :class:`SolverANCF`, with 12 coordinates
            per node. This is the paper's FN element, not its 60-DOF ENMC variant.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 4 or not np.isfinite(positions).all():
            raise ValueError("positions must be a finite (node_count, 3) array with at least four nodes")
        topology = np.asarray(elements)
        if (
            topology.ndim != 2
            or topology.shape[1] != 4
            or not len(topology)
            or not np.issubdtype(topology.dtype, np.integer)
        ):
            raise ValueError("elements must be a nonempty integer array with four nodes per row")
        if np.any(topology < 0) or np.any(topology >= len(positions)):
            raise ValueError("element node indices are out of range")
        if len(np.unique(topology)) != len(positions) or any(len(np.unique(row)) != 4 for row in topology):
            raise ValueError("elements must reference every node and cannot repeat a node within an element")
        topology = topology.astype(np.int32)
        count, node_count = len(topology), len(positions)
        modulus = _element_values(young_modulus, count, "young_modulus")
        densities = _element_values(density, count, "density")
        poisson = np.broadcast_to(np.asarray(poisson_ratio, dtype=np.float64), (count,)).copy()
        if not np.isfinite(poisson).all() or np.any(poisson <= -1.0) or np.any(poisson >= 0.5):
            raise ValueError("poisson_ratio must be finite and lie in (-1, 0.5)")
        q = np.concatenate((positions[:, None], np.tile(np.eye(3), (node_count, 1, 1))), axis=1).reshape(-1)
        dofs = (12 * topology[:, :, None] + np.arange(12)).reshape(count, 48).astype(np.int32)
        points, reference_weights = _quadrature()
        derivatives = np.empty((count, len(points), 16, 3))
        weights = np.empty((count, len(points)))
        volumes = np.empty(count)
        mass = np.empty((count, dofs.shape[1], dofs.shape[1]), dtype=np.float64)
        mass_blocks = np.empty((count, 16, 16))
        # Bound host scratch memory while constructing full-resolution meshes.
        for first in range(0, count, 256):
            last = min(first + 256, count)
            corners = positions[topology[first:last]]
            determinant = np.linalg.det(np.swapaxes(corners[:, 1:] - corners[:, :1], -1, -2))
            shape, gradient = SolverANCFTetrahedron.shape_functions(points[None], corners[:, None])
            derivatives[first:last] = gradient
            weights[first:last] = determinant[:, None] * reference_weights
            volumes[first:last] = determinant / 6.0
            mass_blocks[first:last] = np.einsum(
                "ep,epa,epb->eab", densities[first:last, None] * weights[first:last], shape, shape
            )
            for element in range(first, last):
                mass[element] = np.kron(mass_blocks[element], np.eye(3))
        target = wp.get_device(device)

        def array(values, dtype=wp.float64, grad=False):
            return wp.array(values, dtype=dtype, device=target, requires_grad=grad)

        return ANCFModel(
            formulation="tetra_fn",
            rest_q=array(q),
            elements=array(topology, wp.int32),
            element_dofs=array(dofs, wp.int32),
            lengths=array(np.cbrt(volumes)),
            mass_matrix=assemble_matrix(dofs, mass, len(q), target),
            node_count=node_count,
            element_count=count,
            dofs_per_node=12,
            nodes_per_element=4,
            device=target,
            element_data=TetrahedralData(
                array(derivatives, wp.vec3d),
                array(
                    np.einsum("epai,epbi->epab", derivatives, derivatives)
                    if cache_gradient_products
                    else np.empty((0, len(points), 16, 16))
                ),
                array(weights),
                array(modulus / (2.0 * (1.0 + poisson)), grad=requires_grad),
                array(modulus / (3.0 * (1.0 - 2.0 * poisson)), grad=requires_grad),
                array(np.empty(0, dtype=np.int32), wp.int32),
            ),
        )

    @staticmethod
    def shape_functions(
        barycentric: np.ndarray, reference_positions: np.ndarray, *, derivatives: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate the paper's cubic FN interpolation in Cartesian coordinates.

        Args:
            barycentric: Finite volume coordinates with final dimension four and
                sum one. Leading dimensions broadcast against ``reference_positions``.
                Extrapolation outside the reference tetrahedron is permitted.
            reference_positions: Positively oriented tetrahedron corners [m], shape
                (..., 4, 3); det([X1-X0, X2-X0, X3-X0]) must be positive.
            derivatives: Compute Cartesian shape gradients. When false, return
                ``(shape, None)`` without forming derivative tensors or inverting
                reference Jacobians. Position interpolation is unchanged.

        Returns:
            Sixteen scalar shape functions [dimensionless, m] and their (16, 3)
            derivatives with respect to reference Cartesian position [1/m, dimensionless].
            Broadcast leading dimensions precede these output dimensions. Each
            node contributes ``[position, Dx, Dy, Dz]`` in this order.
            Multiplication by an array of sixteen nodal vectors gives position and
            the deformation gradient, respectively. The derivative array is ``None``
            when ``derivatives=False``.
        """
        if not isinstance(derivatives, (bool, np.bool_)):
            raise ValueError("derivatives must be a boolean")
        coordinates = np.asarray(barycentric, dtype=np.float64)
        corners = np.asarray(reference_positions, dtype=np.float64)
        if not derivatives or coordinates.ndim > 1 or corners.ndim > 2:
            if coordinates.shape[-1:] != (4,) or corners.shape[-2:] != (4, 3):
                raise ValueError("barycentric and reference_positions must end in (4,) and (4, 3)")
            if not np.isfinite(coordinates).all() or not np.isfinite(corners).all():
                raise ValueError("barycentric and reference_positions must be finite")
            if np.any(np.abs(coordinates.sum(axis=-1) - 1.0) > 1.0e-10):
                raise ValueError("barycentric coordinates must sum to one")
            leading = np.broadcast_shapes(coordinates.shape[:-1], corners.shape[:-2])
            # Invert each supplied parent once, before broadcasting over its many
            # quadrature or surface samples.
            jacobian = np.swapaxes(corners[..., 1:, :] - corners[..., :1, :], -1, -2)
            determinant = np.linalg.det(jacobian)
            span = np.linalg.norm(np.ptp(corners, axis=-2), axis=-1)
            if np.any(determinant <= 1.0e-14 * span**3):
                raise ValueError("reference tetrahedron has a singular or inverted Jacobian")
            if derivatives:
                inverse = np.linalg.inv(jacobian)
                gradients = np.concatenate((-inverse.sum(axis=-2, keepdims=True), inverse), axis=-2)
            coordinates = np.broadcast_to(coordinates, (*leading, 4))
            corners = np.broadcast_to(corners, (*leading, 4, 3))
            shape, derivative = _shape_functions_batch(coordinates, corners, derivatives=derivatives)
            return shape, derivative @ gradients if derivatives else None
        if coordinates.shape != (4,) or not np.isfinite(coordinates).all() or abs(coordinates.sum() - 1.0) > 1.0e-10:
            raise ValueError("barycentric must contain four finite coordinates summing to one")
        barycentric_derivatives, _ = _reference_coordinates(corners)
        shape, derivative = _shape_functions(coordinates, corners)
        return shape, derivative @ barycentric_derivatives

    @staticmethod
    def _evaluate(
        model: ANCFModel, q: wp.array[wp.float64], *, tangent: bool = True
    ) -> tuple[wp.array[wp.float64], wp.array2d[wp.float64], wp.array3d[wp.float64] | None]:
        """Evaluate FN tetra energy [J], local gradients, and exact conservative tangents.

        Coordinate and shear/bulk modulus reverse derivatives pass through cached
        quadrature kinematics. The Hessian differentiates the same sampled energy;
        integration accuracy depends on the model's quadrature rule.
        """
        data = model.element_data
        if not isinstance(data, TetrahedralData):
            raise ValueError("SolverANCFTetrahedron evaluation requires a tetra_fn model")
        if q.dtype != wp.float64 or q.shape != (model.dof_count,) or q.device != model.device:
            raise ValueError("q must be a float64 vector with model.dof_count entries on model.device")
        requires_grad = q.requires_grad or any(parameter.requires_grad for parameter in data.parameters.values())
        workspace = EvaluationWorkspace(model.device, requires_grad=requires_grad)
        energy = workspace.array("energy", model.element_count, zero=False)
        gradient = workspace.array("gradient", (model.element_count, 48), zero=False)
        deformation = workspace.array("deformation", data.weights.shape, wp.mat33d)
        stress = workspace.array("stress", data.weights.shape, _Vec6d)
        kinematics_inputs = [q, model.element_dofs, data.derivatives, data.shear_modulus, data.bulk_modulus]
        energy_inputs = [deformation, stress, data.weights]
        gradient_inputs = [deformation, stress, data.derivatives, data.weights]
        wp.launch(
            _kinematics,
            data.weights.shape,
            inputs=[*kinematics_inputs, data.active_elements],
            outputs=[deformation, stress],
            device=model.device,
        )
        wp.launch(
            _energy,
            model.element_count,
            inputs=[*energy_inputs, data.active_elements],
            outputs=[energy],
            device=model.device,
        )
        wp.launch(
            _gradient,
            gradient.shape,
            inputs=[*gradient_inputs, data.active_elements],
            outputs=[gradient],
            device=model.device,
        )
        hessian = None
        if tangent:
            hessian = workspace.array("hessian", (model.element_count, 48, 48), requires_grad=False)
            projected = workspace.array("projected", (*data.weights.shape, 16), wp.vec3d, requires_grad=False)
            metric = workspace.array("metric", data.weights.shape, wp.mat33d, requires_grad=False)
            stress_gradients = workspace.array(
                "stress_gradients", (*data.weights.shape, 16), wp.vec3d, requires_grad=False
            )
            geometry_inputs = [deformation, stress, data.derivatives]
            hessian_inputs = [
                projected,
                metric,
                stress_gradients,
                data.derivatives,
                data.gradient_products,
                data.weights,
                data.shear_modulus,
                data.bulk_modulus,
            ]
            wp.launch(
                _tangent_geometry,
                projected.shape,
                inputs=[*geometry_inputs, data.active_elements],
                outputs=[projected, metric, stress_gradients],
                device=model.device,
            )
            wp.launch(
                _hessian,
                (model.element_count, 136),
                inputs=[*hessian_inputs, data.active_elements, True],
                outputs=[hessian],
                device=model.device,
            )
        return energy, gradient, hessian
