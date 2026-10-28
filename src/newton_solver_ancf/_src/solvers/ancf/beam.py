# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shear-deformable 3243 and 3333 ANCF continuum beam elements.

Isotropic Saint-Venant--Kirchhoff elasticity uses exact analytical energy gradients and
Hessians. The Poisson contribution is integrated on the centerline to mitigate
volumetric locking; the remaining constitutive terms use 3 x 2 x 2 quadrature.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..implicit import EvaluationWorkspace
from ..implicit.sparse import assemble_matrix
from .model import ANCFModel, _element_values
from .solver import SolverANCF

_Vec6d = wp.types.vector(length=6, dtype=wp.float64)
_Mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)


@wp.func
def _deviatoric_stress(strain: _Vec6d, shear_modulus: wp.float64):
    mean = (strain[0] + strain[1] + strain[2]) / wp.float64(3.0)
    return shear_modulus * _Vec6d(
        wp.float64(2.0) * (strain[0] - mean),
        wp.float64(2.0) * (strain[1] - mean),
        wp.float64(2.0) * (strain[2] - mean),
        strain[3],
        strain[4],
        strain[5],
    )


@dataclass
class BeamData:
    """Fixed reference quadrature and optimizable constitutive tensors [Pa]."""

    derivatives: wp.array3d[wp.vec3d]
    weights: wp.array2d[wp.float64]
    constitutive: wp.array2d[_Mat66d]

    @property
    def parameters(self) -> dict[str, wp.array]:
        return {"constitutive": self.constitutive}


def _default_directors(positions: np.ndarray, formulation: str) -> np.ndarray:
    tangent = np.gradient(positions, axis=0)
    lengths = np.linalg.norm(tangent, axis=1)
    if np.any(lengths < 1.0e-12):
        raise ValueError("provide directors for a reversing or unordered centerline")
    tangent /= lengths[:, None]
    frames = []
    normal = np.eye(3)[np.argmin(np.abs(tangent[0]))]
    for direction in tangent:
        normal = normal - np.dot(direction, normal) * direction
        if np.linalg.norm(normal) < 1.0e-10:
            normal = np.cross(np.eye(3)[np.argmin(np.abs(direction))], direction)
        normal /= np.linalg.norm(normal)
        frame = np.array([direction, normal.copy(), np.cross(direction, normal)])
        frames.append(frame if formulation == "3243" else frame[1:])
    return np.asarray(frames)


@wp.func
def _deformation_gradient(
    q: wp.array[wp.float64],
    dofs: wp.array2d[int],
    derivatives: wp.array3d[wp.vec3d],
    element: int,
    point: int,
    vectors_per_node: int,
):
    deformation = wp.mat33d(0.0)
    origin = wp.vec3d(q[dofs[element, 0]], q[dofs[element, 1]], q[dofs[element, 2]])
    for vector in range(derivatives.shape[2]):
        position = wp.vec3d(
            q[dofs[element, 3 * vector]], q[dofs[element, 3 * vector + 1]], q[dofs[element, 3 * vector + 2]]
        )
        if vector % vectors_per_node == 0:
            position = position - origin
        deformation = deformation + wp.outer(position, derivatives[element, point, vector])
    return deformation


@wp.func
def _green_strain(deformation: wp.mat33d):
    metric = wp.transpose(deformation) * deformation
    return _Vec6d(
        wp.float64(0.5) * (metric[0, 0] - wp.float64(1.0)),
        wp.float64(0.5) * (metric[1, 1] - wp.float64(1.0)),
        wp.float64(0.5) * (metric[2, 2] - wp.float64(1.0)),
        metric[1, 2],
        metric[0, 2],
        metric[0, 1],
    )


@wp.func
def _strain_derivative(deformation: wp.mat33d, derivative: wp.vec3d, component: int):
    return _Vec6d(
        deformation[component, 0] * derivative[0],
        deformation[component, 1] * derivative[1],
        deformation[component, 2] * derivative[2],
        deformation[component, 1] * derivative[2] + deformation[component, 2] * derivative[1],
        deformation[component, 0] * derivative[2] + deformation[component, 2] * derivative[0],
        deformation[component, 0] * derivative[1] + deformation[component, 1] * derivative[0],
    )


@wp.kernel
def _continuum_kinematics(
    q: wp.array[wp.float64],
    dofs: wp.array2d[int],
    derivatives: wp.array3d[wp.vec3d],
    constitutive: wp.array2d[_Mat66d],
    vectors_per_node: int,
    deformation: wp.array2d[wp.mat33d],
    stress: wp.array2d[_Vec6d],
):
    element, point = wp.tid()
    value = _deformation_gradient(q, dofs, derivatives, element, point, vectors_per_node)
    deformation[element, point] = value
    stress[element, point] = constitutive[element, point] * _green_strain(value)


@wp.kernel
def _continuum_energy(
    deformation: wp.array2d[wp.mat33d],
    stress: wp.array2d[_Vec6d],
    weights: wp.array2d[wp.float64],
    energy: wp.array[wp.float64],
):
    element = wp.tid()
    value = wp.float64(0.0)
    for point in range(weights.shape[1]):
        value += (
            wp.float64(0.5)
            * weights[element, point]
            * wp.dot(_green_strain(deformation[element, point]), stress[element, point])
        )
    energy[element] = value


@wp.kernel
def _continuum_gradient(
    deformation: wp.array2d[wp.mat33d],
    stress: wp.array2d[_Vec6d],
    derivatives: wp.array3d[wp.vec3d],
    weights: wp.array2d[wp.float64],
    gradient: wp.array2d[wp.float64],
):
    element, i = wp.tid()
    value = wp.float64(0.0)
    for point in range(weights.shape[1]):
        strain_i = _strain_derivative(deformation[element, point], derivatives[element, point, i // 3], i % 3)
        value += weights[element, point] * wp.dot(strain_i, stress[element, point])
    gradient[element, i] = value


@wp.kernel(enable_backward=False)
def _continuum_tangent_basis(
    deformation: wp.array2d[wp.mat33d],
    derivatives: wp.array3d[wp.vec3d],
    constitutive: wp.array2d[_Mat66d],
    basis: wp.array3d[_Vec6d],
    weighted_basis: wp.array3d[_Vec6d],
):
    element, point, i = wp.tid()
    value = _strain_derivative(deformation[element, point], derivatives[element, point, i // 3], i % 3)
    basis[element, point, i] = value
    weighted_basis[element, point, i] = constitutive[element, point] * value


@wp.kernel(enable_backward=False)
def _continuum_hessian(
    basis: wp.array3d[_Vec6d],
    weighted_basis: wp.array3d[_Vec6d],
    stress: wp.array2d[_Vec6d],
    derivatives: wp.array3d[wp.vec3d],
    weights: wp.array2d[wp.float64],
    hessian: wp.array3d[wp.float64],
):
    element, i, j = wp.tid()
    value = wp.float64(0.0)
    for point in range(weights.shape[1]):
        entry = wp.dot(basis[element, point, i], weighted_basis[element, point, j])
        if i % 3 == j % 3:
            derivative_i, derivative_j = derivatives[element, point, i // 3], derivatives[element, point, j // 3]
            sigma = stress[element, point]
            stress_tensor = wp.mat33d(
                sigma[0], sigma[5], sigma[4], sigma[5], sigma[1], sigma[3], sigma[4], sigma[3], sigma[2]
            )
            entry = entry + wp.dot(derivative_i, stress_tensor * derivative_j)
        value = value + weights[element, point] * entry
    hessian[element, i, j] = value


class SolverANCFBeam(SolverANCF):
    """Implicit ANCF3243 or ANCF3333 rectangular beam."""

    element_type = BeamData

    @staticmethod
    def create_model(
        positions: np.ndarray,
        *,
        formulation: str = "3243",
        elements: np.ndarray | None = None,
        directors: np.ndarray | None = None,
        rest_lengths: float | np.ndarray | None = None,
        young_modulus: float | np.ndarray = 1.0e7,
        poisson_ratio: float | np.ndarray = 0.3,
        density: float | np.ndarray = 1000.0,
        width: float | np.ndarray = 0.01,
        height: float | np.ndarray = 0.01,
        device: str | wp.Device | None = None,
        requires_grad: bool = False,
    ) -> ANCFModel:
        """Build rectangular, shear-deformable ANCF beam elements.

        The 3243 element has two nodes with ``[position, Dx, Dy, Dz]`` coordinates
        (24 element DOFs). The 3333 element has three nodes with
        ``[position, Dy, Dz]`` coordinates (27 element DOFs). Both model axial,
        bending, shear, cross-section deformation, and twist responses.

        Args:
            positions: Reference node positions [m], shape (node_count, 3).
            formulation: ``"3243"`` or ``"3333"``.
            elements: Node indices per element. A 3333 row is ``[left, right, middle]``.
                Omitted topology follows a sequential centerline; 3333 then requires
                an odd number of nodes and shares every second node.
            directors: Dimensionless reference position gradients, shape
                (node_count, 3, 3) for 3243 or (node_count, 2, 3) for 3333.
                Omitted gradients use transported orthonormal frames.
            rest_lengths: Reference interpolation lengths [m], default endpoint distances.
            young_modulus: Young's modulus [Pa], scalar or per element.
            poisson_ratio: Poisson ratio, scalar or per element; must lie in (-1, 0.5).
            density: Reference mass density [kg/m³], scalar or per element.
            width: Cross-section width [m], scalar or per element.
            height: Cross-section height [m], scalar or per element.
            device: Warp device.
            requires_grad: Enable constitutive-tensor gradients. Reference geometry
                and the consistent mass matrix remain fixed.

        Returns:
            A model for :class:`SolverANCF` with consistent mass and reference
            quadrature. Material gradients are stored on ``element_data.constitutive``.
        """
        if formulation not in ("3243", "3333"):
            raise ValueError("formulation must be '3243' or '3333'")
        positions = np.asarray(positions, dtype=np.float64)
        nodes_per_element, gradients_per_node = (2, 3) if formulation == "3243" else (3, 2)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or len(positions) < nodes_per_element
            or not np.isfinite(positions).all()
        ):
            raise ValueError("positions must be a finite (node_count, 3) array with enough nodes")
        node_count = len(positions)
        if elements is None:
            if formulation == "3243":
                topology = np.column_stack((np.arange(node_count - 1), np.arange(1, node_count))).astype(np.int32)
            else:
                if node_count % 2 != 1:
                    raise ValueError("a sequential 3333 centerline requires an odd number of nodes")
                left = np.arange(0, node_count - 2, 2)
                topology = np.column_stack((left, left + 2, left + 1)).astype(np.int32)
        else:
            topology = np.asarray(elements)
            if (
                topology.ndim != 2
                or topology.shape[1] != nodes_per_element
                or len(topology) == 0
                or not np.issubdtype(topology.dtype, np.integer)
            ):
                raise ValueError("elements must be an integer array with the formulation's node count per row")
            if np.any(topology < 0) or np.any(topology >= node_count):
                raise ValueError("element node indices are out of range")
            topology = topology.astype(np.int32)
            if directors is None:
                raise ValueError("explicit topology requires explicit directors")
        if len(np.unique(topology)) != node_count:
            raise ValueError("element topology must reference every node")
        if any(len(np.unique(row)) != nodes_per_element for row in topology):
            raise ValueError("an element cannot repeat a node")
        if directors is None:
            directors = _default_directors(positions, formulation)
        directors = np.asarray(directors, dtype=np.float64)
        if directors.shape != (node_count, gradients_per_node, 3) or not np.isfinite(directors).all():
            raise ValueError("directors have an invalid shape or contain non-finite values")
        count = len(topology)
        chord_lengths = np.linalg.norm(positions[topology[:, 1]] - positions[topology[:, 0]], axis=1)
        lengths = _element_values(chord_lengths if rest_lengths is None else rest_lengths, count, "rest_lengths")
        widths, heights = _element_values(width, count, "width"), _element_values(height, count, "height")
        moduli, densities = (
            _element_values(young_modulus, count, "young_modulus"),
            _element_values(density, count, "density"),
        )
        poisson = np.broadcast_to(np.asarray(poisson_ratio, dtype=np.float64), (count,)).copy()
        if not np.isfinite(poisson).all() or np.any(poisson <= -1.0) or np.any(poisson >= 0.5):
            raise ValueError("poisson_ratio must be finite and lie in (-1, 0.5)")
        dofs_per_node = 3 * (1 + gradients_per_node)
        q = np.concatenate((positions[:, None, :], directors), axis=1).reshape(-1)
        dofs = (dofs_per_node * topology[:, :, None] + np.arange(dofs_per_node)).reshape(count, -1).astype(np.int32)
        vector_count = dofs.shape[1] // 3
        derivatives = np.zeros((count, 15, vector_count, 3))
        weights = np.zeros((count, 15))
        constitutive = np.zeros((count, 15, 6, 6))
        mass = np.empty((count, dofs.shape[1], dofs.shape[1]), dtype=np.float64)
        axial_points, axial_weights = np.polynomial.legendre.leggauss(3)
        cross_points, cross_weights = np.polynomial.legendre.leggauss(2)
        mass_points, mass_weights = np.polynomial.legendre.leggauss(4)
        for element in range(count):
            local = q[dofs[element]].reshape(vector_count, 3)
            modulus, nu = moduli[element], poisson[element]
            mu, lam = modulus / (2.0 * (1.0 + nu)), modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            d0 = np.diag([modulus, modulus, modulus, mu, mu, mu])
            dv = np.zeros((6, 6))
            dv[:3, :3] = lam
            dv[np.arange(3), np.arange(3)] += 2.0 * mu - modulus
            quadrature = [
                (x, y, z, wx * wy * wz, d0)
                for x, wx in zip(axial_points, axial_weights, strict=True)
                for y, wy in zip(cross_points, cross_weights, strict=True)
                for z, wz in zip(cross_points, cross_weights, strict=True)
            ]
            quadrature.extend((x, 0.0, 0.0, 4.0 * wx, dv) for x, wx in zip(axial_points, axial_weights, strict=True))
            for index, (x, y, z, weight, elasticity) in enumerate(quadrature):
                _, derivative = SolverANCFBeam.shape_functions(
                    formulation, x, y, z, lengths[element], widths[element], heights[element]
                )
                jacobian = local.T @ derivative
                determinant = np.linalg.det(jacobian)
                if determinant <= 1.0e-14 * lengths[element] * widths[element] * heights[element]:
                    raise ValueError("reference beam has a singular or inverted Jacobian")
                derivatives[element, index] = derivative @ np.linalg.inv(jacobian)
                weights[element, index] = weight * determinant
                constitutive[element, index] = elasticity
            local_mass = np.zeros((vector_count, vector_count))
            for x, wx in zip(mass_points, mass_weights, strict=True):
                for y, wy in zip(cross_points, cross_weights, strict=True):
                    for z, wz in zip(cross_points, cross_weights, strict=True):
                        shape, derivative = SolverANCFBeam.shape_functions(
                            formulation, x, y, z, lengths[element], widths[element], heights[element]
                        )
                        determinant = np.linalg.det(local.T @ derivative)
                        if determinant <= 0.0:
                            raise ValueError("reference beam has an inverted Jacobian at a mass quadrature point")
                        local_mass += np.outer(shape, shape) * (densities[element] * wx * wy * wz * determinant)
            mass[element] = np.kron(local_mass, np.eye(3))
        target = wp.get_device(device)

        def array(value, dtype=wp.float64, grad=False):
            return wp.array(value, dtype=dtype, device=target, requires_grad=grad)

        return ANCFModel(
            formulation=formulation,
            rest_q=array(q),
            elements=array(topology, wp.int32),
            element_dofs=array(dofs, wp.int32),
            lengths=array(lengths),
            mass_matrix=assemble_matrix(dofs, mass, len(q), target),
            node_count=node_count,
            element_count=count,
            dofs_per_node=dofs_per_node,
            nodes_per_element=nodes_per_element,
            device=target,
            element_data=BeamData(
                array(derivatives, wp.vec3d), array(weights), array(constitutive, _Mat66d, requires_grad)
            ),
        )

    @staticmethod
    def shape_functions(
        formulation: str, xi: float, eta: float, zeta: float, length: float, width: float, height: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate beam interpolation in normalized coordinates [-1, 1].

        Args:
            formulation: ``"3243"`` or ``"3333"``.
            xi: Axial coordinate.
            eta: First transverse coordinate.
            zeta: Second transverse coordinate.
            length: Reference axial interpolation length [m].
            width: Reference cross-section width [m].
            height: Reference cross-section height [m].

        Returns:
            Shape values [dimensionless, m] and their normalized-coordinate
            derivatives, with one scalar shape per nodal position/gradient vector.
        """
        if not np.isfinite([xi, eta, zeta, length, width, height]).all() or min(length, width, height) <= 0.0:
            raise ValueError("coordinates must be finite and dimensions must be positive")
        if formulation == "3243":
            x2, x3 = xi * xi, xi * xi * xi
            shape = np.array(
                [
                    0.25 * (x3 - 3.0 * xi + 2.0),
                    0.125 * length * (x3 - x2 - xi + 1.0),
                    0.25 * width * eta * (1.0 - xi),
                    0.25 * height * zeta * (1.0 - xi),
                    0.25 * (-x3 + 3.0 * xi + 2.0),
                    0.125 * length * (x3 + x2 - xi - 1.0),
                    0.25 * width * eta * (1.0 + xi),
                    0.25 * height * zeta * (1.0 + xi),
                ]
            )
            derivative = np.zeros((8, 3))
            derivative[:, 0] = [
                0.75 * (x2 - 1.0),
                0.125 * length * (3.0 * x2 - 2.0 * xi - 1.0),
                -0.25 * width * eta,
                -0.25 * height * zeta,
                0.75 * (1.0 - x2),
                0.125 * length * (3.0 * x2 + 2.0 * xi - 1.0),
                0.25 * width * eta,
                0.25 * height * zeta,
            ]
            derivative[[2, 6], 1] = 0.25 * width * np.array([1.0 - xi, 1.0 + xi])
            derivative[[3, 7], 2] = 0.25 * height * np.array([1.0 - xi, 1.0 + xi])
            return shape, derivative
        if formulation == "3333":
            lagrange = np.array([0.5 * xi * (xi - 1.0), 0.5 * xi * (xi + 1.0), 1.0 - xi * xi])
            lagrange_derivative = np.array([xi - 0.5, xi + 0.5, -2.0 * xi])
            shape = np.column_stack((lagrange, 0.5 * width * eta * lagrange, 0.5 * height * zeta * lagrange)).reshape(
                -1
            )
            derivative = np.zeros((9, 3))
            derivative[:, 0] = np.column_stack(
                (
                    lagrange_derivative,
                    0.5 * width * eta * lagrange_derivative,
                    0.5 * height * zeta * lagrange_derivative,
                )
            ).reshape(-1)
            derivative[1::3, 1] = 0.5 * width * lagrange
            derivative[2::3, 2] = 0.5 * height * lagrange
            return shape, derivative
        raise ValueError("formulation must be '3243' or '3333'")

    @staticmethod
    def _evaluate(
        model: ANCFModel, q: wp.array[wp.float64], *, tangent: bool = True
    ) -> tuple[wp.array[wp.float64], wp.array2d[wp.float64], wp.array3d[wp.float64] | None]:
        """Evaluate continuum beam energy [J], local gradients, and exact Hessians.

        Args:
            model: A 3243 or 3333 model made by :meth:`SolverANCFBeam.create_model`.
            q: Flattened generalized positions [m, dimensionless].
            tangent: Compute the energy Hessian [N/m, N, N m].

        Returns:
            Element energies, local positive energy gradients [N, N m], and
            optional exact local Hessians. Energy and gradient launches support Warp
            reverse differentiation with respect to coordinates and constitutive tensors.
        """
        data = model.element_data
        if not isinstance(data, BeamData):
            raise ValueError("SolverANCFBeam evaluation requires a 3243 or 3333 beam model")
        if q.dtype != wp.float64 or q.ndim != 1 or len(q) != model.dof_count:
            raise ValueError("q must be a float64 vector with model.dof_count entries")
        requires_grad = q.requires_grad or data.constitutive.requires_grad
        workspace = EvaluationWorkspace(model.device, requires_grad=requires_grad)
        energy = workspace.array("energy", model.element_count)
        gradient = workspace.array("gradient", (model.element_count, model.element_dof_count))
        deformation = workspace.array("deformation", data.weights.shape, wp.mat33d)
        stress = workspace.array("stress", data.weights.shape, _Vec6d)
        wp.launch(
            _continuum_kinematics,
            data.weights.shape,
            inputs=[q, model.element_dofs, data.derivatives, data.constitutive, model.dofs_per_node // 3],
            outputs=[deformation, stress],
            device=model.device,
        )
        wp.launch(
            _continuum_energy,
            model.element_count,
            inputs=[deformation, stress, data.weights],
            outputs=[energy],
            device=model.device,
        )
        wp.launch(
            _continuum_gradient,
            gradient.shape,
            inputs=[deformation, stress, data.derivatives, data.weights],
            outputs=[gradient],
            device=model.device,
        )
        hessian = None
        if tangent:
            hessian = workspace.array(
                "hessian",
                (model.element_count, model.element_dof_count, model.element_dof_count),
                requires_grad=False,
            )
            basis = workspace.array(
                "basis", (*data.weights.shape, model.element_dof_count), _Vec6d, requires_grad=False
            )
            weighted_basis = workspace.array("weighted_basis", basis.shape, _Vec6d, requires_grad=False)
            wp.launch(
                _continuum_tangent_basis,
                basis.shape,
                inputs=[deformation, data.derivatives, data.constitutive],
                outputs=[basis, weighted_basis],
                device=model.device,
            )
            wp.launch(
                _continuum_hessian,
                hessian.shape,
                inputs=[basis, weighted_basis, stress, data.derivatives, data.weights],
                outputs=[hessian],
                device=model.device,
            )
        return energy, gradient, hessian
