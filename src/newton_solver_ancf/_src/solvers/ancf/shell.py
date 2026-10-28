# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservative four-node ANCF3423 shell with assumed natural strains.

The element uses a bilinear position/director field,
mid-edge transverse-shear ANS ties, corner thickness-strain ANS ties, and material
fiber transformation. The fixed reference geometry permits representing every
assumed strain as a quadratic form. Its analytic force and exact tangent follow
the same energy, including geometric stiffness, and support Warp differentiation.

EAS enhancement, pressure loading, contact, and material damping are outside
this element implementation.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..implicit import EvaluationWorkspace
from ..implicit.sparse import assemble_matrix
from .model import ANCFModel, _element_values
from .solver import SolverANCF

# These nested quadratic-form loops are small at runtime but expensive to unroll
# through reverse mode. Keep compilation size independent of their product.
wp.set_module_options({"max_unroll": 1})


_Vec6d = wp.types.vector(length=6, dtype=wp.float64)
_Mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
_Mat88d = wp.types.matrix(shape=(8, 8), dtype=wp.float64)


@dataclass
class ShellData:
    """Fixed shell quadrature and differentiable dimensionless stiffness scales."""

    strain_matrices: wp.array3d[_Mat88d]
    reference_strains: wp.array2d[_Vec6d]
    weights: wp.array2d[wp.float64]
    constitutive: wp.array[_Mat66d]
    stiffness_scale: wp.array[wp.float64]

    @property
    def parameters(self) -> dict[str, wp.array]:
        return {"stiffness_scale": self.stiffness_scale}


def _shell_interpolation(xi: float, eta: float, zeta: float, thickness: float):
    signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    shape = 0.25 * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 1] * eta)
    first_xi = 0.25 * signs[:, 0] * (1.0 + signs[:, 1] * eta)
    first_eta = 0.25 * signs[:, 1] * (1.0 + signs[:, 0] * xi)
    values = np.empty(8)
    values[::2] = shape
    values[1::2] = 0.5 * thickness * zeta * shape
    derivatives = np.zeros((8, 3))
    derivatives[::2, 0] = first_xi
    derivatives[::2, 1] = first_eta
    derivatives[1::2, 0] = 0.5 * thickness * zeta * first_xi
    derivatives[1::2, 1] = 0.5 * thickness * zeta * first_eta
    derivatives[1::2, 2] = 0.5 * thickness * shape
    return values, derivatives, shape


def _natural_strain_matrices(derivatives: np.ndarray) -> np.ndarray:
    result = np.empty((6, 8, 8))
    for axis in range(3):
        result[axis] = np.outer(derivatives[:, axis], derivatives[:, axis])
    for index, (a, b) in enumerate(((1, 2), (0, 2), (0, 1)), 3):
        result[index] = np.outer(derivatives[:, a], derivatives[:, b]) + np.outer(derivatives[:, b], derivatives[:, a])
    return result


def _voigt_transform(beta: np.ndarray) -> np.ndarray:
    result = np.empty((6, 6))
    pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    for column, (a, b) in enumerate(pairs):
        basis = np.zeros((3, 3))
        basis[a, b] = basis[b, a] = 1.0 if a == b else 0.5
        transformed = beta @ basis @ beta.T
        result[:, column] = [
            transformed[0, 0],
            transformed[1, 1],
            transformed[2, 2],
            2.0 * transformed[1, 2],
            2.0 * transformed[0, 2],
            2.0 * transformed[0, 1],
        ]
    return result


@wp.func
def _coordinate(q: wp.array[wp.float64], dofs: wp.array2d[wp.int32], element: int, block: int) -> wp.vec3d:
    offset = 3 * block
    value = wp.vec3d(q[dofs[element, offset]], q[dofs[element, offset + 1]], q[dofs[element, offset + 2]])
    if block % 2 == 0:
        value -= wp.vec3d(q[dofs[element, 0]], q[dofs[element, 1]], q[dofs[element, 2]])
    return value


@wp.kernel
def _compute_strains(
    q: wp.array[wp.float64],
    dofs: wp.array2d[wp.int32],
    matrices: wp.array3d[_Mat88d],
    reference: wp.array2d[_Vec6d],
    material: wp.array[_Mat66d],
    strains: wp.array2d[_Vec6d],
    stresses: wp.array2d[_Vec6d],
):
    element, point = wp.tid()
    strain = _Vec6d(wp.float64(0.0))
    for component in range(6):
        matrix = matrices[element, point, component]
        value = wp.float64(0.0)
        for a in range(8):
            qa = _coordinate(q, dofs, element, a)
            for b in range(8):
                value += wp.float64(0.5) * matrix[a, b] * wp.dot(qa, _coordinate(q, dofs, element, b))
        strain[component] = value - reference[element, point][component]
    strains[element, point] = strain
    stresses[element, point] = material[element] * strain


@wp.func
def _strain_gradient(
    q: wp.array[wp.float64],
    dofs: wp.array2d[wp.int32],
    matrices: wp.array3d[_Mat88d],
    element: int,
    point: int,
    local_dof: int,
) -> _Vec6d:
    gradient = _Vec6d(wp.float64(0.0))
    block, axis = local_dof // 3, local_dof % 3
    for component in range(6):
        matrix = matrices[element, point, component]
        value = wp.float64(0.0)
        for b in range(8):
            value += matrix[block, b] * _coordinate(q, dofs, element, b)[axis]
        gradient[component] = value
    return gradient


@wp.kernel
def _compute_strain_gradients(
    q: wp.array[wp.float64],
    dofs: wp.array2d[wp.int32],
    matrices: wp.array3d[_Mat88d],
    strain_gradients: wp.array3d[_Vec6d],
):
    element, point, local_dof = wp.tid()
    strain_gradients[element, point, local_dof] = _strain_gradient(q, dofs, matrices, element, point, local_dof)


@wp.kernel(enable_backward=False)
def _compute_stress_gradients(
    strain_gradients: wp.array3d[_Vec6d],
    material: wp.array[_Mat66d],
    stress_gradients: wp.array3d[_Vec6d],
):
    element, point, local_dof = wp.tid()
    stress_gradients[element, point, local_dof] = material[element] * strain_gradients[element, point, local_dof]


@wp.kernel
def _compute_energy(
    strains: wp.array2d[_Vec6d],
    stresses: wp.array2d[_Vec6d],
    weights: wp.array2d[wp.float64],
    scale: wp.array[wp.float64],
    energy: wp.array[wp.float64],
):
    element = wp.tid()
    result = wp.float64(0.0)
    for point in range(weights.shape[1]):
        result += (
            wp.float64(0.5)
            * weights[element, point]
            * scale[element]
            * wp.dot(strains[element, point], stresses[element, point])
        )
    energy[element] = result


@wp.kernel
def _compute_gradient(
    strain_gradients: wp.array3d[_Vec6d],
    stresses: wp.array2d[_Vec6d],
    weights: wp.array2d[wp.float64],
    scale: wp.array[wp.float64],
    gradient: wp.array2d[wp.float64],
):
    element, local_dof = wp.tid()
    result = wp.float64(0.0)
    for point in range(weights.shape[1]):
        b = strain_gradients[element, point, local_dof]
        stress = stresses[element, point]
        result += weights[element, point] * scale[element] * wp.dot(b, stress)
    gradient[element, local_dof] = result


@wp.kernel(enable_backward=False)
def _compute_hessian(
    strain_gradients: wp.array3d[_Vec6d],
    stress_gradients: wp.array3d[_Vec6d],
    matrices: wp.array3d[wp.float64],
    stresses: wp.array2d[_Vec6d],
    weights: wp.array2d[wp.float64],
    scale: wp.array[wp.float64],
    hessian: wp.array3d[wp.float64],
):
    element, i, j = wp.tid()
    result = wp.float64(0.0)
    for point in range(weights.shape[1]):
        bi = strain_gradients[element, point, i]
        stress_j = stress_gradients[element, point, j]
        value = wp.dot(bi, stress_j)
        if i % 3 == j % 3:
            stress = stresses[element, point]
            for component in range(6):
                index = 64 * component + 8 * (i // 3) + j // 3
                value += stress[component] * matrices[element, point, index]
        result += weights[element, point] * scale[element] * value
    hessian[element, i, j] = result


class SolverANCFShell(SolverANCF):
    """Converged implicit ANCF3423 quadrilateral shell with ANS."""

    element_type = ShellData

    @staticmethod
    def create_model(
        positions: np.ndarray,
        elements: np.ndarray,
        *,
        directors: np.ndarray | None = None,
        thickness: float | np.ndarray = 0.01,
        young_modulus: float | np.ndarray = 1.0e7,
        poisson_ratio: float | np.ndarray = 0.3,
        density: float | np.ndarray = 1000.0,
        fiber_angle: float | np.ndarray = 0.0,
        constitutive: np.ndarray | None = None,
        device: wp.DeviceLike = None,
        requires_grad: bool = False,
    ) -> ANCFModel:
        """Build four-node ANCF3423 shells with conservative assumed natural strains.

        Args:
            positions: Reference positions [m], shape (node_count, 3).
            elements: Quad node indices in counterclockwise surface order.
            directors: Reference thickness gradients [dimensionless]; omitted values
                are normalized averages of adjacent face normals.
            thickness: Reference thickness [m], scalar or one value per element.
            young_modulus: Isotropic Young's modulus [Pa], scalar or per element.
            poisson_ratio: Isotropic Poisson ratio, scalar or per element.
            density: Reference density [kg/m³], scalar or per element.
            fiber_angle: Material direction angle [rad] relative to each element's
                first surface coordinate, scalar or per element.
            constitutive: Optional symmetric positive-definite material matrices [Pa],
                shape (element_count, 6, 6) or (6, 6). Engineering Voigt order is
                ``[E11, E22, E33, 2E23, 2E13, 2E12]``. Overrides isotropic constants.
            device: Warp device; defaults to the current device.
            requires_grad: Differentiate per-element dimensionless stiffness scales.
                Reference geometry and density are fixed because mass is precomputed.

        Returns:
            Shell model for :class:`SolverANCF`, with six coordinates per node.
        """
        positions = np.asarray(positions, dtype=np.float64)
        topology = np.asarray(elements)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 4 or not np.all(np.isfinite(positions)):
            raise ValueError("positions must be finite with shape (node_count >= 4, 3)")
        if topology.ndim != 2 or topology.shape[1] != 4 or not len(topology):
            raise ValueError("elements must have shape (element_count >= 1, 4)")
        if not np.issubdtype(topology.dtype, np.integer) or np.any(topology < 0) or np.any(topology >= len(positions)):
            raise ValueError("elements must contain valid integer node indices")
        if any(len(set(element)) != 4 for element in topology):
            raise ValueError("each shell element must contain four different nodes")
        if len(np.unique(topology)) != len(positions):
            raise ValueError("every node must belong to a shell element")
        topology = topology.astype(np.int32)
        count, node_count = len(topology), len(positions)
        thicknesses = _element_values(thickness, count, "thickness")
        densities = _element_values(density, count, "density")
        angles = np.broadcast_to(np.asarray(fiber_angle, dtype=np.float64), (count,))
        if not np.all(np.isfinite(angles)):
            raise ValueError("fiber_angle must be finite")
        if directors is None:
            directors = np.zeros_like(positions)
            for element in topology:
                normal = np.cross(
                    positions[element[1]] - positions[element[0]], positions[element[3]] - positions[element[0]]
                )
                directors[element] += normal
            norms = np.linalg.norm(directors, axis=1)
            if np.any(norms < 1.0e-12):
                raise ValueError("degenerate shell faces require valid geometry and directors")
            directors /= norms[:, None]
        else:
            directors = np.asarray(directors, dtype=np.float64)
            if directors.shape != positions.shape or not np.all(np.isfinite(directors)):
                raise ValueError("directors must be finite and have the same shape as positions")
        if constitutive is None:
            moduli = _element_values(young_modulus, count, "young_modulus")
            poisson = np.broadcast_to(np.asarray(poisson_ratio, dtype=np.float64), (count,))
            if not np.all(np.isfinite(poisson)) or np.any(poisson <= -1.0) or np.any(poisson >= 0.5):
                raise ValueError("poisson_ratio must lie strictly between -1 and 0.5")
            materials = np.zeros((count, 6, 6))
            for element in range(count):
                shear = moduli[element] / (2.0 * (1.0 + poisson[element]))
                lame = moduli[element] * poisson[element] / ((1.0 + poisson[element]) * (1.0 - 2.0 * poisson[element]))
                materials[element, :3, :3] = lame
                materials[element, np.arange(3), np.arange(3)] += 2.0 * shear
                materials[element, np.arange(3, 6), np.arange(3, 6)] = shear
        else:
            materials = np.broadcast_to(np.asarray(constitutive, dtype=np.float64), (count, 6, 6)).copy()
            if not np.all(np.isfinite(materials)) or not np.allclose(
                materials, materials.transpose(0, 2, 1), rtol=1.0e-12
            ):
                raise ValueError("constitutive matrices must be finite and symmetric")
            if np.any(np.linalg.eigvalsh(materials) <= 0.0):
                raise ValueError("constitutive matrices must be positive definite")

        q = np.column_stack((positions, directors)).reshape(-1)
        dofs = (6 * topology[:, :, None] + np.arange(6)[None, None, :]).reshape(count, 24).astype(np.int32)
        gp, gw = np.polynomial.legendre.leggauss(2)
        gz, wz = np.polynomial.legendre.leggauss(3)
        matrices = np.zeros((count, 12, 6, 8, 8))
        reference = np.zeros((count, 12, 6))
        weights = np.zeros((count, 12))
        mass = np.empty((count, dofs.shape[1], dofs.shape[1]), dtype=np.float64)
        for element in range(count):
            local_q = q[dofs[element]].reshape(8, 3).copy()
            local_q[::2] -= local_q[0].copy()
            h = thicknesses[element]
            corners = [
                _natural_strain_matrices(_shell_interpolation(x, y, 0.0, h)[1])[2]
                for x, y in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
            ]
            shear_13 = [_natural_strain_matrices(_shell_interpolation(0.0, y, 0.0, h)[1])[4] for y in (-1.0, 1.0)]
            shear_23 = [_natural_strain_matrices(_shell_interpolation(x, 0.0, 0.0, h)[1])[3] for x in (-1.0, 1.0)]
            local_mass = np.zeros((8, 8))
            quadrature = 0
            for i, xi in enumerate(gp):
                for j, eta in enumerate(gp):
                    for k, zeta in enumerate(gz):
                        values, derivatives, shape = _shell_interpolation(xi, eta, zeta, h)
                        jacobian = local_q.T @ derivatives
                        determinant = np.linalg.det(jacobian)
                        if not np.isfinite(determinant) or abs(determinant) < 1.0e-15:
                            raise ValueError("reference shell Jacobian is degenerate")
                        natural = _natural_strain_matrices(derivatives)
                        natural[2] = sum(shape[corner] * corners[corner] for corner in range(4))
                        natural[4] = 0.5 * ((1.0 - eta) * shear_13[0] + (1.0 + eta) * shear_13[1])
                        natural[3] = 0.5 * ((1.0 - xi) * shear_23[0] + (1.0 + xi) * shear_23[1])
                        axis1 = jacobian[:, 0] / np.linalg.norm(jacobian[:, 0])
                        normal = np.cross(jacobian[:, 0], jacobian[:, 1])
                        normal /= np.linalg.norm(normal)
                        axis2 = np.cross(normal, axis1)
                        cosine, sine = np.cos(angles[element]), np.sin(angles[element])
                        axes = np.array([cosine * axis1 + sine * axis2, -sine * axis1 + cosine * axis2, normal])
                        beta = axes @ np.linalg.inv(jacobian).T
                        material = np.einsum("rs,sab->rab", _voigt_transform(beta), natural)
                        weight = gw[i] * gw[j] * wz[k] * abs(determinant)
                        matrices[element, quadrature] = material
                        reference[element, quadrature] = 0.5 * np.einsum("rab,ac,bc->r", material, local_q, local_q)
                        weights[element, quadrature] = weight
                        local_mass += densities[element] * weight * np.outer(values, values)
                        quadrature += 1
            mass[element] = np.kron(local_mass, np.eye(3))

        target = wp.get_device(device)

        def array(values, dtype=wp.float64, grad=False):
            return wp.array(values, dtype=dtype, device=target, requires_grad=grad)

        data = ShellData(
            array(matrices, _Mat88d),
            array(reference, _Vec6d),
            array(weights),
            array(materials, _Mat66d),
            array(np.ones(count), grad=requires_grad),
        )
        return ANCFModel(
            formulation="shell3423",
            rest_q=array(q),
            elements=array(topology, wp.int32),
            element_dofs=array(dofs, wp.int32),
            lengths=array(np.linalg.norm(positions[topology[:, 1]] - positions[topology[:, 0]], axis=1)),
            mass_matrix=assemble_matrix(dofs, mass, len(q), target),
            node_count=node_count,
            element_count=count,
            dofs_per_node=6,
            nodes_per_element=4,
            device=target,
            element_data=data,
        )

    @staticmethod
    def _evaluate(
        model: ANCFModel, q: wp.array[wp.float64], *, tangent: bool = True
    ) -> tuple[wp.array[wp.float64], wp.array2d[wp.float64], wp.array3d[wp.float64] | None]:
        """Evaluate ANS shell energy [J], gradient [N, N m], and exact tangent.

        A false ``tangent`` omits the Hessian; energy and gradients always support
        Warp reverse mode with respect to coordinates and stiffness scales.
        """
        data = model.element_data
        if not isinstance(data, ShellData):
            raise ValueError("SolverANCFShell evaluation requires a shell3423 model")
        if q.dtype != wp.float64 or q.ndim != 1 or q.shape[0] != model.dof_count:
            raise ValueError("q must be a float64 vector with model.dof_count entries")
        grad = q.requires_grad or data.stiffness_scale.requires_grad
        workspace = EvaluationWorkspace(model.device, requires_grad=grad)
        strains = workspace.array("strains", data.weights.shape, _Vec6d)
        stresses = workspace.array("stresses", data.weights.shape, _Vec6d)
        # Each Jacobian row appears in 48 tangent entries. Compute it once per
        # quadrature point; recorded evaluations retain their own storage.
        strain_gradients = workspace.array("strain_gradients", (*data.weights.shape, 24), _Vec6d)
        energy = workspace.array("energy", model.element_count)
        gradient = workspace.array("gradient", (model.element_count, 24))
        wp.launch(
            _compute_strains,
            dim=data.weights.shape,
            inputs=[q, model.element_dofs, data.strain_matrices, data.reference_strains, data.constitutive],
            outputs=[strains, stresses],
            device=model.device,
        )
        wp.launch(
            _compute_strain_gradients,
            dim=strain_gradients.shape,
            inputs=[q, model.element_dofs, data.strain_matrices],
            outputs=[strain_gradients],
            device=model.device,
        )
        wp.launch(
            _compute_energy,
            dim=model.element_count,
            inputs=[strains, stresses, data.weights, data.stiffness_scale],
            outputs=[energy],
            device=model.device,
        )
        arguments = [
            strain_gradients,
            stresses,
            data.weights,
            data.stiffness_scale,
        ]
        wp.launch(
            _compute_gradient, dim=(model.element_count, 24), inputs=arguments, outputs=[gradient], device=model.device
        )
        hessian = None
        if tangent:
            # Scalar access avoids loading an entire 8x8 matrix for each geometric
            # stiffness entry. This is a zero-copy view of the same fixed data.
            matrix_entries = data.strain_matrices.flatten().view(wp.float64).reshape((*data.weights.shape, 384))
            stress_gradients = workspace.array(
                "stress_gradients", (*data.weights.shape, 24), _Vec6d, requires_grad=False
            )
            wp.launch(
                _compute_stress_gradients,
                dim=strain_gradients.shape,
                inputs=[strain_gradients, data.constitutive],
                outputs=[stress_gradients],
                device=model.device,
            )
            hessian = workspace.array("hessian", (model.element_count, 24, 24), requires_grad=False)
            wp.launch(
                _compute_hessian,
                dim=(model.element_count, 24, 24),
                inputs=[strain_gradients, stress_gradients, matrix_entries, *arguments[1:]],
                outputs=[hessian],
                device=model.device,
            )
        return energy, gradient, hessian
