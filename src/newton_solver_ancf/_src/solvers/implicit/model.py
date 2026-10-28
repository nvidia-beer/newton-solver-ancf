# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed reference data and Newton state registration for finite elements."""

from dataclasses import dataclass
from typing import ClassVar

import newton
import warp as wp
from warp.sparse import BsrMatrix


@dataclass
class ImplicitModel:
    """Fixed reference configuration and generalized coordinates for finite elements.

    Generalized coordinates are stored node by node. Each node begins with its
    position [m], followed by the formulation's dimensionless position gradients.
    The cable layout is ``[position.xyz, slope.xyz]`` per node. Reference geometry
    and topology are fixed after construction; stiffness arrays may be optimized.

    Attributes:
        formulation: Element formulation identifier.
        rest_q: Flattened reference generalized coordinates [m, dimensionless].
        elements: Element node indices.
        element_dofs: Element local-to-global generalized coordinate indices.
        lengths: Reference interpolation lengths [m].
        mass_matrix: Scalar float64 CSR consistent mass [kg, kg m, kg m²].
            Its fixed topology contains every element clique, including structural zeros.
        node_count: Number of nodes.
        element_count: Number of elements.
        dofs_per_node: Generalized coordinates per node.
        nodes_per_element: Nodes per element.
        device: Warp execution device.
        element_data: Dataclass instance containing formulation-specific fixed
            quadrature and material parameters, or ``None``. A ``parameters``
            property may expose named material arrays for optimization.
    """

    namespace: ClassVar[str] = "implicit"

    formulation: str
    rest_q: wp.array[wp.float64]
    elements: wp.array2d[wp.int32]
    element_dofs: wp.array2d[wp.int32]
    lengths: wp.array[wp.float64]
    mass_matrix: BsrMatrix
    node_count: int
    element_count: int
    dofs_per_node: int
    nodes_per_element: int
    device: wp.Device
    element_data: object | None = None

    def register(self, builder: newton.ModelBuilder):
        """Register generalized state/control arrays before public Model.finalize.

        One FEM model owns this namespace. References/mass stay in ImplicitModel;
        time-varying q/qd/qdd and force histories live in Newton State;
        applied loads and prescribed positions live in Control.
        """
        if builder.has_custom_attribute(f"{self.namespace}:q"):
            raise ValueError("A FEM model is already registered with this builder")
        builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name="dof", namespace=self.namespace))
        for name, assignment in (
            ("q", newton.Model.AttributeAssignment.STATE),
            ("qd", newton.Model.AttributeAssignment.STATE),
            ("qdd", newton.Model.AttributeAssignment.STATE),
            ("internal", newton.Model.AttributeAssignment.STATE),
            ("external", newton.Model.AttributeAssignment.STATE),
            ("load", newton.Model.AttributeAssignment.CONTROL),
            ("boundary_q", newton.Model.AttributeAssignment.CONTROL),
        ):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    dtype=wp.float64,
                    frequency=f"{self.namespace}:dof",
                    assignment=assignment,
                    namespace=self.namespace,
                    default=0.0,
                )
            )
        builder.add_custom_values_batch(
            [{f"{self.namespace}:q": float(q), f"{self.namespace}:boundary_q": float(q)} for q in self.rest_q.numpy()]
        )

    @property
    def parameters(self) -> dict[str, wp.array]:
        """Material arrays exposed by this formulation for optimization.

        Cable: ``axial_stiffness`` [N] and ``bending_stiffness`` [N m²].
        Beam: ``constitutive`` tensors [Pa]. Hex8 and ANCF tetrahedra:
        ``shear_modulus`` and ``bulk_modulus`` [Pa]. Shell:
        ``stiffness_scale`` [dimensionless]. Factories enable their gradients
        with ``requires_grad``.
        Geometry and mass remain fixed when these arrays are changed.
        """
        return getattr(self.element_data, "parameters", {})

    @property
    def dof_count(self) -> int:
        """Total number of generalized coordinates."""
        return self.node_count * self.dofs_per_node

    @property
    def element_dof_count(self) -> int:
        """Number of generalized coordinates in one element."""
        return self.nodes_per_element * self.dofs_per_node
