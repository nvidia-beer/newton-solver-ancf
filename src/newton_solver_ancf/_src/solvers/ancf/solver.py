# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ANCF element family using the shared converged HHT integrator and adjoint."""

from ..implicit import ImplicitState as ANCFState
from ..implicit import ImplicitStepInfo as ANCFStepInfo
from ..implicit import SolverImplicit

__all__ = ["SolverANCF", "ANCFState", "ANCFStepInfo"]


class SolverANCF(SolverImplicit):
    """Common HHT-alpha base for ANCF cable, beam, shell and tetrahedral elements."""
