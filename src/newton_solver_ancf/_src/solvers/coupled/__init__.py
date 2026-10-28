# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Rigid-deformable attachments and ANCF shell/spindle coupling."""

from .attachment import MuJoCoAttachment
from .coupling import InterfaceCouplerGS
from .solver_ancf_shell_rigid import SolverANCFShellRigid

__all__ = ["MuJoCoAttachment", "InterfaceCouplerGS", "SolverANCFShellRigid"]
