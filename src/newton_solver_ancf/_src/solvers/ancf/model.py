# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ANCF coordinates use the public shared finite-element model in their own namespace."""

import numpy as np

from ..implicit import ImplicitModel


class ANCFModel(ImplicitModel):
    """Generalized ANCF reference data registered under State/Control.ancf."""

    namespace = "ancf"


def _element_values(value: float | np.ndarray, count: int, name: str) -> np.ndarray:
    values = np.broadcast_to(np.asarray(value, dtype=np.float64), (count,)).copy()
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} must contain finite, positive values")
    return values
