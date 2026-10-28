# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reusable scratch arrays for element evaluation."""

import warp as wp


class EvaluationWorkspace:
    """Reuse named buffers for forward work; use a fresh instance per taped evaluation."""

    def __init__(self, device, *, requires_grad=False):
        self.device = device
        self.requires_grad = requires_grad
        self.arrays = {}

    def array(self, name, shape, dtype=wp.float64, *, zero=False, requires_grad=None):
        if name not in self.arrays:
            self.arrays[name] = wp.empty(
                shape,
                dtype=dtype,
                device=self.device,
                requires_grad=self.requires_grad if requires_grad is None else requires_grad,
            )
        result = self.arrays[name]
        if zero:
            result.zero_()
        return result
