# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact sparse shell tangent plus the rank-one sealed-cavity derivative."""

import numpy as np


class ShellSparsity:
    """Fixed assembly map, including the eliminated bead/rim coordinates.

    Negative row/column indices denote fixed coordinates. Multiple bead-height
    entries may share the final reduced row; they are summed during assembly.
    """

    def __init__(self, rows, cols, size):
        rows, cols = np.asarray(rows), np.asarray(cols)
        self.keep = (rows >= 0) & (cols >= 0)
        keys, slots = np.unique(
            np.r_[rows[self.keep] + size * cols[self.keep], np.arange(size) * (size + 1)], return_inverse=True
        )
        self.element_slots = slots[: np.count_nonzero(self.keep)]
        self.rows, self.cols = keys % size, keys // size
        self.diagonal = np.flatnonzero(self.rows == self.cols)
        self.offsets = np.r_[0, np.cumsum(np.bincount(self.cols, minlength=size))]
        self.shape = (size, size)

    def assemble(self, elements, diagonal):
        from scipy.sparse import csc_matrix

        data = np.bincount(self.element_slots, weights=elements.ravel()[self.keep], minlength=len(self.rows))
        data[self.diagonal] += diagonal[self.rows[self.diagonal]]
        return csc_matrix((data, self.rows, self.offsets), shape=self.shape)


class SparseTangent:
    """Sparse shell matrix with an exact cavity outer product.

    The nonsymmetric follower-pressure tangent is solved by sparse LU of an
    augmented system. Its last scalar represents the cavity volume response,
    avoiding a global dense outer product or a required inverse of the local
    shell matrix. All residual checks use the original physical equations.
    """

    def __init__(self, local, left, right):
        self.local = local.tocsc()
        self.left, self.right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
        self.shape = local.shape
        self._factorization = None

    @property
    def T(self):
        return SparseTangent(self.local.T, self.right, self.left)

    def __matmul__(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim not in (1, 2) or vector.shape[0] != self.shape[1]:
            raise ValueError("Tangent product requires one or more vectors matching its column count")
        columns = vector.reshape(self.shape[1], -1)
        result = self.local @ columns + self.left[:, None] * (self.right @ columns)[None, :]
        return result.reshape(vector.shape)

    def diagonal(self):
        return self.local.diagonal() + self.left * self.right

    def _factor(self):
        if self._factorization is None:
            from scipy.sparse import bmat, csc_matrix, diags
            from scipy.sparse.linalg import splu

            scale = 1 / np.sqrt(np.maximum(np.abs(self.diagonal()), 1e-12))
            scaling = diags(scale, format="csc")
            # [S K S, S u; -(S v)^T, 1] [y; t] = [S rhs; 0]
            # gives x = S y and t = v^T x, so (K + u v^T) x = rhs.
            augmented = bmat(
                [
                    [scaling @ self.local @ scaling, csc_matrix((scale * self.left)[:, None])],
                    [csc_matrix((-scale * self.right)[None, :]), csc_matrix([[1.0]])],
                ],
                format="csc",
            )
            try:
                factor = splu(augmented)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Sparse equilibrium tangent is singular; no unique local pressure derivative"
                ) from exc
            self._factorization = scale, factor
        return self._factorization

    def solve(self, rhs):
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.shape[0] or not np.isfinite(rhs).all():
            raise ValueError("Expected finite right-hand-side vectors matching the tangent")
        scale, factor = self._factor()
        columns = rhs.reshape(self.shape[0], -1)
        answer = np.zeros_like(columns)
        residual = columns.copy()
        tolerance = 1e-11 + 1e-9 * np.max(np.abs(columns), axis=0)
        for _ in range(3):
            augmented_rhs = np.vstack((scale[:, None] * residual, np.zeros((1, columns.shape[1]))))
            answer += scale[:, None] * factor.solve(augmented_rhs)[:-1]
            residual = columns - self @ answer
            if np.isfinite(answer).all() and np.all(np.max(np.abs(residual), axis=0) <= tolerance):
                return answer.reshape(rhs.shape)
        raise RuntimeError("Sparse equilibrium solve has an excessive physical residual")
