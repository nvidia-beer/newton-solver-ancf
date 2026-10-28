# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scalar CSR assembly and checked sparse solves for generalized coordinates."""

import numpy as np
import warp as wp
from warp.optim.linear import aslinearoperator, gmres
from warp.optim.linear import preconditioner as diagonal_preconditioner
from warp.sparse import bsr_copy, bsr_set_from_triplets, bsr_transposed, bsr_zeros


def assemble_matrix(dofs, local, size, device):
    """Sum exact element blocks; retain the element cliques as fixed CSR topology."""
    dofs = np.asarray(dofs, dtype=np.int32)
    if dofs.ndim != 2 or np.unique(dofs).size != size or np.any(dofs < 0) or np.any(dofs >= size):
        raise ValueError("Element topology must reference every generalized coordinate")
    width = dofs.shape[1]
    if np.shape(local) != (len(dofs), width, width):
        raise ValueError("Local matrix blocks must match element generalized coordinates")
    rows = np.broadcast_to(dofs[:, :, None], (len(dofs), width, width)).reshape(-1)
    columns = np.broadcast_to(dofs[:, None, :], (len(dofs), width, width)).reshape(-1)
    matrix = bsr_zeros(size, size, wp.float64, device=device)
    bsr_set_from_triplets(
        matrix,
        wp.array(rows, dtype=int, device=device),
        wp.array(columns, dtype=int, device=device),
        wp.array(np.asarray(local).reshape(-1), dtype=wp.float64, device=device),
        prune_numerical_zeros=False,
    )
    return matrix


def assembly_indices(matrix, dofs):
    """Cache each local Hessian entry's CSR slot without global dense storage."""
    if matrix.nrow != matrix.ncol or matrix.block_shape != (1, 1) or matrix.dtype != wp.float64:
        raise ValueError("Consistent mass must be a square scalar float64 CSR matrix")
    offsets, columns = matrix.offsets.numpy(), matrix.columns.numpy()
    size = matrix.nrow
    if (
        len(offsets) < size + 1
        or offsets[0] != 0
        or np.any(np.diff(offsets[: size + 1]) < 0)
        or offsets[size] > min(len(columns), matrix.values.size)
    ):
        raise ValueError("Consistent mass has invalid CSR offsets")
    active_columns = columns[: offsets[size]]
    if np.any(active_columns < 0) or np.any(active_columns >= size):
        raise ValueError("Consistent mass has out-of-range CSR columns")
    if dofs.ndim != 2 or np.unique(dofs).size != size or np.any(dofs < 0) or np.any(dofs >= size):
        raise ValueError("Element topology must reference every generalized coordinate")
    for row in range(size):
        if np.any(np.diff(columns[offsets[row] : offsets[row + 1]]) <= 0):
            raise ValueError("Consistent mass CSR columns must be strictly sorted")
    slots = np.empty((*dofs.shape, dofs.shape[1]), dtype=np.int32)
    for element, indices in enumerate(dofs):
        for local_row, row in enumerate(indices):
            row_columns = columns[offsets[row] : offsets[row + 1]]
            entries = np.searchsorted(row_columns, indices)
            if np.any(entries >= len(row_columns)) or not np.array_equal(row_columns[entries], indices):
                raise ValueError("Consistent mass topology must contain every element clique")
            slots[element, local_row] = offsets[row] + entries
    rows = np.repeat(np.arange(matrix.nrow, dtype=np.int32), np.diff(offsets[: matrix.nrow + 1]))
    return wp.array(slots, dtype=int, device=matrix.device), wp.array(rows, dtype=int, device=matrix.device)


@wp.kernel(enable_backward=False)
def _mask_matrix(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.float64],
    fixed: wp.array[int],
    shift: wp.float64,
):
    i = wp.tid()
    for entry in range(offsets[i], offsets[i + 1]):
        j = columns[entry]
        if fixed[i] or fixed[j]:
            values[entry] = wp.float64(1.0) if i == j else wp.float64(0.0)
        elif i == j:
            values[entry] += shift * wp.max(wp.abs(values[entry]), wp.float64(1.0e-20))


@wp.kernel(enable_backward=False)
def _mask_rhs(rhs: wp.array[wp.float64], fixed: wp.array[int], multiplier: wp.float64, out: wp.array[wp.float64]):
    i = wp.tid()
    out[i] = wp.float64(0.0) if fixed[i] else multiplier * rhs[i]


@wp.kernel(enable_backward=False)
def _initialize(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    inverse: wp.array[wp.float64],
    r: wp.array[wp.float64],
    p: wp.array[wp.float64],
    sums: wp.array[wp.float64],
    status: wp.array[int],
):
    i = wp.tid()
    diagonal = wp.float64(0.0)
    for entry in range(offsets[i], offsets[i + 1]):
        if columns[entry] == i:
            diagonal = values[entry]
    if diagonal <= wp.float64(0.0) or not wp.isfinite(diagonal):
        status[0] = -1
        diagonal = wp.float64(1.0)
    inverse[i] = wp.float64(1.0) / diagonal
    r[i] = rhs[i]
    p[i] = rhs[i] * inverse[i]
    wp.atomic_add(sums, 2, rhs[i] * p[i])
    wp.atomic_add(sums, 3, rhs[i] * rhs[i])


@wp.kernel(enable_backward=False)
def _begin(sums: wp.array[wp.float64], status: wp.array[int], tolerance_squared: wp.float64):
    if status[0] == 1 and sums[3] <= tolerance_squared:
        status[0] = 0
    sums[0] = sums[2]
    sums[1] = wp.float64(0.0)
    sums[2] = wp.float64(0.0)
    sums[3] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def _product(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.float64],
    p: wp.array[wp.float64],
    ap: wp.array[wp.float64],
    sums: wp.array[wp.float64],
    status: wp.array[int],
):
    i = wp.tid()
    if status[0] == 1:
        value = wp.float64(0.0)
        for entry in range(offsets[i], offsets[i + 1]):
            value += values[entry] * p[columns[entry]]
        ap[i] = value
        wp.atomic_add(sums, 1, p[i] * value)


@wp.kernel(enable_backward=False)
def _update(
    x: wp.array[wp.float64],
    r: wp.array[wp.float64],
    p: wp.array[wp.float64],
    ap: wp.array[wp.float64],
    inverse: wp.array[wp.float64],
    sums: wp.array[wp.float64],
    status: wp.array[int],
):
    i = wp.tid()
    if status[0] == 1:
        if sums[1] <= wp.float64(0.0) or not wp.isfinite(sums[1]) or not wp.isfinite(sums[0]):
            status[0] = -1
        else:
            alpha = sums[0] / sums[1]
            x[i] += alpha * p[i]
            r[i] -= alpha * ap[i]
            wp.atomic_add(sums, 2, r[i] * r[i] * inverse[i])
            wp.atomic_add(sums, 3, r[i] * r[i])


@wp.kernel(enable_backward=False)
def _direction(
    r: wp.array[wp.float64],
    inverse: wp.array[wp.float64],
    p: wp.array[wp.float64],
    sums: wp.array[wp.float64],
    status: wp.array[int],
    tolerance_squared: wp.float64,
):
    i = wp.tid()
    if status[0] == 1:
        if not wp.isfinite(sums[2]) or not wp.isfinite(sums[3]):
            status[0] = -1
        elif sums[3] <= tolerance_squared:
            status[0] = 0
        else:
            p[i] = r[i] * inverse[i] + (sums[2] / sums[0]) * p[i]


def _pcg(matrix, rhs, *, maxiter, tol, atol):
    """Reject nonpositive encountered curvature; never return an unchecked iterate."""
    n, device = rhs.size, rhs.device
    solution, inverse, r, p, ap = (wp.zeros_like(rhs) for _ in range(5))
    sums = wp.zeros(4, dtype=wp.float64, device=device)
    status = wp.ones(1, dtype=int, device=device)
    args = [matrix.offsets, matrix.columns, matrix.values]
    wp.launch(_initialize, n, inputs=[*args, rhs, inverse, r, p, sums, status], device=device)
    norm_squared = float(sums.numpy()[3])
    threshold = max(atol * atol, tol * tol * norm_squared)
    for first in range(0, maxiter, 8):
        for _ in range(min(8, maxiter - first)):
            wp.launch(_begin, 1, inputs=[sums, status, threshold], device=device)
            wp.launch(_product, n, inputs=[*args, p, ap, sums, status], device=device)
            wp.launch(_update, n, inputs=[solution, r, p, ap, inverse, sums, status], device=device)
            wp.launch(_direction, n, inputs=[r, inverse, p, sums, status, threshold], device=device)
        result = int(status.numpy()[0])
        if result != 1:
            return solution if result == 0 else None
    return None


@wp.kernel(enable_backward=False)
def _norms(residual: wp.array[wp.float64], rhs: wp.array[wp.float64], out: wp.array[wp.float64]):
    i = wp.tid()
    error = wp.abs(residual[i]) if wp.isfinite(residual[i]) else wp.float64(wp.inf)
    wp.atomic_max(out, 0, error)
    wp.atomic_max(out, 1, wp.abs(rhs[i]))


def _checked(operator, rhs, solution, *, tol, atol):
    residual = wp.empty_like(rhs)
    aslinearoperator(operator).matvec(solution, rhs, residual, 1.0, -1.0)
    norms = wp.zeros(2, dtype=wp.float64, device=rhs.device)
    wp.launch(_norms, rhs.size, inputs=[residual, rhs, norms], device=rhs.device)
    error, scale = norms.numpy()
    return solution if np.isfinite(error) and error <= atol + tol * scale else None


def checked_gmres(operator, rhs, preconditioner=None, *, maxiter=None, restart=31, tol=1.0e-9, atol=1.0e-11):
    """Solve an operator and check its actual residual, including the zero RHS case."""
    solution = wp.zeros_like(rhs)
    gmres(
        operator,
        rhs,
        solution,
        M=preconditioner,
        tol=tol * 0.01,
        atol=atol * 0.01,
        maxiter=max(64, 4 * rhs.size) if maxiter is None else maxiter,
        restart=min(restart, rhs.size),
        check_every=restart,
    )
    return _checked(operator, rhs, solution, tol=tol, atol=atol)


def solve(matrix, rhs, fixed, *, regularization=0.0, multiplier=1.0, transpose=False, method="cg"):
    """Solve the free CSR operator. A failed solve returns None for the Newton shift loop.

    Fixed rows and columns are eliminated only here; residuals retain full mass
    and stiffness coupling. Transpose solves never inherit a forward shift.
    """
    operator = bsr_transposed(matrix) if transpose else bsr_copy(matrix)
    wp.launch(
        _mask_matrix,
        rhs.size,
        inputs=[operator.offsets, operator.columns, operator.values, fixed, regularization],
        device=rhs.device,
    )
    target = wp.empty_like(rhs)
    wp.launch(_mask_rhs, rhs.size, inputs=[rhs, fixed, multiplier, target], device=rhs.device)
    if method == "gmres":
        return checked_gmres(operator, target, diagonal_preconditioner(operator, "diag_abs"))
    if method != "cg":
        raise ValueError("Sparse method must be cg or gmres")
    solution = _pcg(operator, target, maxiter=max(64, 4 * rhs.size), tol=1.0e-12, atol=1.0e-13)
    return None if solution is None else _checked(operator, target, solution, tol=1.0e-9, atol=1.0e-11)
