# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Block-local symmetric Gauss-Seidel for small ANCF systems."""

import functools

import numpy as np
import warp as wp


@functools.cache
def make_colored_sweep(padded: int, n_colors: int):
    """Sweep preselected lower/upper block neighbors in one CUDA block per tire."""

    @wp.kernel(enable_backward=False, module="unique")
    def sweep(
        starts: wp.array2d[int],
        blocks: wp.array[int],
        columns: wp.array[int],
        scaled: wp.array[float],
        nodes: wp.array2d[int],
        counts: wp.array[int],
        output: wp.array[float],
        nnz: int,
    ):
        env, lane = wp.tid()
        z = wp.tile_zeros(shape=padded, dtype=float, storage="shared")
        # The highest-color backward pass has no upper neighbors.
        for pass_index in range(wp.static(2 * n_colors - 1)):
            forward = pass_index < n_colors
            color = pass_index
            direction = int(0)
            if not forward:
                color = 2 * n_colors - pass_index - 2
                direction = 1
            active = lane < 6 * counts[color]
            node = int(0)
            row = lane % 6
            value = float(0.0)
            if active:
                node = nodes[color, lane // 6]
                if forward:
                    value = output[env * padded + node * 6 + row]
                else:
                    value = z[node * 6 + row]
                for edge in range(starts[direction, node], starts[direction, node + 1]):
                    block = blocks[edge]
                    other = columns[block]
                    for col in range(6):
                        value -= scaled[env * nnz + block * 36 + row * 6 + col] * z[other * 6 + col]
            # Every lane reaches the barrier, including inactive padding.
            wp.tile_scatter_masked(z, node * 6 + row, value, active)
        wp.tile_store(output, z, offset=env * padded)

    return sweep


def packed_sweep_layout(starts, blocks, columns, nodes, counts, nnz):
    """Construct a lane-contiguous layout, keeping each node's CSR edge order."""
    nc = len(counts)
    width = ((int(max(counts)) * 6 + 31) // 32) * 32
    maxedges = int(np.diff(starts, axis=1).max())
    size = 2 * nc * maxedges * 6 * width
    # Dense graphs with many small colors can waste excessive padding.
    if size == 0 or size > 4 * nnz:
        return None
    source = np.full((2, nc, maxedges, 6, width), -1, np.int32)
    other = np.zeros((2, nc, maxedges, nodes.shape[1]), np.int32)
    degree = np.zeros((2, nc, nodes.shape[1]), np.int32)
    for direction in range(2):
        for color in range(nc):
            for slot in range(counts[color]):
                node = nodes[color, slot]
                edges = blocks[starts[direction, node] : starts[direction, node + 1]]
                degree[direction, color, slot] = len(edges)
                for edge, block in enumerate(edges):
                    other[direction, color, edge, slot] = columns[block]
                    for row in range(6):
                        for col in range(6):
                            source[direction, color, edge, col, slot * 6 + row] = block * 36 + row * 6 + col
    return source.reshape(-1), other, degree, width, maxedges


@wp.kernel(enable_backward=False)
def pack_sweep_values(
    values: wp.array[float],
    inverse: wp.array[float],
    rows: wp.array[int],
    source: wp.array[int],
    n_nodes: int,
    nnz: int,
    size: int,
    out: wp.array[float],
    repeats: int,
):
    """Scale only the off-diagonal entries used by SGS, directly into packed storage."""
    i = wp.tid()
    env = (i // size) * repeats
    index = source[i % size]
    value = float(0.0)
    if index >= 0:
        block = index // 36
        row, col = (index % 36) // 6, index % 6
        inv_base = (env * n_nodes + rows[block]) * 36 + row * 6
        for k in range(6):
            value += inverse[inv_base + k] * values[env * nnz + block * 36 + k * 6 + col]
    out[env * size + i % size] = value


@functools.cache
def make_packed_colored_sweep(padded: int, ncolors: int, width: int, maxedges: int):
    """Coalesce matrix reads without changing the CSR summation order."""

    @wp.kernel(enable_backward=False, module="unique")
    def sweep(
        packed: wp.array[float],
        other: wp.array4d[int],
        degree: wp.array3d[int],
        nodes: wp.array2d[int],
        counts: wp.array[int],
        output: wp.array[float],
        repeats: int,
    ):
        env, lane = wp.tid()
        matrix_env = (env // repeats) * repeats
        z = wp.tile_zeros(shape=padded, dtype=float, storage="shared")
        for phase in range(wp.static(2 * ncolors - 1)):
            forward = phase < ncolors
            color = phase
            direction = int(0)
            if not forward:
                color = 2 * ncolors - phase - 2
                direction = 1
            active = lane < 6 * counts[color]
            node = int(0)
            row = lane % 6
            value = float(0.0)
            if active:
                slot = lane // 6
                node = nodes[color, slot]
                if forward:
                    value = output[env * padded + node * 6 + row]
                else:
                    value = z[node * 6 + row]
                for edge in range(degree[direction, color, slot]):
                    neighbor = other[direction, color, edge, slot]
                    for col in range(6):
                        index = (
                            (((matrix_env * 2 + direction) * ncolors + color) * maxedges + edge) * 6 + col
                        ) * width + lane
                        value -= packed[index] * z[neighbor * 6 + col]
            wp.tile_scatter_masked(z, node * 6 + row, value, active)
        wp.tile_store(output, z, offset=env * padded)

    return sweep


@wp.kernel(enable_backward=False)
def pack_matrix(values: wp.array[float], source: wp.array[int], nnz: int, out: wp.array[float], repeats: int):
    env, i = wp.tid()
    env *= repeats
    index = source[i]
    value = float(0.0)
    if index >= 0:
        value = values[env * nnz + index]
    out[env * source.shape[0] + i] = value


@functools.cache
def make_local_pcg(padded: int, ncolors: int, width: int, maxedges: int, block_dim: int, row_degree: int):
    """Keep small-system PCG vectors in shared memory between iterations."""

    # Warp 1.17 needs the existing launch size to bound this fused kernel's
    # register allocation; the numerical work and thread layout are unchanged.
    @wp.kernel(enable_backward=False, module="unique", launch_bounds=block_dim)
    def solve(
        columns: wp.array2d[int],
        values: wp.array[float],
        inverse: wp.array[float],
        packed: wp.array[float],
        other: wp.array4d[int],
        degree: wp.array3d[int],
        nodes: wp.array2d[int],
        counts: wp.array[int],
        solution: wp.array[float],
        residual: wp.array[float],
        preconditioned: wp.array[float],
        rz_last: wp.array[wp.float64],
        n: int,
        nnz: int,
        iterations: int,
        report: bool,
        repeats: int,
    ):
        env, lane = wp.tid()
        matrix_env = (env // repeats) * repeats
        base = env * padded
        x = wp.tile_load(solution, shape=padded, offset=base, storage="shared")
        r = wp.tile_load(residual, shape=padded, offset=base, storage="shared")
        z = wp.tile_load(preconditioned, shape=padded, offset=base, storage="shared")
        p = wp.tile_zeros(shape=padded, dtype=float, storage="shared")
        ap = wp.tile_zeros(shape=padded, dtype=float, storage="shared")
        rz = wp.tile_sum(r * z)[0]
        previous = float(0.0)
        for iteration in range(iterations):
            beta = float(0.0)
            if wp.abs(previous) > 1.0e-30:
                beta = rz / previous
            p = z + beta * p
            for chunk in range(wp.static((padded + block_dim - 1) // block_dim)):
                i = chunk * block_dim + lane
                active = i < n
                acc = float(0.0)
                if active:
                    node = i // 6
                    for edge in range(wp.static(row_degree)):
                        col = 6 * columns[edge, node]
                        for j in range(6):
                            v = ((matrix_env * row_degree + edge) * 6 + j) * padded + i
                            acc += values[v] * p[col + j]
                wp.tile_scatter_masked(ap, i, acc, active)
            pap = wp.tile_sum(p * ap)[0]
            alpha = float(0.0)
            if wp.abs(pap) > 1.0e-30:
                alpha = rz / pap
            x = x + alpha * p
            r = r - alpha * ap
            if report or iteration + 1 < iterations:
                for chunk in range(wp.static((padded + block_dim - 1) // block_dim)):
                    i = chunk * block_dim + lane
                    active = i < n
                    value = float(0.0)
                    if active:
                        node = i // 6
                        inv_base = (matrix_env * (n // 6) + node) * 36 + (i % 6) * 6
                        for col in range(6):
                            value += inverse[inv_base + col] * r[node * 6 + col]
                    wp.tile_scatter_masked(z, i, value, active)
                for phase in range(wp.static(2 * ncolors - 1)):
                    color = phase
                    direction = int(0)
                    if phase >= ncolors:
                        color = 2 * ncolors - phase - 2
                        direction = 1
                    active = lane < 6 * counts[color]
                    node = int(0)
                    row = lane % 6
                    value = float(0.0)
                    if active:
                        slot = lane // 6
                        node = nodes[color, slot]
                        value = z[node * 6 + row]
                        for edge in range(degree[direction, color, slot]):
                            neighbor = other[direction, color, edge, slot]
                            for col in range(6):
                                index = (
                                    (((matrix_env * 2 + direction) * ncolors + color) * maxedges + edge) * 6 + col
                                ) * width + lane
                                value -= packed[index] * z[neighbor * 6 + col]
                    wp.tile_scatter_masked(z, node * 6 + row, value, active)
                previous = rz
                rz = wp.tile_sum(r * z)[0]
        wp.tile_store(solution, x, offset=base)
        wp.tile_store(residual, r, offset=base)
        if lane == 0:
            rz_last[env] = wp.float64(rz)

    return solve
