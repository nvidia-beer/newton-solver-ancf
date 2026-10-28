"""Recover one bending-stiffness multiplier from synthetic cable motion using implicit gradients.

A two-element clamped cable generates eight nodal-position observations with
a known uniform bending-stiffness scale of 2. Starting from scale 1, the fit
minimizes the nodal trajectory error while keeping axial stiffness, geometry
and density fixed. It checks the implicit gradient against finite differences
before optimization. This is a test helper, not a menu example.
"""

import argparse

import newton
import numpy as np
import warp as wp

from newton_solver_ancf import SolverANCFCable

HEADLESS = True


@wp.kernel
def _loss(q: wp.array[wp.float64], target: wp.array2d[wp.float64], frame: int, out: wp.array[wp.float64]):
    node = wp.tid()
    error = wp.float64(0.0)
    for axis in range(3):
        delta = (q[6 * node + axis] - target[frame, 6 * node + axis]) / wp.float64(0.3)
        error += delta * delta
    wp.atomic_add(out, 0, error / wp.float64(target.shape[0] * (q.shape[0] // 6)))


def experiment(device="cpu"):
    positions = np.array([[0.0, 0.0, 0.4], [0.15, 0.0, 0.4], [0.3, 0.0, 0.4]])
    model = SolverANCFCable.create_model(
        positions,
        young_modulus=2.0e6,
        radius=0.006,
        density=1000.0,
        device=device,
        requires_grad=True,
    )
    builder = newton.ModelBuilder()
    model.register(builder)
    native = builder.finalize(device=device, requires_grad=True)
    solver = SolverANCFCable(native, model, fixed_dofs=range(6), mass_damping=2.0, tolerance=1.0e-9)
    base = model.parameters["bending_stiffness"].numpy().copy()

    def rollout(scale, target=None, gradient=False):
        model.parameters["bending_stiffness"].assign(base * scale)
        state = native.state(requires_grad=gradient)
        loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
        observed = []
        targets = wp.array(target, dtype=wp.float64, device=device) if target is not None else None
        tape = wp.Tape()
        with tape if gradient else wp.ScopedDevice(device):
            for frame in range(8):
                next_state = native.state(requires_grad=gradient)
                solver.step(state, next_state, None, None, 0.02)
                state = next_state
                observed.append(state)
                if targets is not None:
                    wp.launch(_loss, model.node_count, inputs=[state.ancf.q, targets, frame, loss], device=device)
        if target is None:
            return np.stack([state.ancf.q.numpy() for state in observed])
        if gradient:
            tape.backward(loss)
            derivative = float(np.sum(model.parameters["bending_stiffness"].grad.numpy() * base * scale))
            tape.zero()
            return float(loss.numpy()[0]), derivative
        return float(loss.numpy()[0])

    return rollout


def optimize(device="cpu", iterations=30):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rollout = experiment(device)
    target = rollout(2.0)
    theta = 0.0
    initial, gradient = rollout(np.exp(theta), target, True)
    eps = 1.0e-4
    finite = (rollout(np.exp(theta + eps), target) - rollout(np.exp(theta - eps), target)) / (2 * eps)
    relative = abs(gradient - finite) / max(abs(gradient), abs(finite), 1.0e-12)
    print(f"Gradient: adjoint={gradient:.9g}, finite difference={finite:.9g}, relative error={relative:.3e}")
    if relative > 2.0e-3:
        raise RuntimeError("Material gradient validation failed")
    loss = initial
    for iteration in range(iterations):
        loss, gradient = rollout(np.exp(theta), target, True)
        # One scalar parameter: safeguarded secant curvature avoids an optimizer dependency.
        curvature = (rollout(np.exp(theta + eps), target, True)[1] - rollout(np.exp(theta - eps), target, True)[1]) / (
            2 * eps
        )
        direction = -gradient / max(curvature, 1.0e-8)
        direction = float(np.clip(direction, -0.3, 0.3))
        for _ in range(12):
            trial = rollout(np.exp(theta + direction), target)
            if trial < loss:
                theta += direction
                loss = trial
                break
            direction *= 0.5
        print(f"Iteration {iteration + 1}: loss={loss:.7g}, stiffness scale={np.exp(theta):.6f}")
        if loss < 1.0e-12:
            break
    recovered = float(np.exp(theta))
    if not loss < initial * 0.01 or abs(recovered - 2.0) > 2.0e-3:
        raise RuntimeError(f"Material recovery failed: scale {recovered:.6f}, target 2.0, loss {loss:.6g}")
    print(f"Recovered scale {recovered:.6f}; target 2.0; loss {initial:.6g} -> {loss:.6g}")
    return recovered, initial, loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    optimize(args.device, args.iterations)


if __name__ == "__main__":
    main()
