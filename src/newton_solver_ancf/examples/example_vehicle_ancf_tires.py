# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Choose a MuJoCo vehicle with matching pressure-controlled ANCF tires and spindles on flat ground.

The shared runtime is in ``_vehicle_simulation``; acceptance checks are in
``newton_solver_ancf.examples._checks``. Existing launcher options are unchanged.
"""

import math

import newton.examples

from newton_solver_ancf.examples._vehicle_simulation import VehicleSimulation
from newton_solver_ancf.examples._wheel_assets import configure_wheels


class Example(VehicleSimulation):
    """Runnable example using the shared vehicle runtime."""

    def set_drive(self, wheel_speed: float, steering: float = 0.0) -> None:
        """Request the GUI drive controls; throttle ramps at the existing rate.

        Wheel speed is in rad/s. Steering is a brake-lever difference in [-1, 1]
        for skid steering (+ turns left), or an angle in radians for kingpins.
        """
        wheel_speed, steering = float(wheel_speed), float(steering)
        max_steering = 1.0 if self.vehicle.steering == "skid" else self.spec.max_steer
        if not math.isfinite(wheel_speed) or abs(wheel_speed) > self.spec.max_wheel_speed:
            raise ValueError("Wheel speed must be finite and within the vehicle's drive limits")
        if not math.isfinite(steering) or abs(steering) > max_steering:
            raise ValueError("Steering must be finite and within the vehicle's steering limits")
        self._target_wheel_speed = wheel_speed
        self.steer_angle = steering

    # The runner requires these hooks; test code is loaded only in test mode.
    def test_post_step(self) -> None:
        from newton_solver_ancf.examples._checks import check_vehicle_step  # noqa: PLC0415

        check_vehicle_step(self)

    def test_final(self) -> None:
        from newton_solver_ancf.examples._checks import check_vehicle_final  # noqa: PLC0415

        check_vehicle_final(self)


if __name__ == "__main__":
    parser = Example.create_parser()
    parser.set_defaults(
        viewer="gl",
        num_frames=600,
        headless=False,
        diag_period=5,
        wheel_speed=0.0,
        steer_angle=0.0,
        mu=0.9,
        substeps=6,
        nr_iters=2,
        pcg_iters=10,
        gs_iters=6,
        coupling_method="auto",
        ride_drop=0.03,
        fast_math=True,
        shell_tires=[
            {"E": 50000000.0, "nu": 0.45, "rho": 700.0, "alpha-damp": 0.15, "pressure": 30000.0, "thickness": None}
        ],
        ctis_fill_time=1.0,
        torque_alpha=1.0,
        vehicle_substeps={"superjeep_vehicle.usdc": 8},
    )
    configure_wheels(parser)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
