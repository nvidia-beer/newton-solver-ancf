# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Split forward stepping for the pinned stock Newton MuJoCo solver."""

import math

import newton
import warp as wp
from newton.solvers import SolverMuJoCo as _SolverMuJoCo
from warp._src import context as wp_context


class SolverMuJoCo(_SolverMuJoCo):
    """Stock Newton with a force-insertion point between kinematics and dynamics.

    Call ``step_kinematics``, accumulate world-axis forces and COM torques in
    ``xfrc_applied``, then call ``step_dynamics`` once. Ordinary ``step`` and all
    model/control updates retain stock Newton behavior. Split stepping requires
    MuJoCo Warp on CUDA and provides no backward operation. RK4 is rejected
    because MuJoCo's split step would silently use Euler instead.

    The implementation reuses the pinned Newton 1.7 conversion/execution
    methods. It is not a native-data checkpoint: repeated completed steps also
    advance MuJoCo's clock and warm starts.
    """

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
        *,
        observables: _SolverMuJoCo.Observables | None = None,
    ) -> None:
        """Delegate ordinary stepping to Newton; a split phase must finish first."""
        if getattr(self, "_split_state_in", None) is not None:
            raise RuntimeError("Complete step_dynamics or reset before ordinary step")
        super().step(state_in, state_out, control, contacts, dt, observables=observables)

    def reset(
        self,
        state: newton.State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: newton.StateFlags | int | None = None,
        *,
        time: float | None = None,
    ) -> None:
        """Apply stock reset, cancel a pending split and optionally set time [s].

        A supplied time applies to all worlds. This only resets the clock in
        addition to stock behavior; it does not restore a native-data snapshot.
        """
        if time is not None:
            if not math.isfinite(time) or time < 0:
                raise ValueError("time must be finite and nonnegative")
            if world_mask is not None:
                raise ValueError("time reset does not support a world_mask")
        super().reset(state, world_mask=world_mask, flags=flags)
        self._split_state_in = None
        if time is not None:
            if self.use_mujoco_cpu:
                self.mj_data.time = time
            else:
                self.mjw_data.time.fill_(time)

    def _check_split(self):
        if self.use_mujoco_cpu or not self.model.device.is_cuda:
            raise NotImplementedError("Split stepping requires the MuJoCo Warp CUDA backend")
        if self.mjw_model.opt.integrator == self._mujoco.mjtIntegrator.mjINT_RK4:
            raise NotImplementedError("Split stepping does not support RK4")
        if wp_context.runtime.tape is not None:
            raise RuntimeError("Split MuJoCo backward is not implemented")

    def step_kinematics(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
    ) -> None:
        """Evaluate position/velocity, collision and passive forces for step ``dt`` [s].

        State inputs synchronize according to ``update_data_interval``, as in
        stock ``step``. Applied forces are copied from ``state_in.body_f`` here;
        add coupling wrenches afterwards. This phase does not advance time or
        overwrite ``state_out``. Repeating it replaces the pending input state.
        """
        self._check_split()
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not self.mjw_model.opt.run_collision_detection and contacts is None:
            raise ValueError("External contact mode requires contact storage")
        with wp.ScopedDevice(self.model.device), self._scoped_mujoco_warp_execution():
            self._enable_rne_postconstraint(state_out)
            self._apply_mjc_control(self.model, state_in, control, self.mjw_data)
            if self.update_data_interval > 0 and self._step % self.update_data_interval == 0:
                self._update_mjc_data(self.mjw_data, self.model, state_in)
            self.mjw_model.opt.timestep.fill_(dt)
            if not self.mjw_model.opt.run_collision_detection:
                self._convert_contacts_to_mjwarp(self.model, state_in, contacts)
            self._mujoco_warp.step1(self.mjw_model, self.mjw_data)
        self._split_state_in = state_in

    def step_dynamics(self, state_out: newton.State) -> None:
        """Solve constraints and integrate the pending step into ``state_out``.

        Wrenches inserted after kinematics enter the native acceleration and
        constraint solve. This consumes the pending phase and advances the
        native clock and Newton step counter once.
        """
        self._check_split()
        state_in = getattr(self, "_split_state_in", None)
        if state_in is None:
            raise RuntimeError("Call step_kinematics before step_dynamics")
        with wp.ScopedDevice(self.model.device), self._scoped_mujoco_warp_execution():
            self._enable_rne_postconstraint(state_out)
            self._mujoco_warp.step2(self.mjw_model, self.mjw_data)
            self._update_newton_state(self.model, state_out, self.mjw_data, state_prev=state_in)
        self._split_state_in = None
        self._step += 1

    @property
    def xfrc_applied(self) -> wp.array2d[wp.spatial_vector]:
        """Writable world wrenches [N, N m], ordered force then body-COM torque.

        Indexed by MuJoCo ``[world, body]``. Kinematics replaces these values
        from Newton inputs; dynamics does not automatically zero them.
        """
        return self.mjw_data.xfrc_applied

    @property
    def cvel(self) -> wp.array2d[wp.spatial_vector]:
        """World-axis spatial velocities about each body's root-subtree COM.

        Native order is angular then linear [rad/s, m/s]; the reference point
        is not generally the body's own center of mass.
        """
        return self.mjw_data.cvel

    @property
    def xpos(self) -> wp.array2d[wp.vec3]:
        """Body-frame origin positions [m], indexed by MuJoCo ``[world, body]``."""
        return self.mjw_data.xpos

    @property
    def xquat(self) -> wp.array2d[wp.quat]:
        """Native body orientations, stored in MuJoCo wxyz order [dimensionless]."""
        return self.mjw_data.xquat
