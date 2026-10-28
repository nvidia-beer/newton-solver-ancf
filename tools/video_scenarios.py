"""Presentation hooks for ANCF captures; simulation settings stay in videos.json."""

import json
from pathlib import Path

import numpy as np

_PROFILES = json.loads((Path(__file__).resolve().parents[1] / "docs" / "videos.json").read_text())


def configure(name, example):
    # Presentation only: keep the physics, asset resolution and iteration budgets.
    if name == "ancf_rigid_mujoco_tires":
        example.viewer.set_camera(pos=(-2.5, -2.5, 2.1), pitch=-19.0, yaw=62.0)
    elif name == "ancf_shell_drop":
        example.viewer.set_camera(pos=(2.4, 1.7, 2.4), pitch=-18.0, yaw=-135.0)
    elif name == "vehicle_ancf_tires":
        example.viewer.set_camera(pos=(-1.5, -4.8, 5.5), pitch=-34.0, yaw=57.0)


def before_frame(name, example, frame):
    # Replay manual commands only while recording documentation. These scenes
    # advance one 60 Hz simulation frame per captured frame.
    time = frame / 60.0
    profile = _PROFILES[name]
    schedule = profile.get("pressure_schedule")
    if schedule:
        times, pressures = np.asarray(schedule, dtype=float).T
        example.set_pressure(np.interp(time, times, pressures))
    schedule = profile.get("rpm_schedule")
    if schedule:
        times, rpms = np.asarray(schedule, dtype=float).T
        example.set_rpm(np.interp(time, times, rpms))
    schedule = profile.get("drive_schedule")
    if schedule:
        times, speeds, steering = np.asarray(schedule, dtype=float).T
        example.set_drive(np.interp(time, times, speeds), np.interp(time, times, steering))


def frame_label(name, example):
    if name == "ancf_rigid_mujoco_tires":
        return f"{example.rpms[0]:+.1f} RPM | nominal pressure {example.pressures[0] / 1000:.1f} kPa"
    if name == "vehicle_ancf_tires":
        turn = "left" if example.steer_angle > 0.05 else "right" if example.steer_angle < -0.05 else "straight"
        return f"steer {turn} | wheel command {example.wheel_speed:.1f} rad/s"
    if name == "diffsim_ancf_tire_lift":
        return (
            f"{example.pressure_pa / 6894.757:.2f} nominal PSI | "
            f"height error {example.height_error * 1000:+.2f} mm | "
            f"{example.training_status}"
        )
    return _PROFILES[name]["label"]
