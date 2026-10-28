"""Reference assets and launch settings, including the validated drop correction."""

import importlib
import io
import json
import runpy
import shutil
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull

from newton_solver_ancf import examples, load_ancf_tire_usd
from newton_solver_ancf.examples._shell_assets import discover_assets
from newton_solver_ancf.examples._wheel_assets import discover_vehicles

ROOT = Path(examples.__file__).parent
ASSETS = ROOT / "assets"


class _Parsed(Exception):
    pass


def parsed_entry(name, arguments=(), *, interactive=False):
    module = f"newton_solver_ancf.examples.example_{name}"
    captured = {}

    def init(parser):
        captured.update(vars(parser.parse_args()))
        raise _Parsed

    with (
        patch("newton.examples.init", init),
        patch.object(sys, "argv", [module, *arguments]),
        patch("sys.stdin.isatty", return_value=interactive),
    ):
        try:
            path = ROOT / f"example_{name}.py"
            runpy.run_path(str(path), run_name="__main__")
        except _Parsed:
            pass
    return captured


class ReferenceExamplesTests(unittest.TestCase):
    def test_vehicle_tire_pair_discovery(self):
        vehicles = discover_vehicles()
        self.assertEqual(len(vehicles), 3)
        tires = [name for vehicle in vehicles.values() for name in vehicle["tires"]]
        self.assertEqual(len(tires), 5)
        self.assertFalse(any("ball" in name for name in tires))
        for vehicle in vehicles.values():
            self.assertTrue(all(info["spindle"] for info in vehicle["tires"].values()))
        with tempfile.TemporaryDirectory() as directory:
            from pxr import Usd

            directory = Path(directory)
            shutil.copyfile(ASSETS / "warthog_vehicle.usdc", directory / "custom_vehicle.usdc")
            shutil.copyfile(ASSETS / "warthog_ancf_tire_simple.usda", directory / "custom_tire.usda")
            stage = Usd.Stage.Open(str(directory / "custom_vehicle.usdc"))
            root = stage.GetDefaultPrim()
            root.SetCustomDataByKey("defaultTireAsset", "custom_tire.usda")
            root.SetCustomDataByKey("simpleTireAsset", "custom_tire.usda")
            stage.GetRootLayer().Save()
            discovered = discover_vehicles(directory)
            self.assertEqual(list(discovered), ["custom_vehicle.usdc"])
            self.assertEqual(list(discovered["custom_vehicle.usdc"]["tires"]), [str(directory / "custom_tire.usda")])

    def test_coupled_example_asset_selection(self):
        for name in ("ancf_rigid_mujoco_tires", "vehicle_ancf_tires"):
            with self.subTest(example=name):
                with patch("builtins.input", side_effect=["sherp_vehicle.usdc", "sherp_ancf_tire_simple.usda", ""]):
                    selected = parsed_entry(name, interactive=True)
                self.assertEqual(
                    (selected["vehicle_asset"], selected["tire_asset"]),
                    ("sherp_vehicle.usdc", "sherp_ancf_tire_simple.usda"),
                )
                self.assertEqual(selected["substeps"], 6)
                selected = parsed_entry(name, ("--vehicle-asset", "superjeep_vehicle.usdc"))
                self.assertEqual(selected["tire_asset"], "superjeep_tire.usda")
                self.assertEqual(selected["substeps"], 8 if name == "vehicle_ancf_tires" else 6)
                selected = parsed_entry(name, ("--tire-asset", "sherp_ancf_tire.usda", "--substeps", "10"))
                self.assertEqual(selected["vehicle_asset"], "sherp_vehicle.usdc")
                self.assertEqual(selected["substeps"], 10)
                with patch("builtins.input") as prompt:
                    selected = parsed_entry(name, ("--viewer", "null"), interactive=True)
                    prompt.assert_not_called()
                expected = (
                    "warthog_ancf_tire.usda" if name == "ancf_rigid_mujoco_tires" else "warthog_ancf_tire_simple.usda"
                )
                self.assertEqual(selected["tire_asset"], expected)
                with patch("sys.stdout", new_callable=io.StringIO) as output, self.assertRaises(SystemExit) as result:
                    parsed_entry(name, ("--list-assets",))
                self.assertEqual(result.exception.code, 0)
                self.assertIn("sherp_vehicle.usdc", output.getvalue())
                self.assertNotIn("ball", output.getvalue())
                with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as result:
                    parsed_entry(name, ("--vehicle-asset", "warthog_vehicle.usdc", "--tire-asset", "minimal_ball.usda"))
                self.assertEqual(result.exception.code, 2)
                with tempfile.TemporaryDirectory() as directory:
                    vehicle = Path(directory) / "custom_vehicle.usdc"
                    shutil.copyfile(ASSETS / "warthog_vehicle.usdc", vehicle)
                    tire = ASSETS / "warthog_ancf_tire_simple.usda"
                    selected = parsed_entry(name, ("--vehicle-asset", str(vehicle), "--tire-asset", str(tire)))
                    self.assertEqual((selected["vehicle_asset"], selected["tire_asset"]), (str(vehicle), str(tire)))
        selected = parsed_entry(
            "ancf_rigid_mujoco_tires", ("--vehicle-asset", "", "--tire-asset", "superjeep_tire.usda")
        )
        self.assertEqual(selected["vehicle_asset"], "")

    def test_combined_elements_use_reference_showcase_timestep(self):
        from newton_solver_ancf.examples.example_elements import Example

        # Reference config/24_nonlinear_fem.json applies four substeps to every
        # specimen, including the shell whose standalone default was two.
        for arguments, substeps in (([], 4), (["--substeps", "3"], 3)):
            with self.subTest(arguments=arguments), wp.ScopedDevice("cpu"), ExitStack() as stack:
                args = Example.create_parser().parse_args(arguments)
                example = Example(ViewerNull(), args)
                advances = {
                    name: stack.enter_context(patch.object(element.solver, "advance", wraps=element.solver.advance))
                    for name, element in example.elements.items()
                }
                example.step()
                self.assertAlmostEqual(example.sim_time, 1.0 / 60.0)
                for name, element in example.elements.items():
                    self.assertEqual(element.sim_substeps, substeps)
                    self.assertAlmostEqual(element.sim_time, example.sim_time)
                    self.assertEqual(advances[name].call_count, substeps)
                    intervals = [call.args[1] for call in advances[name].call_args_list]
                    np.testing.assert_allclose(intervals, np.full(substeps, 1.0 / (60 * substeps)))
                    self.assertAlmostEqual(sum(intervals), example.frame_dt)

    def test_tire_launcher_reference_substeps(self):
        # Effective defaults of reference launcher configs 01, 02 and 03.
        for name in ("ancf_shell_drop", "ancf_rigid_mujoco_tires", "vehicle_ancf_tires"):
            with self.subTest(example=name):
                self.assertEqual(parsed_entry(name)["substeps"], 6)
                self.assertEqual(parsed_entry(name, ("--substeps", "8"))["substeps"], 8)

    def test_shell_asset_discovery_and_loading(self):
        assets = discover_assets()
        self.assertNotIn("warthog_vehicle.usdc", assets)
        self.assertEqual(len(assets), 8)
        for name, info in assets.items():
            with self.subTest(asset=name):
                mesh, meta = load_ancf_tire_usd(str(ASSETS / name), device="cpu")
                self.assertEqual((mesh.n_nodes, mesh.n_elems), (info["nodes"], info["quads"]))
                self.assertEqual((meta.pressure, meta.contact_kn), (info["pressure"], info["kn"]))
                self.assertTrue(np.isfinite(mesh.node_x0.numpy()).all())
        with tempfile.TemporaryDirectory() as directory:
            from pxr import Usd

            directory = Path(directory)
            shutil.copyfile(ASSETS / "minimal_ball.usda", directory / "custom_shell.usda")
            (directory / "visual.usda").write_text('#usda 1.0\ndef Mesh "Visual" {}\n')
            discovered = discover_assets(directory)
            self.assertEqual(list(discovered), ["custom_shell.usda"])
            self.assertEqual(discovered["custom_shell.usda"], assets["minimal_ball.usda"])
            stage = Usd.Stage.Open(str(directory / "custom_shell.usda"))
            prim = stage.GetPrimAtPath("/Tire/Mesh")
            fields = prim.GetCustomDataByKey("materialFields")
            fields[0], fields[1] = fields[1], fields[0]
            prim.SetCustomDataByKey("materialFields", fields)
            stage.GetRootLayer().Save()
            self.assertEqual(discover_assets(directory), {})

    def test_drop_selection_and_cli_precedence(self):
        with patch("builtins.input", side_effect=["warthog_ancf_tire_simple.usda", "9"]):
            selected = parsed_entry("ancf_shell_drop", interactive=True)
        self.assertEqual(Path(selected["tire_asset"]).name, "warthog_ancf_tire_simple.usda")
        self.assertEqual(selected["substeps"], 9)
        self.assertEqual(selected["kn"], discover_assets()["warthog_ancf_tire_simple.usda"]["kn"])
        with patch("builtins.input") as prompt:
            override = parsed_entry(
                "ancf_shell_drop",
                ("--tire-asset", "ball_240.usda", "--substeps=8", "--kn=1234"),
                interactive=True,
            )
            prompt.assert_not_called()
        self.assertEqual((override["tire_asset"], override["substeps"], override["kn"]), ("ball_240.usda", 8, 1234))
        with patch("builtins.input") as prompt:
            headless = parsed_entry("ancf_shell_drop", ("--viewer", "null"), interactive=True)
            prompt.assert_not_called()
        self.assertEqual(Path(headless["tire_asset"]).name, "minimal_ball.usda")
        energy_bound = parsed_entry("ancf_shell_drop", ("--contact-penetration=0.02",))
        self.assertIsNone(energy_bound["kn"])

    def test_asset_list_exits_before_viewer_initialization(self):
        with patch("sys.stdout", new_callable=io.StringIO) as output, self.assertRaises(SystemExit) as result:
            parsed_entry("ancf_shell_drop", ("--list-assets",))
        self.assertEqual(result.exception.code, 0)
        self.assertIn("ball_960.usda", output.getvalue())
        self.assertIn("2128 nodes / 2016 quads", output.getvalue())

    def test_reference_defaults_and_explicit_overrides(self):
        cases = (
            ("ancf_shell_drop", 6, 3, 200, 300),
            ("ancf_rigid_mujoco_tires", 6, 2, 10, 300),
            ("vehicle_ancf_tires", 6, 2, 10, 600),
        )
        for name, substeps, newton_iters, pcg_iters, frames in cases:
            with self.subTest(example=name):
                parsed = parsed_entry(name)
                self.assertEqual(
                    (parsed["substeps"], parsed["nr_iters"], parsed["pcg_iters"], parsed["num_frames"]),
                    (substeps, newton_iters, pcg_iters, frames),
                )
        rig = parsed_entry("ancf_rigid_mujoco_tires")
        self.assertEqual(len(rig["shell_tires"]), 4)
        self.assertEqual([t["position"][0] for t in rig["shell_tires"]], [0, 1.2, 2.4, 3.6])
        self.assertTrue(all(t["E"] == 50e6 and t["alpha-damp"] == 0.15 for t in rig["shell_tires"]))
        vehicle = parsed_entry("vehicle_ancf_tires")
        self.assertEqual((vehicle["gs_iters"], vehicle["coupling_method"]), (6, "auto"))
        lift = parsed_entry("diffsim_ancf_tire_lift")
        self.assertEqual((lift["train"], lift["ground"], lift["num_frames"]), (True, "telemetry", 960))
        self.assertIsNone(lift["telemetry_config"])
        drop = parsed_entry("ancf_shell_drop")
        self.assertEqual(len(drop["shell_tires"]), 1)
        self.assertEqual((drop["substeps"], drop["nr_iters"], drop["pcg_iters"]), (6, 3, 200))
        self.assertIsNone(drop["shell_tires"][0]["E"])
        for name in ("ancf_rigid_mujoco_tires", "vehicle_ancf_tires"):
            selected = parsed_entry(name)
            expected = (
                "warthog_ancf_tire.usda" if name == "ancf_rigid_mujoco_tires" else "warthog_ancf_tire_simple.usda"
            )
            self.assertEqual(selected["tire_asset"], expected)
            self.assertTrue((ASSETS / selected["tire_asset"]).is_file())
            tire_path = str(ASSETS / "warthog_ancf_tire.usda")
            selected = parsed_entry(name, ("--tire-asset", tire_path))
            self.assertEqual(selected["tire_asset"], tire_path)
        override = parsed_entry("ancf_shell_drop", ("--substeps", "9", "--shell-tires", "[]"))
        self.assertEqual((override["substeps"], override["shell_tires"]), (9, "[]"))

    def test_lift_import_does_not_inherit_scene02_launch_defaults(self):
        rig = importlib.import_module("newton_solver_ancf.examples.example_ancf_rigid_mujoco_tires")
        raw = rig.Example.create_parser().parse_args([])
        self.assertEqual((raw.substeps, raw.nr_iters, raw.pcg_iters), (20, 16, None))

    def test_authored_assets_and_lift_material_load(self):
        from newton_solver_ancf.examples._vehicle_usd import VehicleUSD
        from newton_solver_ancf.examples.example_diffsim_ancf_tire_lift import _telemetry_setup

        for name in ("superjeep_tire.usda", "warthog_ancf_tire_simple.usda"):
            mesh, meta = load_ancf_tire_usd(str(ASSETS / name), device="cpu")
            self.assertEqual((mesh.n_nodes, mesh.n_elems), (240, 224))
            self.assertEqual(mesh.elem_h.shape, (224,))
            self.assertTrue(np.isfinite(mesh.node_x0.numpy()).all())
            self.assertGreater(meta.contact_kn, 0)
            selected = parsed_entry("ancf_shell_drop", ("--tire-asset", name))
            self.assertEqual(selected["kn"], meta.contact_kn)
        vehicle = VehicleUSD(str(ASSETS / "warthog_vehicle.usdc"))
        self.assertEqual(vehicle.kind, "rigid_hull")
        self.assertAlmostEqual(vehicle.wheel_mass, 12.900840675519746)
        self.assertAlmostEqual(vehicle.rigid_corner_mass - vehicle.wheel_mass, 50.0)
        _, options, material, _, bounds = _telemetry_setup()
        self.assertEqual((material["E"], material["nu"], material["rho"]), (50e6, 0.45, 700.0))
        self.assertEqual((material["alpha-damp"], material["thickness"], options["build_pressure"]), (0.15, None, 0.0))
        np.testing.assert_allclose(bounds, (2 * 6894.757, 4 * 6894.757))

        from newton_solver_ancf.examples._vehicle_config import load_telemetry_preset

        baseline = load_telemetry_preset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.json"
            baseline["args"]["shell-tires"][0]["E"] = 40e6
            path.write_text(json.dumps(baseline))
            _, _, changed_material, _, _ = _telemetry_setup(path)
            self.assertEqual(changed_material["E"], 40e6)
        self.assertEqual(load_telemetry_preset()["args"]["shell-tires"][0]["E"], 50e6)


if __name__ == "__main__":
    unittest.main()
