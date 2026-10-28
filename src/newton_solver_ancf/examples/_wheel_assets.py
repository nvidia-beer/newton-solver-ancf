# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Select authored vehicle/tire/spindle pairs before creating a simulator."""

import sys
from pathlib import Path

from ._shell_assets import ASSET_DIR, asset_info, choose, discover_assets, prompt_substeps, show_assets, supplied


def resolve_asset(name):
    path = Path(name).expanduser()
    return path if path.is_absolute() else ASSET_DIR / path


def vehicle_info(path):
    from pxr import Usd

    path = Path(path)
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    root = stage.GetDefaultPrim()
    if not root:
        return None
    data = root.GetCustomData()
    if data.get("vehicleKind") not in ("rigid_hull", "articulated") or not all(
        key in data for key in ("defaultTireAsset", "spindleBodies", "axleJoints", "hubs")
    ):
        return None
    tires = {}
    for key in ("simpleTireAsset", "defaultTireAsset"):
        name = data.get(key)
        if not name:
            continue
        tire_path = path.parent / str(name)
        if not tire_path.is_file():
            continue
        info = asset_info(tire_path)
        if info is not None and info["spindle"]:
            # Keep packaged filenames concise; external vehicle bundles resolve
            # their declared tires relative to the vehicle, not our assets folder.
            argument = tire_path.name if tire_path.parent.resolve() == ASSET_DIR.resolve() else str(tire_path.resolve())
            tires[argument] = info
    return {"tires": tires}


def discover_vehicles(directory=ASSET_DIR):
    from pxr import Tf

    vehicles = {}
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".usd", ".usda", ".usdc"):
            continue
        try:
            info = vehicle_info(path)
        except (Tf.ErrorException, TypeError, ValueError):
            continue
        if info is not None and info["tires"]:
            vehicles[path.name] = info
    return vehicles


def configure_wheels(parser, *, allow_tire_spindle=False, preferred_tire=None):
    """Discover vehicle, tire and spindle choices from USD metadata."""
    parser.add_argument(
        "--list-assets", action="store_true", help="List vehicles and compatible spindle-equipped tires."
    )
    parser.allow_abbrev = False
    args = parser.parse_args()
    notes = getattr(args, "asset_notes", {})
    vehicles = discover_vehicles()
    if args.vehicle_asset == "":
        if not allow_tire_spindle:
            parser.error("The full-vehicle example requires a vehicle asset.")
        tires = {name: info for name, info in discover_assets().items() if info["spindle"]}
        vehicles = {"": {"tires": tires}}
    if args.vehicle_asset:
        try:
            info = vehicle_info(resolve_asset(args.vehicle_asset))
        except Exception as error:
            parser.error(f"Cannot read vehicle {args.vehicle_asset}: {error}")
        if info is None:
            parser.error(f"{args.vehicle_asset} has no supported vehicle metadata")
        vehicles = {args.vehicle_asset: info}
    if args.list_assets:
        for name, info in vehicles.items():
            print(f"Vehicle / spindle: {name}")
            show_assets(info["tires"])
            for tire in info["tires"]:
                if Path(tire).name in notes:
                    print(f"   {Path(tire).name}: {notes[Path(tire).name]}")
        raise SystemExit(0)
    if not vehicles:
        parser.error(f"No vehicle/tire pairs found in {ASSET_DIR}")
    selected_vehicle = args.vehicle_asset
    if args.tire_asset and selected_vehicle is None:
        matches = [
            name
            for name, info in vehicles.items()
            if any(resolve_asset(tire).resolve() == resolve_asset(args.tire_asset).resolve() for tire in info["tires"])
        ]
        if len(matches) != 1:
            parser.error("This tire has no unique vehicle pairing; supply --vehicle-asset explicitly.")
        selected_vehicle = matches[0]
    default = "warthog_vehicle.usdc" if "warthog_vehicle.usdc" in vehicles else next(iter(vehicles))
    interactive = sys.stdin.isatty() and args.viewer != "null" and not args.headless and not args.test
    if interactive and selected_vehicle is None:
        selected_vehicle = choose({name: name for name in vehicles}, default, "Vehicle / matching spindle:")
    selected_vehicle = selected_vehicle or default
    tires = vehicles[selected_vehicle]["tires"]
    if not tires and not args.tire_asset:
        parser.error("No declared tire was found beside this vehicle; supply --tire-asset explicitly.")
    default_tire = (
        "warthog_ancf_tire_simple.usda"
        if selected_vehicle == "" and "warthog_ancf_tire_simple.usda" in tires
        else next(iter(tires), None)
    )  # Authored simple tire first, otherwise full mesh.
    if preferred_tire in tires:
        default_tire = preferred_tire
    selected_tire = args.tire_asset or default_tire
    if interactive and args.tire_asset is None:
        labels = {
            name: f"{name}: {info['nodes']} nodes / {info['quads']} quads {notes.get(Path(name).name, '')}".rstrip()
            for name, info in tires.items()
        }
        selected_tire = choose(labels, selected_tire, "Compatible tire (with spindle):")
    try:
        info = asset_info(resolve_asset(selected_tire))
    except Exception as error:
        parser.error(f"Cannot read tire {selected_tire}: {error}")
    if info is None or not info["spindle"]:
        parser.error(f"{selected_tire} must contain an ANCF tire and /Tire/Spindle mass, inertia and mesh")
    if Path(selected_tire).name in notes:
        print(notes[Path(selected_tire).name])
    parser.set_defaults(vehicle_asset=selected_vehicle, tire_asset=selected_tire)
    # Some vehicles require a different default timestep. Keep those
    # numerical defaults in the example parser; explicit --substeps always wins.
    defaults = getattr(args, "vehicle_substeps", {})
    substeps = args.substeps if supplied("--substeps") else defaults.get(Path(selected_vehicle).name, args.substeps)
    parser.set_defaults(substeps=substeps)
    if interactive:
        prompt_substeps(parser, substeps)
    print(
        f"[ANCF] Vehicle/spindle: {selected_vehicle or '/Tire/Spindle'}; tire: {selected_tire}; substeps: {parser.parse_args().substeps}"
    )
