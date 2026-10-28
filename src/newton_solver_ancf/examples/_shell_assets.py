# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Discover baked ANCF shells and select drop options before starting the viewer."""

import sys
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().with_name("assets")


def asset_info(path):
    """Read the shell schema and display metadata without creating a solver."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    prim = stage.GetPrimAtPath("/Tire/Mesh")
    required = ("primvars:D0", "primvars:material", "primvars:shellThickness")
    if not prim or not prim.IsA(UsdGeom.Mesh) or not all(prim.HasAttribute(key) for key in required):
        return None
    data = prim.GetCustomData()
    if not all(
        key in data for key in ("nCirc", "nBeadRows", "rOuter", "rInner", "width", "pressure", "materialFields")
    ):
        return None
    fields = ("C11", "C22", "C33", "C12", "C13", "C23", "G23", "G13", "G12", "rho", "alpha_damp")
    if tuple(data["materialFields"]) != fields:
        return None
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    faces = mesh.GetFaceVertexCountsAttr().Get()
    if not points or not faces or any(count != 4 for count in faces):
        return None
    if (
        len(prim.GetAttribute("primvars:D0").Get()) != len(points)
        or len(prim.GetAttribute("primvars:shellThickness").Get()) != len(faces)
        or len(prim.GetAttribute("primvars:material").Get()) != len(faces) * len(fields)
    ):
        return None
    spindle = stage.GetPrimAtPath("/Tire/Spindle")
    has_spindle = False
    if spindle and spindle.IsA(UsdGeom.Mesh) and int(data["nBeadRows"]) > 0:
        properties = spindle.GetCustomData()
        has_spindle = (
            float(properties.get("mass", 0.0)) > 0.0
            and len(properties.get("com", ())) == 3
            and len(properties.get("inertia", ())) == 9
            and bool(UsdGeom.Mesh(spindle).GetPointsAttr().Get())
            and bool(UsdGeom.Mesh(spindle).GetFaceVertexIndicesAttr().Get())
        )
    return {
        "nodes": len(points),
        "quads": len(faces),
        "pressure": float(data["pressure"]),
        "kn": float(data.get("contactKn", 2000.0)),
        "spindle": has_spindle,
    }


def discover_assets(directory=ASSET_DIR):
    """List compatible USD files, ignoring vehicle visuals and unrelated meshes."""
    from pxr import Tf

    assets = {}
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".usd", ".usda", ".usdc"):
            continue
        try:
            info = asset_info(path)
        except (Tf.ErrorException, TypeError, ValueError):
            continue
        if info is not None:
            assets[path.name] = info
    return assets


def show_assets(assets):
    for index, (name, info) in enumerate(assets.items(), 1):
        print(f"{index:2}) {name:<32} {info['nodes']:5} nodes / {info['quads']:4} quads")


def choose(labels, default, title):
    """Choose a discovered filename, preserving the default on Enter."""
    print(title)
    names = list(labels)
    for index, label in enumerate(labels.values(), 1):
        print(f"{index:2}) {label}")
    while True:
        choice = input(f"Select (number/name, Enter for {default}, q to quit): ").strip()
        if choice.lower() == "q":
            raise SystemExit(0)
        if not choice:
            return default
        if choice in labels:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print("Choose a listed number or filename.")


def supplied(option):
    return any(arg == option or arg.startswith(option + "=") for arg in sys.argv[1:])


def prompt_substeps(parser, default):
    if not supplied("--substeps"):
        while True:
            choice = input(f"Substeps per 60 Hz frame (Enter for {default}): ").strip()
            if not choice:
                break
            if choice.isdigit() and int(choice) > 0:
                parser.set_defaults(substeps=int(choice))
                break
            print("Substeps must be a positive integer.")


def configure_drop(parser):
    """Select a packaged shell and substeps; explicit CLI values take precedence."""
    parser.allow_abbrev = False
    args = parser.parse_args()
    assets = discover_assets()
    if args.list_assets:
        show_assets(assets)
        raise SystemExit(0)
    if not assets and args.tire_asset is None:
        parser.error(f"No baked ANCF shells found in {ASSET_DIR}")
    default = "minimal_ball.usda" if "minimal_ball.usda" in assets else next(iter(assets), None)
    selected = args.tire_asset or default
    interactive = sys.stdin.isatty() and args.viewer != "null" and not args.headless and not args.test
    if interactive and args.tire_asset is None:
        labels = {name: f"{name}: {info['nodes']} nodes / {info['quads']} quads" for name, info in assets.items()}
        selected = choose(labels, default, "ANCF shell asset:")
    path = Path(selected).expanduser()
    if not path.is_absolute():
        path = ASSET_DIR / path
    try:
        info = asset_info(path)
    except Exception as error:
        parser.error(f"Cannot read ANCF shell {path}: {error}")
    if info is None:
        parser.error(f"{path} does not contain the baked ANCF shell schema at /Tire/Mesh")
    parser.set_defaults(tire_asset=str(path))

    if args.kn is None and not supplied("--contact-penetration"):
        parser.set_defaults(kn=info["kn"])
    if interactive:
        prompt_substeps(parser, args.substeps)
