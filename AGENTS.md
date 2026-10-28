# Solver development

This is an independent differentiable Newton solver package. Keep all ANCF element and optimized shell mechanics together under `src/newton_solver_ancf/_src/solvers/ancf/`; use Newton's public API and preserve the installed Newton package.

Both ANCF paths use the HHT predictors, correctors, force-history weighting and tangent coefficients in `_src/solvers/implicit/hht.py`. Keep that math shared. Preserve the working tire implementation's sparse block solver, contact/pressure laws, element formulations, iteration budgets and stepping order. Element-specific state layouts and nonlinear convergence policies remain with their solvers. Global mass and tangent matrices and linear solves must be sparse; small element blocks and reduced rigid-body matrices may be dense. Keep rigid-deformable coupling under `_src/solvers/coupled/`, separate from element mechanics and HHT.

Keep each solver self-contained. Put runtime geometry loading under `_src/geometry/`
and any required HHT implementation under `_src/solvers/implicit/`; copy only code
that this solver uses. Store geometry-only example meshes and their reference
embeddings as USD assets under `examples/assets/`, not runtime mesh generators.
Keep materials, quadrature, interpolation and differentiation in solver code.
Baymax is an explicit exception: retain its resolution-controlled cage generation
in local `_src/geometry/` rather than replacing it with pre-baked cages.

Position-only Hex8 belongs to FEM. Validate shared HHT and sparse assembly changes against the existing forward states and implicit gradients.

The family package and `_src/solvers/__init__.py` explicitly export the existing public API; the package root re-exports it. Keep examples under `src/newton_solver_ancf/examples/`.

Keep element construction in each concrete solver’s `create_model` method and public interpolation in its `shape_functions` method. Avoid standalone construction/shape exports or forwarding aliases.

Examples consume only the solver API exported from the package `__init__.py`; do not import solver `_src` modules or access private solver members. Keep shared rendering helpers inside `examples`.

Document the implemented equations and gradient assumptions in `docs/method.md`. Map paper equations and pseudocode to actual functions in `docs/papers.md`. Keep `docs/usage.md` for example-specific prerequisites. Regenerate the root README with `python3 tools/build_readme.py`; do not edit generated text. Keep one actual demonstration video per discovered example, recorded with `tools/record_examples.py`. Preserve required copyright and license notices.

Use `uv`, Warp kernels, and `unittest`. Keep ordinary modules and direct calls; add abstractions only when an actual solver needs them. Preserve physical accuracy during optimization, and measure live simulation separately from rendering or replay.

Run `uv run --locked ruff check .`, `uv run --locked ruff format --check .`, and `uv run --locked python -m unittest discover -s tests -v`. GPU checks use `NEWTON_TEST_DEVICE=cuda:0`. Inspect the supported derivative branches in tests before adding gradient claims. Keep dependencies and lockfile synchronized.

Keep all runnable examples and their shared helpers directly in `examples/`; do not add solver or diffsim subpackages.
Example launchers discover runnable `examples/example_*.py` files. Give each a
module docstring and a `__main__` guard; use `HEADLESS = True` for terminal-only
examples. Do not add hard-coded example lists to Docker or Python launchers.
