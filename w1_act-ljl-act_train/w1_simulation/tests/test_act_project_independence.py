from __future__ import annotations

import ast
from pathlib import Path

from w1_simulation.simulation.config import SOURCE_URDF
from w1_simulation.w1_profile import (
    DEFAULT_BRIDGE_SCRIPT,
    DEFAULT_POLICY_SCRIPT,
    DEFAULT_PROFILE,
    DEFAULT_WHOLE_SCRIPT,
    PROJECT_ROOT,
)


def _import_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_w1_simulation_has_no_legacy_or_rl_python_dependency() -> None:
    forbidden = {"w1_act_sim", "w1_sim", "w1_rl", "inference_codes"}
    imports = set().union(
        *(
            _import_roots(path)
            for path in PROJECT_ROOT.rglob("*.py")
            if "artifacts" not in path.parts and "version_records" not in path.parts
        )
    )
    assert imports.isdisjoint(forbidden)


def test_w1_simulation_runtime_and_configuration_are_project_local() -> None:
    for path in (DEFAULT_POLICY_SCRIPT, DEFAULT_BRIDGE_SCRIPT, DEFAULT_WHOLE_SCRIPT, DEFAULT_PROFILE.source):
        assert path.is_file()
        assert path.is_relative_to(PROJECT_ROOT)
    assert SOURCE_URDF.is_file()
    assert SOURCE_URDF.is_relative_to(PROJECT_ROOT)
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
