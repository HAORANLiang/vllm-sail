# SPDX-License-Identifier: Apache-2.0
"""Read selected definitions without importing upstream or device libraries.

Executing an extracted body does not exercise its module imports, decorators
or patch installation. Feature tests needing those guarantees must test them
separately. The caller supplies only the globals needed by the selected body.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from vllm_sail import envs

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(envs.__file__).resolve().parents[1]


def source_root(source: str | None, *, required: bool = False) -> Path:
    if source is None:
        if required:
            raise pytest.UsageError(
                "--require-upstream-source requires VLLM_SOURCE_ROOT"
            )
        pytest.skip("set VLLM_SOURCE_ROOT to run upstream source checks")
    root = Path(source).expanduser().resolve()
    if not (root / "vllm" / "__init__.py").is_file():
        raise pytest.UsageError(
            f"VLLM_SOURCE_ROOT must contain vllm/__init__.py: {root}"
        )
    return root


def definition(path: Path, name: str):
    node = ast.parse(path.read_text())
    for part in name.split("."):
        matches = [
            child
            for child in node.body
            if isinstance(child, ast.ClassDef | ast.FunctionDef) and child.name == part
        ]
        assert len(matches) == 1, f"Missing or ambiguous definition {name!r} in {path}"
        node = matches[0]
    return copy.deepcopy(node)


def function(relative, name, namespace):
    path = (PACKAGE_ROOT if str(relative).startswith("vllm_sail/") else ROOT) / relative
    node = definition(path, name)
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[node.name]


def assert_accepts_upstream_keywords(
    local, local_name, upstream_root, upstream, upstream_name
):
    """Check keyword names and variadics; this is not full call compatibility."""
    replacement = definition(
        PACKAGE_ROOT / f"vllm_sail/patch/enhancement/{local}.py", local_name
    )
    original = definition(upstream_root / f"vllm/{upstream}.py", upstream_name)
    accepted = {a.arg for a in replacement.args.args + replacement.args.kwonlyargs}
    offered = {a.arg for a in original.args.args + original.args.kwonlyargs}
    context = f"{local}:{local_name} against {upstream}:{upstream_name}"
    assert replacement.args.kwarg is not None or offered <= accepted, (
        f"{context}: missing upstream keywords {offered - accepted}"
    )
    if original.args.kwarg is not None:
        assert replacement.args.kwarg is not None, f"{context}: missing **kwargs"
    if original.args.vararg is not None:
        assert replacement.args.vararg is not None, f"{context}: missing *args"
