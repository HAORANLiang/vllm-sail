# SPDX-License-Identifier: Apache-2.0
"""Portable packing layout invariants and optional independent CPU reference."""

import ast
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def _portable():
    spec = importlib.util.spec_from_file_location(
        "portable", ROOT / "vllm_sail/native/portable.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("bits", [4, 8])
def test_weight_layout_permutation_is_bijective(bits):
    permutation = _portable()._weight_permutation(bits)
    assert len(permutation) == 1024
    assert sorted(permutation) == list(range(1024))


@pytest.mark.parametrize("bits", [4, 8])
def test_packed_weights_match_upstream_independent_reference(bits):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    source = Path(os.environ.get("VLLM_SOURCE_ROOT", ROOT.parent / "vllm"))
    reference = (
        source / "vllm/model_executor/layers/quantization/utils/marlin_utils_test.py"
    )
    if not reference.exists():
        pytest.skip(
            "set VLLM_SOURCE_ROOT to the pinned upstream checkout for reference packing"
        )
    functions = [
        node
        for node in ast.parse(reference.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("get_weight_perm", "marlin_permute_weights", "marlin_weights")
    ]
    namespace = {
        "torch": torch,
        "np": np,
        "GPTQ_MARLIN_TILE": 16,
        "get_pack_factor": lambda bits: 32 // bits,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(reference), "exec"),
        namespace,
    )
    generator = torch.Generator().manual_seed(17)
    portable = _portable()
    for k, n in ((32, 64), (64, 128), (256, 256)):
        weights = torch.randint(
            0, 1 << bits, (k, n), dtype=torch.int32, generator=generator
        )
        pack = 32 // bits
        gptq = torch.zeros((k // pack, n), dtype=torch.int32)
        for i in range(pack):
            gptq |= weights[i::pack] << (bits * i)
        actual = portable.gptq_marlin_repack(gptq, k, n, bits)
        expected = namespace["marlin_weights"](
            weights,
            k,
            n,
            bits,
            namespace["get_weight_perm"](bits),
        )
        assert torch.equal(actual, expected)
