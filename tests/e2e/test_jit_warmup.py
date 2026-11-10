# SPDX-License-Identifier: Apache-2.0
"""Real registry/Triton regression for preflight filtering and shared pools."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from threading import Barrier, get_ident
from time import monotonic_ns
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")

pytestmark = [pytest.mark.ppu, pytest.mark.cap80]
_compiling_owner = ContextVar("warmup_regression_owner", default=None)


@triton.jit
def copy_kernel(X, Y, BLOCK: tl.constexpr, TAG: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(Y + offsets, tl.load(X + offsets))


@triton.jit
def broken_kernel(X, Y, BLOCK: tl.constexpr, TAG: tl.constexpr):
    tl.static_assert(BLOCK == 999, "warmup-regression-sentinel")
    offsets = tl.arange(0, BLOCK)
    tl.store(Y + offsets, tl.load(X + offsets))


@pytest.fixture
def runtime(monkeypatch):
    from vllm.platforms import current_platform

    import vllm_sail

    vllm_sail.register_out_of_tree()
    assert current_platform.is_ppu()
    assert torch.cuda.get_device_capability() == (8, 0)
    monkeypatch.setenv("VLLM_TRITON_JIT_WARMUP_NUM_THREADS", "2")


def make_registry(kernel, cases, *, config=None):
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.model_executor.warmup.jit_warmup_triton_helper import (
        TritonWarmupTensor,
        VllmTritonJitKernel,
    )

    config = object() if config is None else config
    tag = monotonic_ns()  # Force real compilation despite earlier JIT caches.
    compiled = []

    class Owner(VllmTritonJitKernel):
        def get_warmup_keys(self, *, vllm_config):
            assert vllm_config is config
            return list(cases)

        def warmup_inputs(self, key):
            return dict(
                X=TritonWarmupTensor(cases[key]),
                Y=TritonWarmupTensor(torch.bfloat16),
                BLOCK=16,
                TAG=tag + key,
                grid=(1,),
            )

        def compile(self, key):
            compiled.append(key)
            token = _compiling_owner.set(self)
            try:
                self.kernel.warmup(**self.warmup_inputs(key))
            finally:
                _compiling_owner.reset(token)

    owner = Owner()
    owner.kernel = kernel
    registry = JitWarmupRegistry(config)
    with registry.activate():
        owner.register_warmup()
    worker = SimpleNamespace(model_runner=SimpleNamespace(jit_warmup_registry=registry))
    return worker, registry, owner, compiled


def test_distinct_owners_share_one_real_async_pool(runtime, monkeypatch):
    from vllm.model_executor.warmup import jit_warmup_triton_helper as helper

    from vllm_sail.patch.enhancement.kernel_warmup import (
        _exclude_sm80_fp8_registrations,
    )

    pools, submitted, threads = [], [], []
    barrier = Barrier(2)

    class ObservedExecutor(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pools.append(self)

        def submit(self, fn, *args, **kwargs):
            owner = _compiling_owner.get()
            submitted.append(owner)

            def compile_in_parallel():
                threads.append(get_ident())
                barrier.wait(timeout=60)
                return fn(*args, **kwargs)

            return super().submit(compile_in_parallel)

    monkeypatch.setattr(helper, "ThreadPoolExecutor", ObservedExecutor)
    config = object()
    worker, registry, first, _ = make_registry(
        copy_kernel, {0: torch.bfloat16}, config=config
    )
    _, _, second, _ = make_registry(copy_kernel, {0: torch.bfloat16}, config=config)
    with registry.activate():
        second.register_warmup(vllm_config=config)
    _exclude_sm80_fp8_registrations(worker)
    registry.warmup()
    assert len(pools) == 1
    assert {id(owner) for owner in submitted} == {id(first), id(second)}
    assert len(submitted) == 2
    assert len(set(threads)) == 2
    assert all(
        actual is original
        for actual, original in zip(
            registry._registrations, (first, second), strict=True
        )
    )


def test_mixed_signature_keys_skip_fp8_before_submission(runtime):
    worker, registry, owner, compiled = make_registry(
        copy_kernel, {0: torch.float8_e4m3fn, 1: torch.bfloat16}
    )
    from vllm_sail.patch.enhancement.kernel_warmup import (
        _exclude_sm80_fp8_registrations,
    )

    original_compile_many = owner.compile_many
    for _ in range(2):
        _exclude_sm80_fp8_registrations(worker)
        registry.warmup()
        with registry.activate():
            owner.register_warmup()
        assert len(registry) == 1
        assert next(iter(registry._registrations)) is owner
    assert compiled == [1, 1]
    assert owner.compile_many == original_compile_many

    x = torch.arange(16, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(x)
    copy_kernel[(1,)](x, output, BLOCK=16, TAG=monotonic_ns())
    torch.testing.assert_close(output, x)


def test_fp8_key_cannot_hide_other_async_compiler_errors(runtime):
    from triton.compiler.errors import CompilationError

    _, registry, _, compiled = make_registry(
        broken_kernel, {0: torch.float8_e4m3fn, 1: torch.bfloat16}
    )
    with pytest.raises(CompilationError, match="warmup-regression-sentinel"):
        registry.warmup()
    assert compiled == [1]
    x = torch.zeros(16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(CompilationError, match="warmup-regression-sentinel"):
        broken_kernel[(1,)](x, x, BLOCK=16, TAG=monotonic_ns())


def test_real_packseq_mixed_registration(runtime):
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.v1.attention.ops.common import PackSeqTritonKernel, pack_seq_triton

    owner = PackSeqTritonKernel()
    registry = JitWarmupRegistry(None)
    with registry.activate():
        owner.register_warmup(dtype=torch.float8_e4m3fn, pad_value=-float("inf"))
        owner.register_warmup(dtype=torch.uint8, pad_value=0)
    registry.warmup()
    assert next(iter(registry._registrations)) is owner
    values = torch.arange(8, device="cuda", dtype=torch.uint8).reshape(4, 2)
    lengths = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    packed = pack_seq_triton(values, lengths, pad_value=0)
    torch.testing.assert_close(packed[0, 0], values[0])
    torch.testing.assert_close(packed[1], values[1:])
    torch.testing.assert_close(packed[0, 1:], torch.zeros_like(packed[0, 1:]))
