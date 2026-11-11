# SPDX-License-Identifier: Apache-2.0
"""Check the installed PPU plugin and real PCCL/NCCL communication.

Run inside the SAIL container with two idle PPUs made visible::

    CUDA_VISIBLE_DEVICES=0,1 timeout 180s python -m pytest tests/e2e/test_pynccl.py -v -s

No symbols, libraries, communicators, or tensors are mocked. General plugins
are loaded through vLLM's lifecycle; a missing/broken plugin or library fails.
These checks cover single-node communication, not multi-node model inference.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

pytestmark = pytest.mark.ppu

# The current operator CI provisions one device; run this file manually on two.
PPU_CI_MIN_DEVICES = 2


def _load_runtime():
    import torch
    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins

    import vllm_sail
    from vllm_sail.patch.utils import PATCH_MARKER

    # Capture the class before installation, as pynccl.py does by value.
    load_general_plugins()
    from vllm.distributed.device_communicators import pynccl, pynccl_wrapper

    assert current_platform.is_ppu(), current_platform
    assert torch.cuda.is_available(), "No usable PPU in this container"
    assert pynccl.NCCLLibrary is NCCLLibrary is pynccl_wrapper.NCCLLibrary
    for target, implementation in (
        (
            "vllm.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.__init__",
            NCCLLibrary.__init__,
        ),
        (
            "vllm.distributed.device_communicators.pynccl.PyNcclCommunicator.suspend",
            pynccl.PyNcclCommunicator.suspend,
        ),
    ):
        assert target in getattr(implementation, PATCH_MARKER, {}), (
            f"PPU override was not installed: {target}"
        )
    print(f"vllm-sail source: {vllm_sail.__file__}", flush=True)
    print(f"NCCL wrapper source: {pynccl_wrapper.__file__}", flush=True)
    print(f"PPU loader source: {NCCLLibrary.__init__.__code__.co_filename}", flush=True)
    return torch, NCCLLibrary


@pytest.fixture(scope="module")
def runtime():
    return _load_runtime()


def test_nccl_library(runtime):
    _, library_cls = runtime
    from vllm import envs
    from vllm.utils.nccl import find_nccl_library

    assert not envs.VLLM_DISABLE_PYNCCL, "Unset VLLM_DISABLE_PYNCCL or set it to 0"
    selected = find_nccl_library()
    print(f"Selected NCCL-compatible library: {selected}")
    # Call directly so a loader exception remains visible instead of being
    # swallowed by PyNcclCommunicator and converted into disabled=True.
    library = library_cls(selected)
    print(f"NCCL-compatible library version: {library.ncclGetVersion()}")
    for symbol in ("ncclCommSuspend", "ncclCommResume", "ncclCommQueryProperties"):
        print(f"{symbol}: {'present' if library.has_symbol(symbol) else 'absent'}")
    maps = Path("/proc/self/maps")
    if maps.is_file():
        loaded = {
            line.split()[-1]
            for line in maps.read_text().splitlines()
            if "libnccl" in line or "libpccl" in line
        }
        print(f"Mapped NCCL/PCCL libraries: {sorted(loaded)}")


def _check_collectives(rank: int, rendezvous: str):
    import torch
    import torch.distributed as dist

    # Select the rank's device before plugin imports can create a cached stream.
    torch.cuda.set_device(rank)
    _load_runtime()
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    communicator = None
    try:
        # A non-TP name keeps optional TP all-reduce backends out of this test.
        communicator = CudaCommunicator(
            cpu_group=dist.group.WORLD,
            device=device,
            unique_name="pynccl_test",
        )
        pynccl = communicator.pynccl_comm
        assert pynccl is not None and pynccl.available and not pynccl.disabled

        if not all(
            pynccl.nccl.has_symbol(name)
            for name in ("ncclCommSuspend", "ncclCommResume")
        ):
            pynccl.suspend()
            assert not pynccl._suspended
            pynccl.resume()

        # Direct PyNccl all-reduce: a torch.distributed fallback cannot pass.
        value = torch.full((8,), rank + 1, dtype=torch.float32, device=device)
        reduced = pynccl.all_reduce(value)
        torch.cuda.synchronize()
        torch.testing.assert_close(reduced, torch.full_like(value, 3))

        value.fill_(17 if rank == 0 else -1)
        broadcast = communicator.broadcast(value, src=0)
        torch.cuda.synchronize()
        torch.testing.assert_close(broadcast, torch.full_like(value, 17))

        # This reaches the exact CudaCommunicator.all_gatherv assertion from #22.
        local = torch.full((rank + 1,), rank, dtype=torch.int32, device=device)
        gathered = communicator.all_gatherv(local, sizes=[1, 2])
        torch.cuda.synchronize()
        expected = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
        # Packed scale words must survive transport bit-for-bit, including
        # values that would be NaNs if interpreted numerically as BF16.
        words = torch.tensor([0, 1, 32768, 32641, 65535], dtype=torch.int32)
        for sizes in ([1, 1], [1, 2], [0, 2]):
            host = words.repeat(sizes[rank], 1)
            local = host.to(device=device, dtype=torch.uint16)
            gathered = communicator.all_gatherv(local, sizes=sizes)
            torch.cuda.synchronize()
            assert gathered.dtype == torch.uint16
            torch.testing.assert_close(
                gathered.cpu().to(torch.int32),
                words.repeat(sum(sizes), 1),
                rtol=0,
                atol=0,
            )
        local = words.to(device=device, dtype=torch.uint16)
        output = torch.empty((2 * words.numel(),), dtype=torch.uint16, device=device)
        pynccl.all_gather(output, local)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pynccl.all_gather(output, local)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.cpu().to(torch.int32), words.repeat(2), rtol=0, atol=0
        )
        print(f"rank {rank}: PyNccl enabled; all-reduce/broadcast/all-gatherv PASS")
        dist.barrier()
    finally:
        if communicator is not None:
            communicator.destroy()
        dist.destroy_process_group()


def test_pynccl_collectives(runtime, tmp_path):
    torch, _ = runtime
    assert torch.cuda.device_count() >= 2, "This test requires two visible PPUs"
    torch.multiprocessing.spawn(
        _check_collectives,
        args=(str(tmp_path / "pynccl-rendezvous"),),
        nprocs=2,
        join=True,
    )
