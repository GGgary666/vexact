#!/usr/bin/env python3
"""Standalone smoke test for `torch_memory_saver`, independent of vexact/verl.

Exercises the exact same code path as `vexact.utils.torch_memory_saver_adapter`
(hook_mode="torch", region/pause/resume) to quickly verify that the installed
`torch_memory_saver` build can locate its CUDA runtime dependency
(e.g. `libcudart.so.12` / `libcudart.so.13`) on this machine, without needing
to launch a full vexact/verl worker.

Usage:
    python3 scripts/check_torch_memory_saver.py
"""

import torch


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is not available in this environment"
    print(f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}")

    import torch_memory_saver

    tms = torch_memory_saver.torch_memory_saver
    tms.hook_mode = "torch"

    def used_gb() -> float:
        # `torch.cuda.memory_allocated()` only reflects PyTorch's own caching-allocator
        # bookkeeping, which does NOT change when torch_memory_saver unmaps the physical
        # backing during pause(). Query the CUDA driver directly instead (same thing
        # `nvidia-smi` reports) to see the real effect of pause/resume.
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1024**3

    print("[1/4] entering region + allocating tensor ...")
    with tms.region(tag="smoke_test", enable_cpu_backup=True):
        x = torch.full((1_000_000_000,), 1, dtype=torch.uint8, device="cuda")
    print(f"      device memory used: {used_gb():.2f} GB")

    print("[2/4] pause (should release physical GPU memory) ...")
    tms.pause(tag="smoke_test")
    print(f"      device memory used after pause: {used_gb():.2f} GB")

    print("[3/4] resume (should restore GPU memory) ...")
    tms.resume(tag="smoke_test")
    print(f"      device memory used after resume: {used_gb():.2f} GB")

    print("[4/4] validating restored tensor contents ...")
    assert torch.all(x == 1), "Tensor contents were not preserved across pause/resume"

    print("\ntorch_memory_saver OK: region/pause/resume all succeeded.")


if __name__ == "__main__":
    main()
