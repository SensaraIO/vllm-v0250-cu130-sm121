#!/usr/bin/env python3
"""Fail-closed runtime audit for r0b0tlab vLLM v0.25.0 SM121 image."""
from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import subprocess
import sys

EXPECTED = {
    "vllm": "0.25.0",
    "flashinfer-python": "0.6.13",
    "nvidia-cutlass-dsl": "4.5.2",
    "nvidia-nccl-cu13": "2.28.9",
}


def version(name: str) -> str:
    return md.version(name)


def main() -> int:
    import torch
    import vllm

    actual = {name: version(name) for name in EXPECTED}
    failures: list[str] = []
    for name, expected in EXPECTED.items():
        if actual[name] != expected:
            failures.append(f"{name}: expected {expected}, got {actual[name]}")

    if vllm.__version__ != "0.25.0":
        failures.append(f"vllm.__version__: expected 0.25.0, got {vllm.__version__}")
    if not torch.__version__.startswith("2.11.0+cu130"):
        failures.append(f"torch: expected 2.11.0+cu130, got {torch.__version__}")
    if torch.version.cuda != "13.0":
        failures.append(f"torch CUDA runtime: expected 13.0, got {torch.version.cuda}")

    nvcc = subprocess.run(
        ["/usr/local/cuda-13.0/bin/nvcc", "--version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    if "release 13.0" not in nvcc:
        failures.append("active CUDA compiler is not release 13.0")

    extensions = {}
    for module in ("vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch"):
        try:
            importlib.import_module(module)
            extensions[module] = "PASS"
        except Exception as exc:
            extensions[module] = f"FAIL: {type(exc).__name__}: {exc}"
            failures.append(f"extension import failed: {module}")

    gpu = None
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        gpu = {"name": torch.cuda.get_device_name(), "capability": capability}
        if capability != (12, 1):
            failures.append(f"GPU capability: expected (12, 1), got {capability}")
        x = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
        y = x @ x
        torch.cuda.synchronize()
        gpu["bf16_matmul_finite"] = bool(torch.isfinite(y).all().item())
        if not gpu["bf16_matmul_finite"]:
            failures.append("CUDA BF16 matmul produced non-finite output")

    report = {
        "status": "FAIL" if failures else "PASS",
        "vllm_api_version": vllm.__version__,
        "packages": actual,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "extensions": extensions,
        "gpu": gpu,
        "failures": failures,
        "provenance": "r0b0tlab-vllm-v0250-cu130-sm121",
    }
    print(json.dumps(report, indent=2, default=list))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
