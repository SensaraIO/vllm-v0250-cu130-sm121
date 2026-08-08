#!/usr/bin/env python3
"""Fail-closed runtime audit for the Sensara FlashInfer 0.6.14 overlay."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import inspect
import json
import subprocess
import sys

EXPECTED = {
    "vllm": "0.25.0+sensara.fi0614.1",
    "flashinfer-python": "0.6.14",
    "flashinfer-cubin": "0.6.14",
    "flashinfer-jit-cache": "0.6.14+cu130",
    "nvidia-cutlass-dsl": "4.5.2",
    "nvidia-nccl-cu13": "2.28.9",
}

REQUIRED_DSV4_PARAMETERS = {
    "swa_topk_lens",
    "extra_sparse_indices",
    "extra_sparse_topk_lens",
}


def version(name: str) -> str:
    return md.version(name)


def normalized_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gpu", action="store_true")
    arguments = parser.parse_args()

    import torch
    import vllm

    actual = {name: version(name) for name in EXPECTED}
    failures: list[str] = []
    dsv4_parameters: list[str] | None = None
    installed_versions: dict[str, list[str]] = {}
    for distribution in md.distributions():
        name = distribution.metadata.get("Name")
        if name is None:
            continue
        canonical = normalized_name(name)
        if canonical in EXPECTED:
            installed_versions.setdefault(canonical, []).append(distribution.version)
    for name, expected in EXPECTED.items():
        observed = sorted(installed_versions.get(name, []))
        if observed != [expected]:
            failures.append(
                f"{name}: expected exactly one {expected} distribution, got {observed}"
            )
    try:
        importlib.import_module("torchcodec")
        importlib.import_module("flashinfer")
    except Exception as exc:
        failures.append(f"runtime dependency import failed: {type(exc).__name__}: {exc}")
    for name, expected in EXPECTED.items():
        if actual[name] != expected:
            failures.append(f"{name}: expected {expected}, got {actual[name]}")

    if vllm.__version__ != EXPECTED["vllm"]:
        failures.append(
            f"vllm.__version__: expected {EXPECTED['vllm']}, got {vllm.__version__}"
        )
    if not torch.__version__.startswith("2.11.0+cu130"):
        failures.append(f"torch: expected 2.11.0+cu130, got {torch.__version__}")
    if torch.version.cuda != "13.0":
        failures.append(f"torch CUDA runtime: expected 13.0, got {torch.version.cuda}")

    try:
        from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4

        dsv4_parameters = sorted(
            inspect.signature(
                trtllm_batch_decode_sparse_mla_dsv4
            ).parameters
        )
        missing = REQUIRED_DSV4_PARAMETERS - set(dsv4_parameters)
        if missing:
            failures.append(
                "FlashInfer DSv4 sparse-MLA API missing: "
                + ", ".join(sorted(missing))
            )
    except Exception as exc:
        failures.append(
            "FlashInfer DSv4 sparse-MLA signature probe failed: "
            f"{type(exc).__name__}: {exc}"
        )

    nvcc = subprocess.run(
        ["/usr/local/cuda-13.0/bin/nvcc", "--version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    if "release 13.0" not in nvcc:
        failures.append("active CUDA compiler is not release 13.0")

    gpu = None
    extension_modules = (
        "vllm._C_stable_libtorch",
        "vllm._moe_C_stable_libtorch",
    )
    if arguments.no_gpu:
        extensions = {module: "SKIP: --no-gpu" for module in extension_modules}
    else:
        extensions = {}
        for module in extension_modules:
            try:
                importlib.import_module(module)
                extensions[module] = "PASS"
            except Exception as exc:
                extensions[module] = f"FAIL: {type(exc).__name__}: {exc}"
                failures.append(f"extension import failed: {module}")

        if not torch.cuda.is_available():
            failures.append("CUDA is unavailable for the required GPU audit")
        else:
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
        "flashinfer_dsv4_parameters": dsv4_parameters,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "extensions": extensions,
        "gpu": gpu,
        "failures": failures,
        "provenance": "sensara-vllm-v0250-fi0614-cu130-sm121",
    }
    print(json.dumps(report, indent=2, default=list))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
