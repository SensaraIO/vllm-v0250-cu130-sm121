#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import types
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "sensara_image_audit_runtime",
    HERE / "audit_runtime.py",
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class FakeDistribution:
    def __init__(self, name: str, version: str) -> None:
        self.metadata = {"Name": name}
        self.version = version


class NoGpuCuda:
    def is_available(self) -> bool:
        return False

    def __getattr__(self, name: str):
        raise AssertionError(f"torch.cuda.{name} must not be called")


def fake_sparse_mla(
    *,
    swa_topk_lens=None,
    extra_sparse_indices=None,
    extra_sparse_topk_lens=None,
):
    del swa_topk_lens, extra_sparse_indices, extra_sparse_topk_lens


class AuditModeTests(unittest.TestCase):
    def run_audit(self, *arguments: str) -> tuple[int, dict, list[str]]:
        torch = types.ModuleType("torch")
        torch.__version__ = "2.11.0+cu130"
        torch.version = types.SimpleNamespace(cuda="13.0")
        torch.cuda = NoGpuCuda()

        vllm = types.ModuleType("vllm")
        vllm.__version__ = "0.25.0+sensara.fi0614.1"

        flashinfer = types.ModuleType("flashinfer")
        flashinfer.__path__ = []
        flashinfer_decode = types.ModuleType("flashinfer.decode")
        flashinfer_decode.trtllm_batch_decode_sparse_mla_dsv4 = fake_sparse_mla

        distributions = [
            FakeDistribution(name, version)
            for name, version in AUDIT.EXPECTED.items()
        ]
        imported: list[str] = []

        def import_module(name: str):
            imported.append(name)
            if name.startswith("vllm._"):
                return types.ModuleType(name)
            if name == "torchcodec":
                return types.ModuleType(name)
            if name == "flashinfer":
                return flashinfer
            raise AssertionError(f"unexpected import: {name}")

        completed = subprocess.CompletedProcess(
            ["nvcc", "--version"],
            0,
            stdout="Cuda compilation tools, release 13.0, V13.0.88\n",
        )
        output = io.StringIO()
        argv = ["audit_runtime.py", *arguments]
        modules = {
            "torch": torch,
            "vllm": vllm,
            "flashinfer": flashinfer,
            "flashinfer.decode": flashinfer_decode,
        }
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(AUDIT.md, "distributions", return_value=distributions),
            mock.patch.object(
                AUDIT,
                "version",
                side_effect=lambda name: AUDIT.EXPECTED[name],
            ),
            mock.patch.object(AUDIT.importlib, "import_module", side_effect=import_module),
            mock.patch.object(AUDIT.subprocess, "run", return_value=completed),
            contextlib.redirect_stdout(output),
        ):
            result = AUDIT.main()
        return result, json.loads(output.getvalue()), imported

    def test_no_gpu_skips_driver_extensions_and_cuda_queries(self) -> None:
        result, report, imported = self.run_audit("--no-gpu")
        self.assertEqual(result, 0)
        self.assertEqual(report["status"], "PASS")
        self.assertNotIn("vllm._C_stable_libtorch", imported)
        self.assertNotIn("vllm._moe_C_stable_libtorch", imported)
        self.assertEqual(
            report["extensions"],
            {
                "vllm._C_stable_libtorch": "SKIP: --no-gpu",
                "vllm._moe_C_stable_libtorch": "SKIP: --no-gpu",
            },
        )

    def test_gpu_mode_imports_extensions_and_requires_cuda(self) -> None:
        result, report, imported = self.run_audit()
        self.assertEqual(result, 1)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("vllm._C_stable_libtorch", imported)
        self.assertIn("vllm._moe_C_stable_libtorch", imported)
        self.assertIn(
            "CUDA is unavailable for the required GPU audit",
            report["failures"],
        )


if __name__ == "__main__":
    unittest.main()
