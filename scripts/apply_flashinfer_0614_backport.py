#!/usr/bin/env python3
"""Apply the reviewed FlashInfer 0.6.14 backport to vLLM 702f4814."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


UPSTREAM_BASE_COMMIT = "702f4814fe54fabff350d43cb753ae3e47c0c276"
UPSTREAM_FIX_COMMIT = "ba47bb5be1bf8e263e978178f0ee453dbc9fa9b6"


@dataclass(frozen=True)
class PatchSpec:
    relative_path: str
    before_sha256: str
    after_sha256: str
    old: str
    new: str


PATCHES = (
    PatchSpec(
        relative_path="requirements/cuda.txt",
        before_sha256=(
            "d49e2cf1c4a7cacd7afef7b7019af336ad260cf28379b778209446053ecbb449"
        ),
        after_sha256=(
            "22abed27e57862b3eea0b50986b7dee6bf1ba8727bf68f3895a1d98085115324"
        ),
        old='''\
# FlashInfer should be updated together with the Dockerfile
flashinfer-python==0.6.13
flashinfer-cubin==0.6.13
''',
        new='''\
# FlashInfer should be updated together with the Dockerfile
# flashinfer-cubin is not on PyPI since 0.6.14; setup.py excludes it from
# install_requires so the published wheel does not carry an unresolvable pin
--extra-index-url https://flashinfer.ai/whl/
flashinfer-python==0.6.14
flashinfer-cubin==0.6.14
''',
    ),
    PatchSpec(
        relative_path="setup.py",
        before_sha256=(
            "731a9e7d0dc7de94f45150fd211bbfdab9043f9c5c96ed896d9d955c6661db4d"
        ),
        after_sha256=(
            "ac16aa6cbe82c4c6d329ee0b05f11b7d8b91faff65ef687f40993d57fc099b2c"
        ),
        old='''\
            if "vllm-flash-attn" in req and cuda_major != "12":
                # vllm-flash-attn is built only for CUDA 12.x.
                # Skip for other versions.
                continue
            if "nvidia-cutlass-dsl[cu13]" in req and cuda_major == "12":
''',
        new='''\
            if "vllm-flash-attn" in req and cuda_major != "12":
                # vllm-flash-attn is built only for CUDA 12.x.
                # Skip for other versions.
                continue
            if "flashinfer-cubin" in req:
                # Not on PyPI since 0.6.14 (only https://flashinfer.ai/whl), so
                # it cannot be a wheel dependency; install it explicitly.
                continue
            if "nvidia-cutlass-dsl[cu13]" in req and cuda_major == "12":
''',
    ),
    PatchSpec(
        relative_path="vllm/utils/jit_monitor.py",
        before_sha256=(
            "48d8e0f74cf15dc0a18ee660198e08919619e1f60cceefa23b51eb4693bb9034"
        ),
        after_sha256=(
            "b85c8db81c559e712ed814d7a08cbd3800e1c83b0a3ae30b525140096fc05ee9"
        ),
        old='''\
def _setup_cutedsl_jit_hook() -> None:
    """Wrap ``cutlass.cute.compile`` to warn on compilation."""
    global _cutedsl_hook_installed
    if _cutedsl_hook_installed:
        return

    try:
        import cutlass.cute as cute
    except Exception:
        logger.debug("CuTeDSL is not available; skipping CuTeDSL JIT monitor.")
        return

    original_compile = cute.compile

    @functools.wraps(original_compile)
    def _compile_with_monitor(*args, **kwargs):
        kernel = args[0] if args else kwargs.get("function")
        kernel_name = getattr(kernel, "__name__", None)
        if kernel_name is None:
            kernel_name = (
                kernel.__class__.__name__ if kernel is not None else "<unknown>"
            )
        _log_cutedsl_jit_compile(kernel_name)
        return original_compile(*args, **kwargs)

    cute.compile = _compile_with_monitor
    _cutedsl_hook_installed = True
''',
        new='''\
class _MonitoredCuteCompile:
    """Logs JIT compilations; a plain function would break ``cute.compile[opts]``."""

    def __init__(self, inner):
        self._inner = inner

    def __getitem__(self, options) -> "_MonitoredCuteCompile":
        return _MonitoredCuteCompile(self._inner[options])

    def __call__(self, *args, **kwargs):
        kernel = args[0] if args else kwargs.get("function")
        kernel_name = getattr(kernel, "__name__", None)
        if kernel_name is None:
            kernel_name = (
                kernel.__class__.__name__ if kernel is not None else "<unknown>"
            )
        _log_cutedsl_jit_compile(kernel_name)
        return self._inner(*args, **kwargs)


def _setup_cutedsl_jit_hook() -> None:
    """Wrap ``cutlass.cute.compile`` to warn on compilation."""
    global _cutedsl_hook_installed
    if _cutedsl_hook_installed:
        return

    try:
        import cutlass.cute as cute
    except Exception:
        logger.debug("CuTeDSL is not available; skipping CuTeDSL JIT monitor.")
        return

    cute.compile = _MonitoredCuteCompile(cute.compile)
    _cutedsl_hook_installed = True
''',
    ),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_bytes(source: bytes, spec: PatchSpec) -> bytes:
    if sha256(source) != spec.before_sha256:
        raise RuntimeError(f"{spec.relative_path} does not match the reviewed base")
    text = source.decode("utf-8")
    if text.count(spec.old) != 1:
        raise RuntimeError(f"{spec.relative_path} patch site is missing or ambiguous")
    result = text.replace(spec.old, spec.new, 1).encode("utf-8")
    if sha256(result) != spec.after_sha256:
        raise RuntimeError(f"{spec.relative_path} patched hash is not reviewed")
    return result


def replace_file(path: Path, data: bytes) -> None:
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    temporary_path: str | None = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        try:
            os.fchmod(descriptor, 0o644)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
            os.fsync(directory_fd)
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
    finally:
        os.close(directory_fd)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_flashinfer_0614_backport.py SOURCE_ROOT")
    source_root = Path(sys.argv[1]).resolve(strict=True)
    for spec in PATCHES:
        path = source_root / spec.relative_path
        result = patch_bytes(path.read_bytes(), spec)
        replace_file(path, result)
        if sha256(path.read_bytes()) != spec.after_sha256:
            raise RuntimeError(f"{spec.relative_path} failed final hash check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
