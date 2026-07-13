# Repository guidance

This repository publishes a source-built vLLM runtime for ARM64 CUDA 13.0 and NVIDIA SM121.

- Preserve the exact upstream release commit and Apache-2.0 attribution.
- Do not change CUDA, Torch, FlashInfer, CUTLASS DSL, or NCCL pins without rebuilding and rerunning GPU/runtime qualification.
- Do not publish model weights, benchmark raw outputs, private host paths, credentials, telemetry, or internal debugging chronology.
- Do not accept Marlin, emulation, or metadata-only claims as native NVFP4 verification.
- Immutable GHCR tags must always identify the exact audited local image.
