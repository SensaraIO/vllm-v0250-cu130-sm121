# vLLM v0.25.0 · CUDA 13.0 · ARM64 · SM121

Reproducible source-built vLLM runtime for NVIDIA GB10 / DGX Spark.

This repository builds the upstream [vLLM v0.25.0](https://github.com/vllm-project/vllm/releases/tag/v0.25.0) release at commit `702f4814fe54fabff350d43cb753ae3e47c0c276`, targeting ARM64 and compute capability 12.1 with the CUDA 13.0 toolkit used by Torch `2.11.0+cu130`.

## Image

```bash
docker pull ghcr.io/r0b0tlab/vllm-v0250-cu130-sm121:v0.25.0-cu130-sm121-arm64-702f4814-r2
```

Registry digest: `sha256:a13c9964937f398b66d4a7e4fb8f80be8a60327052ca50bc8fbc2ce40c36beae`.

The moving `latest` tag currently resolves to this immutable revision. The image passes dependency, extension, TorchCodec/FFmpeg, CLI-import, and SM121 GPU audits; model-level serving remains a deployment-specific gate.

## Why a source build?

The upstream v0.25.0 release does not publish an ARM64 CUDA-13.0 image. Installing a generic wheel can replace SM121-native extensions or compile against a CUDA toolkit newer than Torch's runtime. This build fails closed unless:

- vLLM reports upstream release version `0.25.0` and the independent compiler audit reports CUDA 13.0;
- Torch reports `2.11.0+cu130`;
- active `nvcc` reports CUDA 13.0;
- FlashInfer is 0.6.13;
- the incompatible inherited `flashinfer-jit-cache` 0.6.12 package is absent;
- TorchCodec can load its FFmpeg shared-library dependencies;
- CUTLASS DSL is 4.5.2;
- NCCL is 2.28.9;
- compute capability is 12.1 at runtime.

## Build

```bash
docker build --progress=plain \
  -f docker/Dockerfile \
  -t vllm-v0250-cu130-sm121:local .
```

GB10-safe build limits are embedded: `MAX_JOBS=6`, `NVCC_THREADS=2`, and `FLASHINFER_NVCC_THREADS=2`.

## Audit

```bash
docker run --rm --gpus all \
  --entrypoint python3 \
  vllm-v0250-cu130-sm121:local \
  /opt/r0b0tlab/audit_runtime.py
```

## Checks

Run the release-file checks locally before pushing; there is no hosted CI (GitHub Actions is off):

```bash
scripts/ci-local.sh        # compile the audit script, verify the release pins, public safety scan
scripts/ci-local.sh fast   # the same checks
```

## Distributed GB10 notes

v0.25.0 includes blocking CUDA events for V1/V2 async-copy paths, avoiding driver-lock busy polling under tensor-parallel collective contention ([vLLM #47081](https://github.com/vllm-project/vllm/pull/47081)).

For a validated multi-node DGX Spark RoCE topology:

- keep rendezvous/Gloo socket coordination on management Ethernet;
- select rank-local RoCE HCAs with `NCCL_IB_HCA`;
- explicitly expose `/dev/infiniband/rdma_cm` and the required `uverbs` devices to unprivileged containers;
- pin the validated RoCEv2 GID with `NCCL_IB_GID_INDEX` rather than relying on automatic selection when a host exposes several GID entries;
- use `NCCL_IB_SUBNET_AWARE_ROUTING=1` and `NCCL_IB_MERGE_NICS=0` where validated;
- do not force MNNVL without NVSwitch multicast;
- do not enable NCCL symmetric memory on SM121.

These are topology gates, not universal defaults. Validate routes, MTU, HCA names, and collectives on the target cluster.

## Scope

This image contains the runtime only. It contains no model weights, benchmark data, private paths, telemetry, callbacks, or tracking.

## Upstream credit and license

vLLM is developed by the vLLM project and distributed under Apache-2.0. This repository preserves upstream licensing and source attribution. NVIDIA CUDA components remain subject to their respective NVIDIA licenses.

Maintained by [r0b0tlab](https://github.com/r0b0tlab).
