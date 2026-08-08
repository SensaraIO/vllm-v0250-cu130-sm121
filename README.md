# vLLM v0.25.0 · CUDA 13.0 · ARM64 · SM121

Reproducible source-built vLLM runtime for NVIDIA GB10 / DGX Spark.

## Sensara FlashInfer 0.6.14 source rebuild

This fork rebuilds the exact vLLM 0.25.0 source at
`702f4814fe54fabff350d43cb753ae3e47c0c276` with a narrow reviewed backport.
The resulting distribution is versioned `0.25.0+sensara.fi0614.1`; it requires
the matched FlashInfer 0.6.14 Python, cubin, and CUDA-13.0 JIT-cache
distributions and includes only the CuTeDSL subscription compatibility change
from upstream vLLM commit
`ba47bb5be1bf8e263e978178f0ee453dbc9fa9b6`.

All three wheels are installed from exact URLs with reviewed SHA-256 hashes.
The build fails unless all three patched upstream files have their exact base
and reviewed output digests, the installed vLLM metadata requires FlashInfer
0.6.14, every package has exactly one installed distribution, and the resulting
`trtllm_batch_decode_sparse_mla_dsv4` callable exposes the three parameters
required by the pinned DeepSeek V4 SM121 caller.

```bash
docker build --platform linux/arm64 --progress=plain \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -f docker/Dockerfile \
  -t vllm-v0250-cu130-sm121:fi0614 .

docker run --rm --gpus all --entrypoint /opt/r0b0tlab/audit_runtime.py \
  vllm-v0250-cu130-sm121:fi0614
```

The GPU audit is necessary but not sufficient: the two-rank model boot and
OpenAI-compatible conformance gate remain mandatory before promotion.

This repository builds the upstream [vLLM v0.25.0](https://github.com/vllm-project/vllm/releases/tag/v0.25.0) release at commit `702f4814fe54fabff350d43cb753ae3e47c0c276`, targeting ARM64 and compute capability 12.1 with the CUDA 13.0 toolkit used by Torch `2.11.0+cu130`.

## Prior image (do not deploy for DeepSeek V4)

```bash
docker pull ghcr.io/r0b0tlab/vllm-v0250-cu130-sm121:v0.25.0-cu130-sm121-arm64-702f4814-r2
```

Registry digest: `sha256:a13c9964937f398b66d4a7e4fb8f80be8a60327052ca50bc8fbc2ce40c36beae`.

That predecessor contains FlashInfer 0.6.13 and is incompatible with the pinned
DeepSeek V4 caller. It is listed only for provenance; this branch must be built,
audited, and published under a new immutable digest before deployment. Do not
use the moving `latest` tag.

## Why a source build?

The upstream v0.25.0 release does not publish an ARM64 CUDA-13.0 image. Installing a generic wheel can replace SM121-native extensions or compile against a CUDA toolkit newer than Torch's runtime. This build fails closed unless:

- vLLM reports the rebuilt distribution version `0.25.0+sensara.fi0614.1` and the independent compiler audit reports CUDA 13.0;
- Torch reports `2.11.0+cu130`;
- active `nvcc` reports CUDA 13.0;
- FlashInfer Python and cubin distributions are exactly 0.6.14;
- the CUDA-13.0 `flashinfer-jit-cache` distribution is exactly `0.6.14+cu130`, with no duplicate inherited FlashInfer distributions;
- TorchCodec can load its FFmpeg shared-library dependencies;
- CUTLASS DSL is 4.5.2;
- NCCL is 2.28.9;
- compute capability is 12.1 at runtime.

## Build

```bash
docker build --progress=plain \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
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
