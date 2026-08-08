# vLLM 0.25.0 + reviewed FlashInfer 0.6.14 backport / CUDA 13 / ARM64 / SM121
# Base release: https://github.com/vllm-project/vllm/releases/tag/v0.25.0
ARG BASE=ghcr.io/r0b0tlab/vllm-laguna-m1-nvfp4-sm121@sha256:6e2dfa4c48ac73965890a864dc1510017f42981e743b3f2834b4b189ad7712a8

FROM ${BASE} AS cuda130base
RUN apt-get update && apt-get install -y --no-install-recommends \
      cuda-toolkit-13-0 cmake git ninja-build && \
    update-alternatives --set cuda /usr/local/cuda-13.0 && \
    test "$(/usr/local/cuda-13.0/bin/nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/\1/p' | head -n1)" = "13.0" && \
    rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:${PATH} \
    CUDACXX=/usr/local/cuda-13.0/bin/nvcc
RUN python3 -m pip uninstall -y \
      vllm flashinfer-python flashinfer-cubin flashinfer-jit-cache && \
    python3 - <<'PY'
import importlib.metadata as metadata

blocked = {
    "vllm",
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
}
remaining = sorted(
    (distribution.metadata["Name"], distribution.version)
    for distribution in metadata.distributions()
    if distribution.metadata["Name"].lower().replace("_", "-") in blocked
)
assert not remaining, remaining
PY

FROM cuda130base AS builder
ARG VLLM_COMMIT=702f4814fe54fabff350d43cb753ae3e47c0c276
RUN git clone https://github.com/vllm-project/vllm.git /workspace/vllm-src && \
    cd /workspace/vllm-src && git checkout --detach ${VLLM_COMMIT} && \
    test "$(git rev-parse HEAD)" = "${VLLM_COMMIT}"
COPY scripts/apply_flashinfer_0614_backport.py /tmp/apply-vllm-backport.py
RUN python3 /tmp/apply-vllm-backport.py /workspace/vllm-src && \
    rm -f /tmp/apply-vllm-backport.py
WORKDIR /workspace/vllm-src
ENV TORCH_CUDA_ARCH_LIST="12.1" \
    VLLM_TARGET_DEVICE=cuda \
    VLLM_USE_PRECOMPILED=0 \
    VLLM_VERSION_OVERRIDE=0.25.0+sensara.fi0614.1 \
    MAX_JOBS=6 \
    NVCC_THREADS=2 \
    FLASHINFER_NVCC_THREADS=2
RUN python3 -m pip install --no-cache-dir pybind11
# Resolve v0.25 runtime dependencies first. They may install CUDA 13.2 Python
# compiler packages, so explicitly restore the system CUDA 13.0 toolkit before
# compiling extensions for Torch 2.11.0+cu130.
COPY docker/flashinfer-0.6.14-requirements.txt /tmp/flashinfer-requirements.txt
RUN python3 -m pip install \
      --no-cache-dir \
      --no-deps \
      --require-hashes \
      -r /tmp/flashinfer-requirements.txt && \
    rm -f /tmp/flashinfer-requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements/cuda.txt
RUN python3 -m pip install --no-cache-dir -r requirements/build/cuda.txt
ENV CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:${PATH} \
    CUDACXX=/usr/local/cuda-13.0/bin/nvcc
RUN test "$(nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/\1/p' | head -n1)" = "13.0"
RUN python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -v .
RUN python3 - <<'PY'
import importlib.metadata as m
import inspect
import torch, vllm
from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4

expected_vllm = "0.25.0+sensara.fi0614.1"
assert vllm.__version__ == expected_vllm, vllm.__version__
assert m.version("vllm") == expected_vllm, m.version("vllm")
assert torch.__version__.startswith("2.11.0+cu130"), torch.__version__
assert m.version("flashinfer-python") == "0.6.14"
assert m.version("flashinfer-cubin") == "0.6.14"
assert m.version("flashinfer-jit-cache") == "0.6.14+cu130"
required = {"swa_topk_lens", "extra_sparse_indices", "extra_sparse_topk_lens"}
actual = set(inspect.signature(trtllm_batch_decode_sparse_mla_dsv4).parameters)
assert not required - actual, sorted(required - actual)
assert m.version("nvidia-cutlass-dsl") == "4.5.2"
assert m.version("nvidia-nccl-cu13") == "2.28.9"
print("builder audit passed", vllm.__version__, torch.__version__)
PY

FROM cuda130base AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.12/dist-packages/ /usr/local/lib/python3.12/dist-packages/
COPY --from=builder /usr/local/bin/vllm /usr/local/bin/vllm
RUN python3 -c "import flashinfer; print('flashinfer import passed', flashinfer.__version__)"
ENV MAX_JOBS=6 \
    NVCC_THREADS=2 \
    FLASHINFER_NVCC_THREADS=2 \
    TORCH_CUDA_ARCH_LIST="12.1" \
    CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:${PATH} \
    CUDACXX=/usr/local/cuda-13.0/bin/nvcc
ARG SOURCE_REVISION
ARG VLLM_COMMIT=702f4814fe54fabff350d43cb753ae3e47c0c276
ARG VLLM_FIX_COMMIT=ba47bb5be1bf8e263e978178f0ee453dbc9fa9b6
RUN test -n "${SOURCE_REVISION}"
LABEL org.opencontainers.image.title="Sensara vLLM 0.25.0 FlashInfer 0.6.14 CUDA13 ARM64 SM121" \
      org.opencontainers.image.version="0.25.0+sensara.fi0614.1" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.source="https://github.com/SensaraIO/vllm-v0250-cu130-sm121" \
      org.opencontainers.image.url="https://github.com/SensaraIO/vllm-v0250-cu130-sm121" \
      org.opencontainers.image.vendor="SensaraIO" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.sensara.upstream.vllm.source="https://github.com/vllm-project/vllm" \
      io.sensara.upstream.vllm.revision="${VLLM_COMMIT}" \
      io.sensara.upstream.vllm.fix="${VLLM_FIX_COMMIT}" \
      io.sensara.flashinfer.version="0.6.14" \
      io.sensara.cuda.version="13.0" \
      io.sensara.compute.capability="12.1"
RUN python3 -c "import torchcodec; from torchcodec.decoders import VideoDecoder; print('torchcodec import passed')"
COPY scripts/audit_runtime.py /opt/r0b0tlab/audit_runtime.py
RUN chmod 0755 /opt/r0b0tlab/audit_runtime.py && \
    /opt/r0b0tlab/audit_runtime.py --no-gpu
ENTRYPOINT ["vllm"]
