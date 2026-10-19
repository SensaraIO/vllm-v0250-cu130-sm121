#!/usr/bin/env bash
# Local CI: the release-file checks that must pass before a change lands (there is no hosted CI).
#
#   scripts/ci-local.sh [all]   compile the audit script, verify the release pins, run the public safety scan
#   scripts/ci-local.sh fast    the same checks (they take a second; there is no separate lint or typecheck)
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-all}

step() { printf '\n==> %s\n' "$*"; }

# The checks used Python 3.12; use it when installed.
PY=$(command -v python3.12 || command -v python3)

checks() {
  step "Compile audit script"
  "$PY" -m py_compile scripts/audit_runtime.py
  # py_compile leaves a __pycache__ next to the script; don't leave it in the tree.
  rm -rf scripts/__pycache__

  step "Verify release pins"
  grep -F '702f4814fe54fabff350d43cb753ae3e47c0c276' docker/Dockerfile
  grep -F 'cuda-toolkit-13-0' docker/Dockerfile
  grep -F 'm.version("vllm") == "0.25.0"' docker/Dockerfile
  grep -F 'flashinfer-python") == "0.6.13"' docker/Dockerfile
  grep -F 'nvidia-nccl-cu13") == "2.28.9"' docker/Dockerfile

  step "Public safety scan"
  # This file holds the scan pattern itself, so it is excluded.
  if grep -RInE '/home/|192\.168\.|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|GITHUB_TOKEN|HF_TOKEN|password=' . \
      --exclude-dir=.git --exclude=ci-local.sh; then
    echo "public safety scan: the matches above must not be published" >&2
    exit 1
  fi
}

case "$mode" in
  fast|all) checks ;;
  *) echo "usage: scripts/ci-local.sh [all|fast]" >&2; exit 2 ;;
esac
printf '\nci-local %s passed.\n' "$mode"
