#!/usr/bin/env bash
# Clone self_critique -> self_critique_olmoe and upgrade everything needed to
# evaluate `allenai/Olmo-3.1-7B-RL-Zero-Math` (architecture `Olmo3ForCausalLM`).
#
# Why:
#   * The self_critique env pins vllm==0.6.3 (Sep 2024), which predates
#     `Olmo3ForCausalLM` (added in vLLM via PR #24534, Sep 2025).
#   * Cloning also copies `verl`, which declares `vllm<=0.6.3` in its metadata.
#     Pip will refuse to upgrade vLLM until `verl` is removed from this clone.
#   * We don't want to touch self_critique because the EURUS / LIMR / STILL
#     workflows still rely on its pinned versions.
#   * `--clone` reuses the existing conda+pip cache via hardlinks so the copy
#     is fast and disk-cheap.
#   * On some RHEL/EL compute nodes, `/lib64/libstdc++.so.6` is older than the
#     one pyzmq / vLLM were built against (`GLIBCXX_3.4.30` missing). This script
#     prepends the target env's `lib/` to `LD_LIBRARY_PATH` for pip and verify.
#
# GPU count is not configured here (this script only prepares the conda env).
# For Olmo inference, set tensor parallelism in run_full_workflow_olmoe.sh
# (TENSOR_PARALLEL_SIZE, default 4) or pass the same via env when launching.
#
# Usage (from anywhere):
#   bash /scratch2/mjgwak/rl-data-contamination-mj/eval_scripts/OLMOE/setup_self_critique_olmoe.sh
#
# Idempotent: re-running after a successful upgrade is a no-op for pip beyond
# version checks, and the conda clone step will fail-fast with a clear message
# if the target env already exists (delete it manually if you want a fresh clone).

set -euo pipefail

SOURCE_ENV="${SOURCE_ENV:-self_critique}"
TARGET_ENV="${TARGET_ENV:-self_critique_olmoe}"

# Target vLLM line. 0.11.x is the first stable line guaranteed to carry the
# Olmo-3 model implementation (PR #24534 merged 2025-09-13). 0.10.x landed it
# at the very tail end so we pin a known-good lower bound and let pip pick the
# latest patch.
VLLM_MIN="${VLLM_MIN:-0.11.0}"

# Top-level packages we want fresh in the new env. pip resolves the dep graph
# (torch / transformers / huggingface_hub / etc.) from these.
PIP_UPGRADE_TOPLEVEL=(
    "vllm>=${VLLM_MIN}"
    "transformers"
    "huggingface_hub"
    "datasets"
    "accelerate"
    "tokenizers"
    "safetensors"
    "numpy"
    "pandas"
    "pyarrow"
    "scikit-learn"
    "tqdm"
)

log() { printf '\n[setup_self_critique_olmoe] %s\n' "$*"; }

log "Source env: ${SOURCE_ENV}"
log "Target env: ${TARGET_ENV}"
log "vLLM lower bound: ${VLLM_MIN}"

# Locate conda. Prefer the user's conda; fall back to base.
if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
    CONDA_BIN="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
else
    echo "ERROR: conda not on PATH. Activate your conda first." >&2
    exit 2
fi
log "Using conda at: ${CONDA_BIN}"

# 1) Clone, unless target already exists.
if "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${TARGET_ENV}"; then
    log "${TARGET_ENV} already exists; skipping clone."
else
    log "Cloning ${SOURCE_ENV} -> ${TARGET_ENV} (this can take a few minutes; pip cache is reused via hardlinks)..."
    "${CONDA_BIN}" create --name "${TARGET_ENV}" --clone "${SOURCE_ENV}" -y
fi

# 2) Run the rest inside the target env without needing `conda activate` in this shell.
TARGET_PYTHON="$("${CONDA_BIN}" run -n "${TARGET_ENV}" which python)"
TARGET_PREFIX="$("${TARGET_PYTHON}" -c "import sys; print(sys.prefix)")"
export LD_LIBRARY_PATH="${TARGET_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
log "Target python: ${TARGET_PYTHON}"
log "Target prefix: ${TARGET_PREFIX}"
log "LD_LIBRARY_PATH (prepended env lib): ${LD_LIBRARY_PATH%%:*} ..."
"${TARGET_PYTHON}" -c "import sys; print('python:', sys.version)"

# 3) Make pip well-behaved. Reuse the local pip cache if writable.
PIP="${TARGET_PYTHON} -m pip"

log "Pip version (before):"
${PIP} --version

log "Upgrading pip + build tooling first..."
${PIP} install --upgrade pip setuptools wheel

# 4) Drop verl if present — it pins vllm<=0.6.3 and prevents Olmo3-capable vLLM.
log "Uninstalling verl if present (it constrains vLLM to 0.6.x via package metadata)..."
${PIP} uninstall -y verl 2>/dev/null || true

# 5) Upgrade the top-level packages. We let pip resolve transitive deps so
#    torch / xformers / flash-attn / triton / transformers all converge on a
#    set consistent with the new vllm.
log "Upgrading top-level packages: ${PIP_UPGRADE_TOPLEVEL[*]}"
${PIP} install --upgrade "${PIP_UPGRADE_TOPLEVEL[@]}"

log "Installed vLLM (sanity check):"
${PIP} show vllm | awk -F': ' '/^Version:/{print "  vllm " $2}'

# 6) Drop pip flash-attn unless we reinstall a matching wheel below.
#    A copy from the old self_critique stack is often ABI-incompatible with the
#    new torch (undefined c10 symbols) and breaks EngineCore at import time.
log "Uninstalling pip package flash-attn if present (prevents torch ABI mismatch)..."
${PIP} uninstall -y flash-attn 2>/dev/null || true

# 7) Optional: flash-attn wheels must match torch/cuda; skip if no wheel exists.
log "Best-effort flash-attn install (wheel-only for flash-attn; skip on failure)..."
${PIP} install --upgrade "flash-attn" --only-binary flash-attn --no-build-isolation || \
    log "flash-attn upgrade skipped (no compatible wheel). vLLM will still run via xformers/SDPA."

# 8) Verify the install: vLLM version, Olmo3ForCausalLM registered.
log "Verifying install..."
"${TARGET_PYTHON}" - <<'PY'
import sys

import vllm
print(f"vllm version: {vllm.__version__}")

# Pre-1.x vLLM uses ModelRegistry.get_supported_archs() but the path moved
# across versions; try a few entry points.
archs = None
registry_exc = None
try:
    from vllm.model_executor.models import ModelRegistry
    archs = ModelRegistry.get_supported_archs()
except Exception as exc:
    registry_exc = exc
    try:
        from vllm.model_executor.models.registry import ModelRegistry as _MR
        archs = _MR().get_supported_archs()
        registry_exc = None
    except Exception as exc2:
        registry_exc = exc2

if archs is None:
    print(
        "FAIL: could not enumerate ModelRegistry archs "
        f"({type(registry_exc).__name__}: {registry_exc})."
    )
    if registry_exc is not None and "GLIBCXX" in str(registry_exc):
        print(
            "HINT: Host libstdc++ is too old for a dependency (often pyzmq). "
            "Use: export LD_LIBRARY_PATH=\"<conda_env_prefix>/lib:$LD_LIBRARY_PATH\" "
            "(this setup script already does that when you run it as written)."
        )
    sys.exit(1)

archs = sorted(archs)
print(f"registered model architectures: {len(archs)}")
target = "Olmo3ForCausalLM"
if target in archs:
    print(f"OK: {target} is registered.")
else:
    olmo = [a for a in archs if a.lower().startswith("olmo")]
    print(f"FAIL: {target} not in registry. Olmo* archs present: {olmo}")
    sys.exit(1)

import transformers, huggingface_hub, datasets
print(f"transformers: {transformers.__version__}")
print(f"huggingface_hub: {huggingface_hub.__version__}")
print(f"datasets: {datasets.__version__}")
PY

log "Done. Use the new env via: conda activate ${TARGET_ENV}"
log "On EL/RHEL nodes, also export before Python/vLLM jobs:"
log "  export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}\""
log "Then re-run: bash eval_scripts/OLMOE/run_full_workflow_olmoe.sh"
