#!/usr/bin/env bash
# Environment for the Qwen3 arm of the scale-vs-depth experiments.
#
# Kept separate from `bengali-rq1` (transformers 4.50) on purpose: Qwen3 support
# landed in transformers 4.51, and the TigerLLM/TituLM adapters were trained
# under 4.50. Everything gets re-evaluated under THIS env so every number in the
# paper comes from one toolchain.
#
# Target GPU: RTX 5060 Ti (Blackwell, sm_120) -> needs a cu128 torch build.

set -euo pipefail

ENV_NAME=bengali-rq2
PY_VERSION=3.11

conda create -n "$ENV_NAME" python="$PY_VERSION" -y

# cu128 wheels: earlier CUDA builds do not carry sm_120 kernels.
conda run -n "$ENV_NAME" --no-capture-output \
  pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch>=2.7,<2.9"

# Generous timeout/retries: pypi.org intermittently times out on this
# connection, and a half-installed env is worse than a slow one.
conda run -n "$ENV_NAME" --no-capture-output \
  pip install --timeout 120 --retries 10 \
  "transformers>=4.55,<5" \
  "peft>=0.15" \
  "accelerate>=1.6" \
  "datasets>=3.0" \
  "sentencepiece" \
  "protobuf" \
  "safetensors" \
  "jupyter" \
  "ipywidgets"

conda run -n "$ENV_NAME" --no-capture-output python - <<'PY'
import torch, transformers, peft
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

print("transformers:", transformers.__version__)
print("peft        :", peft.__version__)
print("torch       :", torch.__version__)
print("cuda avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu         :", torch.cuda.get_device_name(0))
    print("capability  :", torch.cuda.get_device_capability(0))

for arch in ("qwen3", "llama", "gemma3_text"):
    print(f"{arch:<12} supported:", arch in CONFIG_MAPPING_NAMES)
PY

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
