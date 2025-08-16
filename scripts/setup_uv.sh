#!/usr/bin/env bash
set -euo pipefail

# Create local venv with uv and install requirements

if ! command -v uv &>/dev/null; then
  echo "Installing uv (pipx recommended)" >&2
  python3 -m pip install --user uv || pip3 install --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv
uv venv .venv
source .venv/bin/activate

# Upgrade pip & install wheel
python -m pip install -U pip wheel

# Install requirements
uv pip install -r requirements.txt

# Install editable subpackages
( cd data_utils && uv pip install -e . )
( cd modeling_utils && uv pip install -e . )

# HuggingFace login hint
echo "If needed, run: huggingface-cli login" >&2

echo "Done. Activate with: source .venv/bin/activate" >&2
