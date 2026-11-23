#!/usr/bin/env bash
# Create a project-local environment on a Jetson running JetPack 6.2. Makes no system changes.
# Usage: scripts/setup_jetson.sh [WORK_DIR]
set -euo pipefail

UPSTREAM_URL="https://github.com/apple-aiml-research/ml-depth-pro-2"
UPSTREAM_COMMIT="aa389cbb0b74341664ed663a0966c859905cd6c0"
JETSON_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/"
SYSTEM_TENSORRT="/usr/lib/python3.10/dist-packages/tensorrt"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${1:-$REPO_DIR/work}"
mkdir -p "$WORK_DIR"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"

if [ ! -d "$SYSTEM_TENSORRT" ]; then
  echo "TensorRT Python bindings not found at $SYSTEM_TENSORRT. Install them through JetPack." >&2
  exit 1
fi

if [ ! -x "$WORK_DIR/venv/bin/python" ]; then
  python3.10 -m venv "$WORK_DIR/venv"
fi
PY="$WORK_DIR/venv/bin/python"
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
ln -sfn "$SYSTEM_TENSORRT" "$SITE/tensorrt"

"$PY" -m pip install --quiet --upgrade pip
# The index is used alone here: PyPI's aarch64 CUDA wheels do not include Orin GPU code.
"$PY" -m pip install --quiet --index-url "$JETSON_INDEX" "torch==2.8.0" "torchvision==0.23.0"
"$PY" -m pip install --quiet -r "$REPO_DIR/requirements-jetson.txt"

if [ ! -d "$WORK_DIR/ml-depth-pro-2/.git" ]; then
  git clone --quiet "$UPSTREAM_URL" "$WORK_DIR/ml-depth-pro-2"
fi
git -C "$WORK_DIR/ml-depth-pro-2" checkout --quiet "$UPSTREAM_COMMIT"

"$PY" - <<'EOF'
import numpy, onnx, onnxruntime, tensorrt, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("tensorrt", tensorrt.__version__, "numpy", numpy.__version__)
print("onnx", onnx.__version__, "onnxruntime", onnxruntime.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this environment")
print("gpu test", float((torch.ones(4, device="cuda") * 2).sum()))
EOF
echo "Environment ready: $WORK_DIR"
