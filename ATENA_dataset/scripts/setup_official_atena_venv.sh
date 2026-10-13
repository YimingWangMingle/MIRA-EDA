#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../ATENA-A-EDA/atena-basic"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.7 >/dev/null 2>&1; then
    PYTHON_BIN="python3.7"
  elif command -v python3.8 >/dev/null 2>&1; then
    PYTHON_BIN="python3.8"
  else
    PYTHON_BIN="python3"
  fi
fi

echo "Using Python for official ATENA: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

"${PYTHON_BIN}" -m venv .venv-official-atena
source .venv-official-atena/bin/activate

# Official ATENA pins old packages such as gym==0.18.0. New setuptools/pip
# versions reject their legacy setup metadata, so keep the build stack old.
python -m pip install --upgrade "pip==21.3.1" "setuptools==57.5.0" "wheel==0.37.1"
# The official stack mixes TensorFlow 2.5 and Chainer 6 era packages whose
# metadata conflicts under pip's modern resolver. Legacy resolver matches the
# historical installation behavior expected by the project.
pip install \
  "typing_extensions==3.7.4.3" \
  "protobuf==3.20.3" \
  "absl-py==0.15.0" \
  "grpcio==1.34.1" \
  "six==1.15.0"
pip install --use-deprecated=legacy-resolver -r requirements.txt

echo ""
echo "Official ATENA venv is ready."
echo "Activate it with:"
echo "  source ATENA_dataset/ATENA-A-EDA/atena-basic/.venv-official-atena/bin/activate"
