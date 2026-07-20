#!/bin/bash
# =============================================================================
# Install the NF4 text encoder (standalone, no gated Llama 3 dependency)
#
# Usage:
#   bash scripts/install_nf4_encoder.sh
#
# Downloads Aero-Ex/KIMODO-Meta3_llm2vec_NF4 (~4.7 GB) via hf-mirror.com.
# Also installs bitsandbytes which is required for NF4 loading.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "=== Installing NF4 Text Encoder ==="

# Install bitsandbytes via Tsinghua mirror (PyPI blocked without proxy)
if ! python3 -c "import bitsandbytes" 2>/dev/null; then
    echo "Installing bitsandbytes..."
    pip install bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo "bitsandbytes already installed"
fi

# Download NF4 model
echo ""
echo "Downloading Aero-Ex/KIMODO-Meta3_llm2vec_NF4 (4.7 GB)..."
python3 -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
path = snapshot_download('Aero-Ex/KIMODO-Meta3_llm2vec_NF4', max_workers=4)
print('Done:', path)
"

echo ""
echo "=== NF4 encoder installed ==="
echo "Verify: TEXT_ENCODER=llm2vec-nf4 kimodo_gen 'test.' --model kimodo-g1-rp -d 1.0 -o /tmp/test_nf4"
