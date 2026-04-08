#!/bin/bash
set -euo pipefail

# Setup Tar-7B inference environment
# Usage: bash setup_tar_env.sh
# Requires: conda

ENV_NAME="tar"

echo "[1/4] Creating conda env '${ENV_NAME}' (python 3.10) ..."
conda create -n ${ENV_NAME} python=3.10 -y

PIP="$(conda run -n ${ENV_NAME} which pip)"
echo "Using pip: ${PIP}"

echo "[2/4] Installing PyTorch 2.7.1 + cu128 ..."
${PIP} install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

echo "[3/4] Installing dependencies ..."
${PIP} install \
  transformers==4.50.0 \
  accelerate==1.13.0 \
  einops==0.8.2 \
  huggingface_hub==0.36.2 \
  easydict==1.13 \
  datasets==4.8.4 \
  scikit-learn==1.7.2 \
  pillow==12.1.1

echo "[4/4] Cloning Tar repo ..."
if [ ! -d "Tar" ]; then
  git clone https://github.com/csuhan/Tar.git
else
  echo "  Tar/ already exists, skipping clone"
fi

# Apply weights_only=False patches for torch 2.7+
echo "[patch] Fixing torch.load weights_only for Tar ..."
cd Tar
sed -i "s/torch\.load(ckpt, map_location='cpu')/torch.load(ckpt, map_location='cpu', weights_only=False)/g" tok/ta_tok.py tok/ar_dtok/vqvae.py
sed -i "s/torch\.load(ckpt, map_location=lambda storage, loc: storage)/torch.load(ckpt, map_location=lambda storage, loc: storage, weights_only=False)/g" tok/ar_dtok/ar_model.py
cd ..

echo ""
echo "========================================="
echo "  Setup complete!"
echo "========================================="
echo ""
echo "Activate:  conda activate ${ENV_NAME}"
echo ""
echo "Test run:"
echo "  CUDA_VISIBLE_DEVICES=0 python tar_run_inputs_single_load.py \\"
echo "    --input_json ../v3_orig_inputs.json \\"
echo "    --out_root ../outputs/tar_v3_orig \\"
echo "    --model_path csuhan/Tar-7B-v0.1 \\"
echo "    --device cuda:0 --batch_size 50 --flat"
echo ""
echo "With BOI injection (for image tasks):"
echo "  CUDA_VISIBLE_DEVICES=0 python tar_run_inputs_single_load.py \\"
echo "    --input_json ../v3_orig_inputs.json \\"
echo "    --out_root ../outputs/tar_v3_orig_inject_boi \\"
echo "    --model_path csuhan/Tar-7B-v0.1 \\"
echo "    --device cuda:0 --batch_size 50 --flat \\"
echo "    --inject-boi --inject-scale 0"
