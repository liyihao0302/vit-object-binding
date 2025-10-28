#!/usr/bin/env bash
set -e

echo "[1/7] Cloning external libs..."
mkdir -p libs
cd libs
[ ! -d ADE20K ] && git clone https://github.com/CSAILVision/ADE20K.git
[ ! -d dinov2 ] && git clone https://github.com/facebookresearch/dinov2.git
cd ..

echo "[2/7] Installing PyTorch (CUDA 11.8)..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

echo "[3/7] Installing core requirements..."
pip install -r requirements.txt
pip install -r requirements_dino.txt

echo "[4/7] Installing segmentation deps..."
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
pip install mmsegmentation==0.30.0

echo "[5/7] Installing xformers..."
pip install --force-reinstall xformers==0.0.23.post1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

echo "[6/7] Installing dinov2 in editable mode..."
cd libs/dinov2 && pip install -e . && cd ../..

echo "[7/7] Numpy compatibility..."
pip install 'numpy<2' --force-reinstall

echo "✅ Setup complete!"
