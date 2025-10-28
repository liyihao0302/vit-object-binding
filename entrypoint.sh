mkdir libs
cd libs
git clone https://github.com/CSAILVision/ADE20K.git
git clone https://github.com/facebookresearch/dinov2.git
cd ..

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -r requirements_dino.txt # replace the original requirements.txt in libs/dinov2/
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
pip install mmsegmentation==0.30.0

pip install --force-reinstall xformers==0.0.23.post1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
cd libs/dinov2/
pip install -e .
cd ../../
pip install "numpy<2" --force-reinstall
