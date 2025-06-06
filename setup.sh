#!/usr/bin/env bash
# create and activate the env
conda create -y -n ReFaceNet python=3.10
conda activate ReFaceNet

# core DL stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3-D & vision deps
pip install opencv-python mediapipe trimesh open3d imageio tqdm Pillow numpy


# DECA
git clone https://github.com/YadiraF/DECA.git
pushd DECA
pip install -r requirements.txt
python setup.py install
# download pretrained checkpoints (~350 MB)
mkdir -p data; \
wget -P data https://github.com/YadiraF/DECA/releases/download/v1.0.0/coeff_v20210930.zip
unzip -q data/coeff_v20210930.zip -d data
popd
