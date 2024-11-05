#!/bin/sh
mkdir -p segment-anything-21
cd segment-anything-21
[ ! -d "sam2" ] && echo "sam2 directory does not exist. Cloning from git." && git clone https://github.com/facebookresearch/sam2.git
pip install -e ./sam2
sh ./segment-anything-21/sam2/checkpoints/download_ckpts.sh