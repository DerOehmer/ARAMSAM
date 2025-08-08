#!/bin/sh
cd segment-anything-21
pip install -e ./sam2
cd ..
sh ./segment-anything-21/sam2/checkpoints/download_ckpts.sh