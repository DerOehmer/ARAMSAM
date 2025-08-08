# ARAMSAM: Agricultural Rapid Annotation Module based on Segment Anything Models 

ARAMSAM is an interactive annotation UI designed to accelerate image labeling for datasets in agriculture and beyond ✨. Built on top of Meta AI's Segment Anything Models (SAM 1 🧠 and SAM 2 🧠), it combines zero-shot segmentation, interactive prompt segmentation and mask propagation on image sequences with traditional annotation tools.

![Interactive Prompting](assets/ARAMSAM_interactive.gif)

## Prerequisites
- Python >= 3.10

The weights of SAM1 can be downloaded from `https://github.com/facebookresearch/segment-anything` and the weights of both SAM2.0 and SAM2.1 can be downloaded from `https://github.com/facebookresearch/sam2`. A CUDA GPU is recommended but using the `vit-b` backbone also gives a sufficient user experience on a CPU-only system.


## Installation
Inside activated virtual environmment run:
```
pip install -r requirements.txt
cd segment-anything-21
pip install -e ./sam2
```

## Run ARAMSAM
```
python -m aramsam_annotator.main
```

Modify `aramsam_annotator/configs.py' to: 
- Load different SAM encoders
- Enable/disable the automatic mask generator
- Define classes (currently only supported with saving bboxes in yolo format)
- Load a YOLO network for bounding box proposals

A more user friendly way to modify the settings is coming in the future.
