# ARAMSAM: Agricultural Rapid Annotation Module based on Segment Anything Models 

ARAMSAM is an interactive annotation UI designed to accelerate image labeling for datasets in agriculture and beyond ✨. Built on top of Meta AI's Segment Anything Models (SAM 1 🧠 and SAM 2 🧠), it combines zero-shot segmentation, interactive prompt segmentation and mask propagation on image sequences with traditional annotation tools.

![Interactive Prompting](assets/ARAMSAM_interactive.gif)

## Prerequisites
- Python >= 3.10

To reproduce the encoder experiment all SAM1, SAM2.0 and SAM2.1 weights are required. For reproducing the AMG experiment and the user experiment, only `sam_vit_h_4b8939.pth` and `sam2.1_hiera_small.pt` are required.
The weights of SAM1 can be downloaded from `https://github.com/facebookresearch/segment-anything` and the weights of both SAM2.0 and SAM2.1 can be downloaded from `https://github.com/facebookresearch/sam2`.
All experiments were run on a RTX 3090 with 24GB Vram. Less VRAM might cause the scripts to not work properly.

## Installation
Inside activated virtual environmment run:
```
pip install -r requirements.txt
cd segment-anything-21
pip install -e ./sam2
```
## Run the experiments
- Encoder experiment: Run `Backbone_evaluation_experiment.py` for each dataset (modify dataset in main function)
- AMG experiment: Run `Amg_evaluation_experiment.py` for each encoder (modify sam_gen, weights_path and config in main function)
- User experiment: Run `COMPLETE_EXPERIMENT.py` per user. `COMPLETE_TUTORIAL.py` is giving instructions.

