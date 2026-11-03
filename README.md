# ARAMSAM: Agricultural Rapid Annotation Module based on Segment Anything Models 

ARAMSAM is an interactive annotation UI designed to accelerate image labeling for datasets in agriculture and beyond ✨. Built on top of Meta AI's Segment Anything Models (SAM 1, SAM 2, and SAM 3), it combines zero-shot segmentation, interactive prompt segmentation and mask propagation on image sequences with traditional annotation tools.

![Interactive Prompting](assets/ARAMSAM_interactive.gif)

## Prerequisites
- Python >= 3.11

The weights of SAM1 can be downloaded from `https://github.com/facebookresearch/segment-anything` and the weights of both SAM2.0 and SAM2.1 can be downloaded from `https://github.com/facebookresearch/sam2`. A CUDA GPU is recommended but simply using SAM 1 and the `vit-b` backbone also gives a sufficient user experience on a CPU-only system.


## Installation
Inside activated virtual environmment run:
```
pip install -r requirements.txt
```
To install SAM2 run the commands below. These steps are not necessary if you only want to use SAM1 (e.g. for CPU-only systems).:
```
cd segment-anything-21
pip install -e ./sam2
```

### Installation using Docker
Use VS Code Dev Containers Extension

or

build and run .devcontainer/Dockerfile. After starting the container run:
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
./install_sam_2.sh
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

## UI Workflow

For instructions on how you can use ARAMSAM UI see [Example Workflow](docs/example_workflow.md).

### SAM3

Use the environment with SAM3 installed and a local `sam3.pt` checkpoint. The
installed SAM3 implementation requires a working CUDA GPU (including during
model construction); SAM1 and SAM2 remain available through `gen: 1` or `gen: 2`.
See the [upstream SAM3 requirements](https://github.com/facebookresearch/sam3#installation).

Select SAM3 in the root `configs.yaml`:

```yaml
sam_configs:
  gen: 3
  model_ckpt_p: "sam3.pt"
  model_type: null
  confidence_threshold: 0.5
use_gpu: true
sam_background_embedding: true
```

The tokenizer vocabulary is found in the local SAM3 checkout or installed
package. Set `sam_configs.bpe_path` to override its location. Model loading uses
the local checkpoint and does not download weights. Settings → Set SAM Model
also accepts a checkpoint named `sam3.pt`.

Load an image, leave Auto Embed enabled, and open **SAM3 text prompt** in the
top-right corner. Enter a word or phrase and press Enter or **Find objects**.
Adjust **Minimum confidence** to control which detections are returned. Prior
prompts are available from the input dropdown. Use **Hide** or the top-right
toggle to collapse the panel.

Text results join the existing good-mask/bad-mask review workflow and retain
class labels and SAM3 origins when saved. Positive/negative interactive points,
undo, polygon/bounding-box drawing, deletion, image navigation, background
embedding, and propagation to the next image work with SAM3. Enable `do_amg` for
automatic proposals generated from a grid of interactive point prompts.

## Supporting Companies

This project is supported by:

- [NPZ Innovation GmbH](https://www.npz-innovation.de/)

## License
This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

## Citation
See the citation metadata: [`CITATION.cff`](./CITATION.cff)
