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
To install SAM3 run the commands below
```
cd sam3
pip install -e .
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

The startup welcome window provides optional controls for every `configs.yaml`
setting, grouped into General, SAM, Image Tiling, Saving, and Classes tabs.
Choose the red **Quit** button at the bottom left to exit without saving edits.
The blue **Load Image** and **Load Folder** buttons save changed settings to the
same `configs.yaml` file used at startup, then open the appropriate input chooser.
Unchanged settings leave the file untouched. If input selection is canceled,
the welcome window stays open; changes have already been saved. Invalid settings
or write errors stay visible in the welcome window so they can be corrected.

The selected output directory is created when loading begins. Leave it blank to
choose an output directory later. Models load in a worker after input selection.
Settings can also be edited directly in `configs.yaml` before starting ARAMSAM.

Choose **General → Mask propagation method → Bounding-box prompts** in the
welcome window to use the same propagation method with SAM1, SAM2, or SAM3.
Alternatively, set `mask_propagation_method: bounding_box` and
`mask_propagation: true` in `configs.yaml`. Accepted masks supply tight bounding
boxes as prompts for fresh segmentation of the next image; object IDs, classes,
and colors are retained. Boxes keep their relative positions if image sizes
differ. This method assumes objects remain near the same locations between
images. The default, `tracking`, retains the existing model tracking / panorama
alignment behavior.

## UI Workflow

YOLO exports copy the original input file into `output/images` without resizing
or recompressing it. With tiling enabled, that file is the generated tile at its
actual dimensions: full tiles use `img_tiles.tile_size`, including sizes above
1024, and border tiles may be smaller when `allow_smaller_border_tiles: true`.
Annotation/model images remain limited to 1024 pixels on their longest side;
control images can use that downsampled size. YOLO coordinates are normalized
against the annotation image, retaining their relative positions on the original
image or tile.

Choose **Instance contours with ID circles** in **Settings → Change Layout** to
show mask contours with translucent circles and white instance IDs. The circle
is placed inside the mask, including for masks with a hole at their centroid.

The welcome window's **General → Annotation layout** provides four panel
selectors. Uncheck **Use default layout** to customize them. In the **Saving**
tab, **Control image visualization** independently selects the image saved under
`control_images` for YOLO exports (or `annotations.jpg` for native exports).
Control images contain accepted objects and exclude interactive previews.

The equivalent `configs.yaml` settings are:

```yaml
# Panel order: top left, bottom left, top right, bottom right.
layout:
  - img
  - img_sam_preview
  - instance_contours
  - mask_collection_cnt
save_data:
  control_image_visualization: instance_contours
```

Use `layout: null` to keep the default layout. Control-image choices are `auto`,
`img`, `masked_img`, `bbox_img`, `mask_collection`, and `instance_contours`.
The default `auto` uses filled masks for segmentation exports and bounding boxes
for box exports. Welcome changes are saved when choosing Load Image or Load Folder;
layout changes made through Settings apply to the current session.

For instructions on how you can use ARAMSAM UI see [Example Workflow](docs/example_workflow.md).

Open **Settings** to toggle **Auto Save**, **Auto Embed**, or **Mask Propagation**.
Mask propagation carries accepted objects into the next image. Set the top-level
`mask_propagation: false` in `configs.yaml` to disable it at startup (default:
`true`). The Settings checkbox changes it for the current session. Turning it
off clears generated masks in the unvisited next image and stops further
propagation, while keeping annotations on the current image and masks loaded
from disk.

Propagated objects retain a stable identity across images and saved reloads, so
Back → edit → Next replaces the previous prediction even if its outline changes.
Native exports store this identity in `annotations.json`. YOLO exports keep the
standard label text unchanged and also write `metadata/<image-stem>.json`; retain
that folder when continuing annotation. Sidecars are checked against the label
contents so externally edited labels are imported without stale identities.
Older files remain readable, with local import identities and geometric matching
as a compatibility fallback when their cross-image identities are unavailable.

Deleting an accepted mask excludes its identity from subsequent propagation and
removes matching copies already prepared for the next image. Exclusions carry
forward through saved images, including when no accepted masks remain, and late
worker results cannot re-add a deleted object. SAM2's tracking state is reset
before using the remaining masks and cleared when the prompt set is empty;
image embeddings remain cached. Native folders store exclusions in
`propagation.json`; YOLO sidecars include them in `metadata/<image-stem>.json`.
Explicitly accepting a deleted mask again allows it to propagate again.

Longer propagation, image preparation, and annotation-saving operations show a
loading window after a short delay. Backend work runs in worker threads while
the UI continues painting the dialog and its busy indicator. Output-directory
selection stays on the UI thread; subsequent disk checks, image loading, tiling,
and saving run in workers.

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

Text results are automatically accepted using the selected class, with the same
overlap filtering as propagated masks. Use Back to undo an acceptance or Delete
to remove a mask. Saved results retain class labels and SAM3 origins. Positive/negative interactive points,
undo, polygon/bounding-box drawing, deletion, image navigation, background
embedding, and propagation to the next image work with SAM3. Enable `do_amg` for
automatic proposals generated from a grid of interactive point prompts.

## Supporting Companies

This project is supported by:

- [NPZ Innovation GmbH](https://www.npz-innovation.de/)

## License

ARAMSAM's original application code is licensed under the [GNU Affero General Public License v3.0](LICENSE).

The project integrates third-party components governed by their own licenses, including:

- **SAM 1 and SAM 2/2.1:** Apache License 2.0
- **SAM 3:** Meta SAM License
- **Ultralytics:** AGPL-3.0 or commercial licensing
- **PyQt6:** GPL-3.0 or commercial licensing

Third-party components retain their respective copyright notices and license terms. ARAMSAM's AGPL-3.0 license does not replace or override these terms.

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for further information.

Users intending to redistribute ARAMSAM, integrate it into proprietary software, or distribute Docker images containing third-party components should review the applicable licenses before deployment.

## Citation
See the citation metadata: [`CITATION.cff`](./CITATION.cff)
