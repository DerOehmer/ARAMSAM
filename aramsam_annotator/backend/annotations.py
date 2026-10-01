"""Annotation records shared by editing, inference, storage, and rendering."""
from dataclasses import dataclass
from os.path import basename
from pathlib import Path
import cv2
import numpy as np


class MaskIdHandler:
    def __init__(self):
        self._id = 0
        self.ids: list[int] = []

    def get_id(self):
        current_id = self._id
        self.ids.append(current_id)
        self._id += 1
        return current_id


@dataclass
class MaskData:
    mid: int
    origin: str
    mask: np.ndarray = None
    bbox: tuple = None  # xyxy
    time_stamp: int = None  # in deciseconds (1/10th of a second)
    center: tuple = None
    color_idx: int = None
    contour: np.ndarray = None
    class_id: int = None


@dataclass
class MaskVisualizationData:
    img: np.ndarray = None
    img_sam_preview: np.ndarray = None
    mask: np.ndarray = None  # mask of current object of interest
    maskinrgb: np.ndarray = None
    masked_img: np.ndarray = None
    mask_collection: np.ndarray = None
    masked_img_cnt: np.ndarray = None
    mask_collection_cnt: np.ndarray = None
    bbox: tuple = None  # xyxy of current object of interest
    bbox_img: np.ndarray = None
    bbox_img_cnt: np.ndarray = None


class AnnotationObject:
    def __init__(self, filepath: Path) -> None:
        self.filepath: str = str(filepath)
        self.img: np.ndarray = self._load_img(self.filepath)
        self.img_name = basename(filepath)
        self.masks: list[MaskData] = []
        self.good_masks: list[MaskData] = []
        self.mask_decisions: list[bool] = []

        self.features = None
        self.original_size = None
        self.input_size = None

        from aramsam_annotator.mask_visualizations import MaskVisualization

        self.mask_visualizer = MaskVisualization()
        self.mask_visualizations: MaskVisualizationData = MaskVisualizationData(
            img=self.img
        )
        self.preview_mask = None
        self.proposals_ready = False

    def _load_img(self, filepath: Path | str) -> np.ndarray:
        img = cv2.imread(str(filepath))
        if img is None:
            raise OSError(f"Could not read image: {filepath}")
        if len(img.shape) == 2:
            print("Loaded image is grayscale - converting to BGR")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            print("Loaded image with 4 channels - ignoring last")
            img = img[:, :, :3]
        height, width = img.shape[:2]
        if max(height, width) > 1024:
            scale = 1024 / max(height, width)
            img = cv2.resize(img, (max(1, round(width * scale)),
                                   max(1, round(height * scale))),
                             interpolation=cv2.INTER_AREA)
        return img

    def set_current_mask(self, mask_idx: int):
        self.mask_visualizations.mask = None
        self.mask_visualizations.bbox = None
        if not 0 <= mask_idx < len(self.masks):
            return
        if self.masks[mask_idx].mask is not None:
            self.mask_visualizations.mask = cv2.cvtColor(
                self.masks[mask_idx].mask, cv2.COLOR_GRAY2BGR
            )
        elif self.masks[mask_idx].bbox is not None:
            self.mask_visualizations.bbox = self.masks[mask_idx].bbox

    def add_masks(self, masks, decision=False):
        self.masks.extend(masks)
        self.mask_decisions.extend([decision for _ in range(len(masks))])

    def set_sam_parameters(self, features, original_size, input_size):
        self.features = features
        self.original_size = original_size
        self.input_size = input_size

    def get_sam_parameters(self):
        return self.features, self.original_size, self.input_size

    def load_masks_from_dir(self, masks_dir: Path, mid_handler: MaskIdHandler):
        from aramsam_annotator.backend.storage import ORIGIN_CODES

        masks_dir = Path(masks_dir)
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {masks_dir}")
        origins = {code: origin for origin, code in ORIGIN_CODES.items()}
        loaded = []
        for mask_file in sorted(masks_dir.glob("*.png")):
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise OSError(f"Could not read mask: {mask_file}")
            parts = mask_file.stem.split('_')
            origin = origins.get(parts[1], 'Polygon_drawing') if len(parts) > 1 else 'Polygon_drawing'
            timestamp = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            loaded.append(MaskData(mid=mid_handler.get_id(), mask=mask,
                                   origin=origin, time_stamp=timestamp))
        self.good_masks = loaded
        self.masks = list(loaded)
        self.mask_decisions = [True] * len(loaded)
