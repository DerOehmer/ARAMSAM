import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path

from src.run_sam import SamInference
from sam_annotator.mask_visualizations import (
    MaskData,
    MaskVisualization,
    MaskVisualizationData,
)


class AnnotationObject:
    def __init__(self, filepath: Path) -> None:
        self.filepath: str = str(filepath)
        self.img: np.ndarray = cv2.imread(self.filepath)

        if self.img.shape[2] == 4:
            print("Loaded image with 4 channels - ignoring last")
            self.img = self.img[:, :, :3]
        self.masks: list[MaskData] = []
        self.good_masks: list[MaskData] = []
        self.mask_decisions: np.ndarray = None

        self.mask_visualizations: MaskVisualizationData = MaskVisualizationData(
            img=self.img
        )

    def set_masks(self, maskobject: list[MaskData]):
        self.masks = maskobject
        self.mask_decisions = np.zeros((len(self.masks)), dtype=bool)

    def set_current_mask(self, mask_idx: int):
        self.mask_visualizations.mask = cv2.cvtColor(
            self.masks[mask_idx].mask, cv2.COLOR_GRAY2BGR
        )


class Annotator:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = SamInference(
            sam_checkpoint="sam_vit_b_01ec64.pth", model_type="vit_b", device=device
        )
        self.annotation: AnnotationObject = None
        self.mask_idx = 0

    def update_mask_idx(self, new_idx: int = 0):
        self.mask_idx = new_idx
        self.annotation.set_current_mask(self.mask_idx)

    def create_new_annotation(self, filepath: Path):
        self.annotation = AnnotationObject(filepath=filepath)

    def predict_with_sam(self):
        if self.annotation is None:
            raise ValueError("No annotation object found.")

        print(self.annotation.img.shape)
        self.sam.image_embedding(self.annotation.img)
        mask_objs, annotated_image = self.sam.custom_amg(roi_pts=False, n_points=100)
        self.annotation.mask_visualizations.masked_img = annotated_image
        self.annotation.set_masks(mask_objs)

        self.update_mask_idx()
        self.update_collections(self.annotation)
        self.preselect_mask()

    def good_mask(self):
        done = False
        annot = self.annotation
        annot.good_masks.append(annot.masks[self.mask_idx])
        annot.mask_decisions[self.mask_idx] = True
        self.mask_idx += 1

        self.update_collections(annot)
        if self.mask_idx >= len(annot.masks) - 1:
            done = True  # all masks have been labeled
        else:
            self.annotation.set_current_mask(self.mask_idx)
            self.preselect_mask()
        # TODO: instead of done return coordinates of center of next mask
        return done

    def bad_mask(self):
        done = False
        annot = self.annotation
        annot.mask_decisions[self.mask_idx] = False
        self.mask_idx += 1

        self.update_collections(annot)
        if self.mask_idx >= len(annot.masks) - 1:
            done = True  # all masks have been labeled
        else:
            self.annotation.set_current_mask(self.mask_idx)
            self.preselect_mask()
        # TODO: instead of done return coordinates of center of next mask
        return done

    def preselect_mask(self, max_overlap_ratio: float = 0.4):
        annot = self.annotation
        mask_obj: MaskData = annot.masks[self.mask_idx]
        mask = mask_obj.mask
        maskorigin = mask_obj.origin

        mask_coll_bin = (
            np.any(
                annot.mask_visualizations.mask_collection != [0, 0, 0], axis=-1
            ).astype(np.uint8)
            * 255
        )
        mask_overlap = cv2.bitwise_and(mask_coll_bin, mask)
        mask_size = np.count_nonzero(mask)
        mask_overlap_size = np.count_nonzero(mask_overlap)
        overlap_ratio = mask_overlap_size / mask_size

        if overlap_ratio > max_overlap_ratio:
            self.bad_mask()

        if maskorigin == "sam_tracking" and len(annot.mask_decisions) == len(
            annot.good_masks
        ):
            self.good_mask()

    def step_back(self):
        annot = self.annotation

        if self.mask_idx > 0:

            if annot.mask_decisions[-1]:
                annot.good_masks.pop()

            annot.mask_decisions[self.mask_idx] = False
            self.mask_idx -= 1
            self.update_collections(annot)

    def update_collections(self, annot: AnnotationObject):

        mask_vis = MaskVisualization(annot.img, annot.good_masks)
        masked_img = mask_vis.get_masked_img()
        mask_collection = mask_vis.get_mask_collection()
        if len(annot.masks) > self.mask_idx:
            mask_obj = annot.masks[self.mask_idx]
            cnt = mask_obj.contour
            maskinrgb = mask_vis.get_maskinrgb(mask_obj)
            masked_img_cnt = mask_vis.get_masked_img_cnt(cnt)
            mask_collection_cnt = mask_vis.get_mask_collection_cnt(cnt)
        else:
            maskinrgb = np.zeros_like(masked_img)
            masked_img_cnt = np.zeros_like(masked_img)
            mask_collection_cnt = np.zeros_like(masked_img)

        mvis_data: MaskVisualizationData = self.annotation.mask_visualizations
        mvis_data.maskinrgb = maskinrgb
        mvis_data.masked_img = masked_img
        mvis_data.mask_collection = mask_collection
        mvis_data.masked_img_cnt = masked_img_cnt
        mvis_data.mask_collection_cnt = mask_collection_cnt

        self.annotation.good_masks = mask_vis.mask_objs
