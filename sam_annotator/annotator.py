import numpy as np
import torch
import cv2
from pathlib import Path


from src.run_sam import SamInference
from sam_annotator.mask_visualizations import (
    MaskData,
    MaskVisualization,
    MaskVisualizationData,
    AnnotationObject,
)


class Annotator:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = SamInference(
            sam_checkpoint="sam_vit_b_01ec64.pth", model_type="vit_b", device=device
        )
        self.annotation: AnnotationObject = None
        self.next_annotation: AnnotationObject = None
        self.mask_idx = 0

        self.manual_annotation_enabled = False
        self.manual_mask_points = []
        self.manual_mask_point_labels = []

    def toggle_manual_annotation(self):
        self.annotation.preview_mask = None
        self.manual_mask_points = []
        self.manual_mask_point_labels = []
        if not self.manual_annotation_enabled and not self.sam.is_img_embedded:
            print("Embed image before manually annotation")
            return
        self.manual_annotation_enabled = not self.manual_annotation_enabled

    def predict_sam_manually(self, position: tuple[int]):
        if self.manual_annotation_enabled:
            # create live mask preview
            self.annotation.preview_mask = (
                self.sam.predict(
                    pts=np.array(
                        [[position[0], position[1]], *self.manual_mask_points]
                    ),
                    pts_labels=np.array([1, *self.manual_mask_point_labels]),
                )
                * 255
            )
            self.update_collections(self.annotation)

    def update_mask_idx(self, new_idx: int = 0):
        # TODO: error handling if new_idx is out of bounds
        # currently its going to n-last mask when negative
        self.mask_idx = new_idx
        self.annotation.set_current_mask(self.mask_idx)

    def create_new_annotation(
        self, filepath: Path, next_filepath: Path = None
    ) -> tuple[bool]:
        embed_current = False
        embed_next = False
        if self.next_annotation is None:
            self.annotation = AnnotationObject(filepath=filepath)
            embed_current = True
        else:
            # TODO: check for bugs of shallow copies
            self.annotation = self.next_annotation

        if next_filepath is None:
            self.next_annotation = None
        else:
            self.next_annotation = AnnotationObject(filepath=next_filepath)
            embed_next = True

        return embed_current, embed_next

    def get_annotation_img_name(self):
        if self.annotation:
            return self.annotation.img_name
        else:
            return None

    def get_next_annotation_img_name(self):
        if self.next_annotation:
            return self.next_annotation.img_name
        else:
            return None

    def update_sam_features_to_current_annotation(self):
        self.sam.predictor.set_features(
            features=self.annotation.features,
            original_size=self.annotation.original_size,
            input_size=self.annotation.input_size,
        )

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
        annot = self.annotation

        if self.manual_annotation_enabled:
            mask_to_store = MaskData(mask=annot.preview_mask, origin="sam_interactive")
            annot.masks.insert(self.mask_idx, mask_to_store)
            annot.mask_decisions.insert(self.mask_idx, True)
        else:
            mask_to_store = annot.masks[self.mask_idx]
            annot.mask_decisions[self.mask_idx] = True

        annot.good_masks.append(mask_to_store)
        self.mask_idx += 1

        self.update_collections(annot)
        if self.mask_idx >= len(annot.masks) - 1:
            next_mask_center = None  # all masks have been labeled
        else:
            self.annotation.set_current_mask(self.mask_idx)
            self.preselect_mask()
            next_mask_center = self.annotation.masks[self.mask_idx].center
        # TODO: instead of done return coordinates of center of next mask
        return next_mask_center

    def bad_mask(self):
        annot = self.annotation

        annot.mask_decisions[self.mask_idx] = False
        self.mask_idx += 1

        self.update_collections(annot)
        if self.mask_idx >= len(annot.masks) - 1:
            next_mask_center = None  # all masks have been labeled
        else:
            self.annotation.set_current_mask(self.mask_idx)
            self.preselect_mask()
            next_mask_center = self.annotation.masks[self.mask_idx].center
        # TODO: instead of done return coordinates of center of next mask
        return next_mask_center

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

        mask_vis = MaskVisualization(annotation=annot)
        mvis_data: MaskVisualizationData = self.annotation.mask_visualizations
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

        if self.manual_annotation_enabled:
            img_sam_preview = mask_vis.get_sam_preview(
                self.manual_mask_points, self.manual_mask_point_labels
            )
            mvis_data.img_sam_preview = img_sam_preview

        mvis_data.maskinrgb = maskinrgb
        mvis_data.masked_img = masked_img
        mvis_data.mask_collection = mask_collection
        mvis_data.masked_img_cnt = masked_img_cnt
        mvis_data.mask_collection_cnt = mask_collection_cnt

        self.annotation.good_masks = mask_vis.mask_objs
