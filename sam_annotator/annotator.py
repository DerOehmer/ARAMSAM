import numpy as np
import torch
import cv2
from pathlib import Path
import os


from sam_annotator.run_sam import SamInference, Sam2Inference
from sam_annotator.tracker import PanoImageAligner
from sam_annotator.mask_visualizations import (
    MaskData,
    MaskVisualization,
    MaskVisualizationData,
    AnnotationObject,
)


class Annotator:
    def __init__(self, sam_ckpt: str = None, sam_model_type: str = None) -> None:
        self.sam_ckpt = sam_ckpt
        self.sam_model_type = sam_model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prev_annotation: AnnotationObject = None
        self.annotation: AnnotationObject = None
        self.next_annotation: AnnotationObject = None
        self.mask_idx = 0

        self.manual_annotation_enabled = False
        self.polygon_drawing_enabled = False
        self.manual_mask_points = []
        self.manual_mask_point_labels = []

        self.origin_codes = {
            "Sam1_proposed": "s1p",
            "Sam2_proposed": "s2p",
            "Sam1_interactive": "s1i",
            "Sam2_interactive": "s2i",
            "Polygon_drawing": "plg",
            "Sam2_tracking": "s2t",
            "Panorama_tracking": "pat",
            "Kalman_tracking": "kat",
        }

    def set_sam_version(self, sam2=False):

        if sam2:
            if self.sam_ckpt is None:
                sam2_ckpt = "sam2_hiera_small.pt"
            else:
                sam2_ckpt = self.sam_ckpt
            if self.sam_model_type is None:
                sam2_model_type = "sam2_hiera_s.yaml"
            else:
                sam2_model_type = self.sam_model_type
            self.sam = Sam2Inference(
                sam2_checkpoint=sam2_ckpt, cfg_path=sam2_model_type
            )
        else:
            if self.sam_ckpt is None:
                sam1_ckpt = "sam_vit_b_01ec64.pth"
            else:
                sam1_ckpt = self.sam_ckpt
            if self.sam_model_type is None:
                sam1_model_type = "vit_b"
            else:
                sam1_model_type = self.sam_model_type
            self.sam = SamInference(
                sam_checkpoint=sam1_ckpt,
                model_type=sam1_model_type,
                device=self.device,
            )

    def reset_toggles(self):
        self.reset_manual_annotation()
        self.manual_annotation_enabled = False
        self.polygon_drawing_enabled = False

    def toggle_manual_annotation(self):
        self.reset_manual_annotation()
        if not self.manual_annotation_enabled and not self.sam.predictor.is_image_set:
            print("Embed image before manually annotation")
            return
        self.manual_annotation_enabled = not self.manual_annotation_enabled
        if self.manual_annotation_enabled:
            self.polygon_drawing_enabled = False

    def toggle_polygon_drawing(self):
        self.reset_manual_annotation()
        self.polygon_drawing_enabled = not self.polygon_drawing_enabled
        if self.polygon_drawing_enabled:
            self.manual_annotation_enabled = False

    def reset_manual_annotation(self):
        self.annotation.preview_mask = None
        self.manual_mask_points = []
        self.manual_mask_point_labels = []

    def predict_sam_manually(self, position: tuple[int]):
        if self.manual_annotation_enabled:
            # create live mask preview
            self.annotation.preview_mask = self.sam.predict(
                pts=np.array(
                    [[position[0], position[1]], *self.manual_mask_points],
                    dtype=np.float32,
                ),
                pts_labels=np.array(
                    [1, *self.manual_mask_point_labels], dtype=np.int32
                ),
            )
            self.update_collections(self.annotation)

    def mask_from_polygon(self):
        if self.polygon_drawing_enabled:
            self.annotation.preview_mask = np.zeros(
                self.annotation.img.shape[:2], dtype=np.uint8
            )
            if len(self.manual_mask_points) > 2:
                polypts = np.array(self.manual_mask_points, np.int32).reshape(
                    (-1, 1, 2)
                )
                cv2.fillPoly(self.annotation.preview_mask, [polypts], 255)
            self.update_collections(self.annotation)

    def update_mask_idx(self, new_idx: int = 0):
        if new_idx < 0:
            new_idx = 0
            print("Mask index cannot be negative. Setting to 0.")
        self.mask_idx = new_idx
        self.annotation.set_current_mask(self.mask_idx)

    def create_new_annotation(
        self, filepath: Path, next_filepath: Path | None = None
    ) -> tuple[bool]:
        embed_current = False
        embed_next = False
        if self.next_annotation is None:
            self.annotation = AnnotationObject(filepath=filepath)
            embed_current = True
        else:
            # TODO: check for bugs of shallow copies
            self.annotation = self.next_annotation
            self.reset_toggles()

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

    def predict_with_sam(self, pano_aligner: PanoImageAligner = None):
        if self.annotation is None:
            raise ValueError("No annotation object found.")

        print(self.annotation.img.shape)
        self.sam.custom_amg.set_visualization_img(self.annotation.img)
        mask_objs, annotated_image = self.sam.custom_amg(roi_pts=False, n_points=100)
        assert (
            isinstance(mask_objs, list)
            and isinstance(mask_objs[0], MaskData)
            and annotated_image.dtype == np.uint8
        )
        if pano_aligner is not None:
            pano_aligner.add_image(self.annotation.img, mask_objs)
            prop_mask_objs = pano_aligner.match_and_align()
            self.annotation.add_masks(prop_mask_objs, decision=True)
        self.annotation.mask_visualizations.masked_img = annotated_image
        self.annotation.add_masks(mask_objs)

        self.update_mask_idx()
        self.update_collections(self.annotation)
        self.preselect_mask()

    def track_good_masks(self):
        self.sam.set_masks(self.annotation.good_masks)
        prop_mask_objs: list[MaskData] = self.sam.propagate_to_next_img()
        self.next_annotation.add_masks(prop_mask_objs, decision=True)

    def good_mask(self):
        annot = self.annotation
        # TODO: ensure that correct mask is stored when switching modes
        if self.manual_annotation_enabled:
            origin = (
                "Sam1_interactive"
                if isinstance(self.sam, SamInference)
                else "Sam2_interactive"
            )
            mask_to_store = MaskData(mask=annot.preview_mask, origin=origin)
            annot.masks.insert(self.mask_idx, mask_to_store)
            annot.mask_decisions.insert(self.mask_idx, True)
            self.reset_manual_annotation()
        elif self.polygon_drawing_enabled:
            mask_to_store = MaskData(mask=annot.preview_mask, origin="Polygon_drawing")
            annot.masks.insert(self.mask_idx, mask_to_store)
            annot.mask_decisions.insert(self.mask_idx, True)
            self.reset_manual_annotation()
        elif len(annot.masks) > self.mask_idx:
            mask_to_store = annot.masks[self.mask_idx]
            annot.mask_decisions[self.mask_idx] = True
        else:
            return None

        annot.good_masks.append(mask_to_store)
        self.mask_idx += 1

        self.update_collections(annot)
        if self.mask_idx >= len(annot.masks):
            next_mask_center = None  # all masks have been labeled
        else:
            self.annotation.set_current_mask(self.mask_idx)
            if self.preselect_mask() is None:
                return None
            next_mask_center = self.annotation.masks[self.mask_idx].center
        return next_mask_center

    def bad_mask(self):
        annot = self.annotation
        if self.mask_idx >= len(annot.masks):
            return None

        annot.mask_decisions[self.mask_idx] = False
        self.mask_idx += 1

        self.update_collections(annot)
        if self.mask_idx >= len(annot.masks):
            next_mask_center = None  # all masks have been labeled
        else:
            self.annotation.set_current_mask(self.mask_idx)
            if self.preselect_mask() is None:
                return None
            next_mask_center = self.annotation.masks[self.mask_idx].center
        return next_mask_center

    def preselect_mask(self, max_overlap_ratio: float = 0.4):
        annot = self.annotation
        mask_obj: MaskData = annot.masks[self.mask_idx]
        mask = mask_obj.mask
        maskorigin = mask_obj.origin
        mcenter = (0, 0)

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
            mcenter = self.bad_mask()

        if "tracking" in maskorigin:
            mcenter = self.good_mask()
        return mcenter

    def step_back(self):
        annot = self.annotation

        if self.mask_idx > 0:

            if annot.mask_decisions[self.mask_idx - 1] and len(annot.good_masks) > 0:
                annot.good_masks.pop()

            annot.mask_decisions[self.mask_idx - 1] = False

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
            # after all proposed masks have been labeled
            maskinrgb = mvis_data.img
            masked_img_cnt = masked_img
            mask_collection_cnt = mask_collection

        if self.manual_annotation_enabled:
            img_sam_preview = mask_vis.get_sam_preview(
                self.manual_mask_points, self.manual_mask_point_labels
            )
            mvis_data.img_sam_preview = img_sam_preview

        elif self.polygon_drawing_enabled:
            img_sam_preview = mask_vis.get_polygon_preview(self.manual_mask_points)
            mvis_data.img_sam_preview = img_sam_preview
        if len(annot.masks) > self.mask_idx:
            self.annotation.set_current_mask(self.mask_idx)
        mvis_data.maskinrgb = maskinrgb
        mvis_data.masked_img = masked_img
        mvis_data.mask_collection = mask_collection
        mvis_data.masked_img_cnt = masked_img_cnt
        mvis_data.mask_collection_cnt = mask_collection_cnt

        self.annotation.good_masks = mask_vis.mask_objs

    def save_annotations(self, save_path: Path):
        output_masks_exist = False
        img_id = os.path.splitext(self.annotation.img_name)[0]
        annots_path = os.path.join(save_path, f"{img_id}_annots")

        if not os.path.exists(annots_path):
            os.makedirs(annots_path)

        cv2.imwrite(
            os.path.join(annots_path, "img.jpg"),
            self.annotation.img,
        )
        cv2.imwrite(
            os.path.join(annots_path, "annotations.jpg"),
            self.annotation.mask_visualizations.masked_img,
        )

        mask_dir = os.path.join(annots_path, "masks")

        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        else:
            output_masks_exist = True
            return output_masks_exist

        for i, m in enumerate(self.annotation.good_masks):

            if m.origin not in self.origin_codes.keys():
                raise ValueError(f"Origin code not found for {m.origin}")

            mask_code = self.origin_codes[m.origin]

            mask_name = f"mask_{mask_code}_{i}.png"
            mask_dest_path = os.path.join(mask_dir, mask_name)
            cv2.imwrite(mask_dest_path, m.mask)

        print(f"Annotations saved to {annots_path}")
        return output_masks_exist
