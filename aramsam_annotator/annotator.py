from aramsam_annotator.backend.session import AnnotationSession
from aramsam_annotator.backend.storage import AnnotationRepository, ORIGIN_CODES
from functools import cached_property
from aramsam_annotator.backend.editing import AnnotationEditor
from aramsam_annotator.backend.rendering import AnnotationRenderer
import numpy as np
import torch
import cv2
from pathlib import Path
import os
import time
import glob

from aramsam_annotator.run_sam import SamInference, Sam2Inference
from aramsam_annotator.run_yolo import YoloInference
from aramsam_annotator.tracker import PanoImageAligner
from aramsam_annotator.mask_visualizations import (
    MaskData,
    MaskVisualizationData,
    AnnotationObject,
    MaskIdHandler,
)
from aramsam_annotator.configs import AramsamConfigs


class Annotator:
    """Annotation state and backwards-compatible editing/model API.

    Editing and rendering operate on this state; session transitions and storage
    have independent, testable services. Model wrappers retain their public API.
    """
    def __init__(self, configs: AramsamConfigs) -> None:
        self.configs = configs
        self.sam_ckpt = self.configs.sam_configs.model_ckpt_p
        self.sam_model_type = self.configs.sam_configs.model_type
        self.sam = None
        self.yolo = None
        self.device = "cuda" if configs.use_gpu and torch.cuda.is_available() else "cpu"
        self.mask_id_handler = MaskIdHandler()

        self.annotation: AnnotationObject = None
        self.next_annotation: AnnotationObject = None
        self.mask_idx = 0
        self.preview_obj_id = None

        self.manual_annotation_enabled = False
        self.polygon_drawing_enabled = False
        self.mask_deletion_enabled = False
        self.manual_mask_points = []
        self.manual_mask_point_labels = []
        self.previoius_toggle_state: dict[str, bool] | None = None

        self.origin_codes = dict(ORIGIN_CODES)
        self.time_stamp = None  # in deciseconds (1/10th of a second)

    def set_sam_version(self, sam_gen: int = 1, background_embedding: bool = True):

        if sam_gen == 2:
            if self.sam_ckpt is None:
                sam2_ckpt = "sam2.1_hiera_small.pt"
            else:
                sam2_ckpt = self.sam_ckpt
            if self.sam_model_type is None:
                sam2_model_type = "configs/sam2.1/sam2.1_hiera_s.yaml"  # Sam2.1 config files have to be starting with "configs/sam2.1/", others don't (e.g. "sam2_hiera_l.yaml")
            else:
                sam2_model_type = self.sam_model_type
            self.sam = Sam2Inference(
                self.mask_id_handler,
                sam2_checkpoint=sam2_ckpt,
                cfg_path=sam2_model_type,
                device=self.device,
                background_embedding=background_embedding,
            )
        elif sam_gen == 1:
            sam1_ckpt = self.sam_ckpt
            sam1_model_type = self.sam_model_type

            self.sam = SamInference(
                self.mask_id_handler,
                sam_checkpoint=sam1_ckpt,
                model_type=sam1_model_type,
                device=self.device,
                background_embedding=background_embedding,
            )
        else:
            raise NotImplementedError("This generation of Sam is not implemented.")

    def init_time_stamp(self):
        self.time_stamp = round(time.time() * 10)

    def _get_time_stamp(self):
        if self.time_stamp is None:
            self.init_time_stamp()
        current_ts = round(time.time() * 10)
        return current_ts - self.time_stamp

    def reset_toggles(self, toggles_only=False):
        return self.editing.reset_toggles(toggles_only)

    def toggle_manual_annotation(self):
        return self.editing.toggle_manual_annotation()

    def toggle_polygon_drawing(self):
        return self.editing.toggle_polygon_drawing()

    def toggle_mask_deletion(self):
        return self.editing.toggle_mask_deletion()

    def reset_manual_annotation(self):
        return self.editing.reset_manual_annotation()

    def predict_sam_manually(self, position: tuple[int]):
        return self.editing.predict_sam_manually(position)

    def mask_from_drawing(self, mouse_pos: tuple[int] = None):
        return self.editing.mask_from_drawing(mouse_pos)

    def update_mask_idx(self, new_idx: int = 0):
        return self.editing.update_mask_idx(new_idx)

    def create_new_annotation(self, filepath: Path, next_filepath: Path | None = None) -> tuple[bool, bool]:
        pair = self.session.prepare_pair(self.annotation, self.next_annotation, filepath, next_filepath)
        self.annotation, self.next_annotation, embed_current, embed_next = pair
        accepted_ids = {mask.mid for mask in self.annotation.good_masks}
        for mask, decision in zip(self.annotation.masks, self.annotation.mask_decisions):
            if decision and mask.mid not in accepted_ids:
                self.annotation.good_masks.append(mask)
                accepted_ids.add(mask.mid)
        self.mask_idx = 0
        while (self.mask_idx < len(self.annotation.mask_decisions)
               and self.annotation.mask_decisions[self.mask_idx]):
            self.mask_idx += 1
        self.preview_obj_id = None
        self.time_stamp = None
        self.previoius_toggle_state = None
        self.reset_toggles()
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

    def prepare_yolo(self):
        if self.annotation is None:
            raise ValueError("No annotation object found.")

        self.mask_idx = 0
        if self.yolo is None:
            self.yolo = YoloInference(self.mask_id_handler, device=self.device)
            self.yolo.load_checkpoint(self.configs.yolo_model_ckpt_p)
        self.yolo.set_img(self.annotation.img)

    def prepare_amg(self, bbox_tracker: PanoImageAligner = None):
        if self.annotation is None:
            raise ValueError("No annotation object found.")

        self.mask_idx = 0
        self.sam.amg.set_visualization_img(self.annotation.img)

        if bbox_tracker is not None:
            self._propagate_bboxes(bbox_tracker)

        accepted_ids = {mask.mid for mask in self.annotation.good_masks}
        for decision, mask in zip(self.annotation.mask_decisions, self.annotation.masks):
            if decision and mask.mid not in accepted_ids:
                mask.time_stamp = mask.time_stamp or 1
                self.annotation.good_masks.append(mask)
                accepted_ids.add(mask.mid)
        while (self.mask_idx < len(self.annotation.mask_decisions)
               and self.annotation.mask_decisions[self.mask_idx]):
            self.mask_idx += 1

    def _propagate_bboxes(self, bbox_tracker: PanoImageAligner):
        tracked_bboxes = bbox_tracker.track(self.annotation)
        if len(tracked_bboxes) == 0:
            print("No bboxes to propagate")
            return
        input_bboxes = self.sam.transform_bboxes(
            tracked_bboxes, self.annotation.img.shape[:2]
        )
        # TODO: fix box propagation with activated background embedding
        prop_mask_out_torch = self.sam.predict_batch(bboxes=input_bboxes)
        prop_mask_out = self.sam._torch_to_npmasks(prop_mask_out_torch)
        prop_mask_objs = [
            MaskData(
                mid=self.mask_id_handler.get_id(),
                mask=mask,
                origin="Panorama_tracking",
                time_stamp=1,
            )
            for mask in prop_mask_out
        ]
        prop_mask_objs = self.convey_color_to_next_annot(prop_mask_objs)
        self.annotation.add_masks(prop_mask_objs, decision=True)

    def automatic_mask_generation(self):
        start_custom_amg = time.time()
        mask_objs, annotated_image = self.sam.amg()

        print(f"Custom AMG time: {time.time() - start_custom_amg}")
        return mask_objs, annotated_image

    def process_amg_masks(self, mask_objs: list[MaskData], annotated_image: np.ndarray):
        if len(mask_objs) == 0:
            print("No masks to process")
            return
        assert (
            isinstance(mask_objs, list)
            and isinstance(mask_objs[0], MaskData)
            and annotated_image.dtype == np.uint8
        )

        self.annotation.mask_visualizations.masked_img = annotated_image
        self.annotation.add_masks(mask_objs)

        self.update_mask_idx(self.mask_idx)
        start_updating_collections = time.time()
        self.update_collections(self.annotation)
        print(f"Updating collections time: {time.time() - start_updating_collections}")
        start_preselect = time.time()
        self.preselect_mask()
        print(f"Preselect time: {time.time() - start_preselect}")

    def process_yolo_bboxes(self, bboxes: list[MaskData]):
        if len(bboxes) == 0:
            return
        self.annotation.add_masks(bboxes, decision=True)
        self.update_mask_idx(self.mask_idx)
        self.preselect_mask()
        self.update_collections(self.annotation)

    def convey_color_to_next_annot(self, next_mask_objs: list[MaskData]):
        for mobj in self.annotation.good_masks:
            mid = mobj.mid
            for next_mobj in next_mask_objs:
                if next_mobj.mid == mid:
                    next_mobj.color_idx = mobj.color_idx
        return next_mask_objs

    def good_mask(self, time_stamp: int | None = None, class_id: int = None):
        return self.editing.good_mask(time_stamp, class_id)

    def bad_mask(self):
        return self.editing.bad_mask()

    def preselect_mask(self, max_overlap_ratio: float = 0.4):
        return self.editing.preselect_mask(max_overlap_ratio)

    def _recycle_mask_meta_data(self, popped_mobj: MaskData):
        return self.editing._recycle_mask_meta_data(popped_mobj)

    def _clear_unfinished_polygon(self):
        return self.editing._clear_unfinished_polygon()

    def step_back(self):
        return self.editing.step_back()

    def get_preview_object_id(self, position: tuple[int]):
        return self.editing.get_preview_object_id(position)

    def delete_mask(self, midtopop: int):
        return self.editing.delete_mask(midtopop)

    def _get_mask_id(self, mask_path: str):
        mask_name = os.path.basename(mask_path).split(".")[0]
        mask_id = mask_name.split("_")[-1]
        return int(mask_id)

    def load_tutorial_masks(self, mode: str):
        """
        Starts the tutorial overlay.
        Parameters:
        - mode: Can be "ui_overview" or "kernel_examples".
        """

        if mode == "ui_overview":
            origin = "Sam2_tracking"
            mask_p = (
                "ExperimentData/TutorialImages/39320223511025_low_192_annots/masks/*"
            )
        elif mode == "kernel_examples":
            origin = "Sam1_proposed"
            mask_p = (
                "ExperimentData/TutorialImages/39320223532020_low_64_annots/masks/*"
            )

        masks_paths0 = glob.glob(mask_p)
        masks_paths = sorted(masks_paths0, key=self._get_mask_id)

        mask_objs = [
            MaskData(
                mid=self.mask_id_handler.get_id(),
                mask=cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE),
                origin=origin,
            )
            for mask_p in masks_paths
        ]
        self.annotation.masks = []
        self.annotation.mask_decisions = []
        if mode == "ui_overview":
            self.annotation.add_masks(mask_objs, decision=True)

        elif mode == "kernel_examples":
            self.annotation.add_masks(mask_objs, decision=False)
        self.update_mask_idx()
        self.update_collections(self.annotation)
        self.preselect_mask()

    def update_collections(self, annot: AnnotationObject, current_mouse_pos=None):
        return self.rendering.update_collections(annot, current_mouse_pos)

    def save_annotations(self, save_path: Path, save_suffix: str = None) -> bool:
        return self.repository.save(self.annotation, save_path, save_suffix)

    def save_bboxes_yolo(self, save_path: Path):
        return self.repository.save_yolo(self.annotation, save_path, segmentation=False)

    def save_masks_yolo(self, save_path: Path):
        return self.repository.save_yolo(self.annotation, save_path, segmentation=True)

    def save_masks_and_logs(self, save_path: Path, save_suffix: str = None):
        return self.repository.save_native(self.annotation, save_path, save_suffix)

    def _log_and_save_masks(self, mask_objs: list[MaskData], mask_dir: str = None):
        return self.repository.log_masks(mask_objs, mask_dir)

    @cached_property
    def editing(self):
        return AnnotationEditor(self)

    @cached_property
    def rendering(self):
        return AnnotationRenderer(self)

    @cached_property
    def session(self):
        return AnnotationSession()

    @property
    def repository(self):
        return AnnotationRepository(self.configs.save_data, self.origin_codes)
