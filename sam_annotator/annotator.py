import numpy as np
import torch
import cv2
from pathlib import Path
import os
import json
import time
import glob

from sam_annotator.run_sam import SamInference, Sam2Inference
from sam_annotator.tracker import PanoImageAligner
from sam_annotator.mask_visualizations import (
    MaskData,
    MaskVisualizationData,
    AnnotationObject,
    MaskIdHandler,
)


class Annotator:
    def __init__(
        self,
        sam_ckpt: str = None,
        sam_model_type: str = None,
    ) -> None:
        self.sam_ckpt = sam_ckpt
        self.sam_model_type = sam_model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mask_id_handler = MaskIdHandler()

        self.annotation: AnnotationObject = None
        self.next_annotation: AnnotationObject = None
        self.mask_idx = 0

        self.manual_annotation_enabled = False
        self.polygon_drawing_enabled = False
        self.mask_deletion_enabled = False
        self.manual_mask_points = []
        self.manual_mask_point_labels = []
        self.previoius_toggle_state: dict[str, bool] | None = None

        self.origin_codes = {
            "Sam1_proposed": "s1p",
            "Sam2_proposed": "s2p",
            "Sam1_interactive": "s1i",
            "Sam2_interactive": "s2i",
            "Polygon_drawing": "plg",
            "Sam2_tracking": "s2t",
            "Panorama_tracking": "pat",
        }
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
                background_embedding=background_embedding,
            )
        elif sam_gen == 1:
            if self.sam_ckpt is None:
                sam1_ckpt = "sam_vit_h_4b8939.pth"
            else:
                sam1_ckpt = self.sam_ckpt
            if self.sam_model_type is None:
                sam1_model_type = "vit_h"
            else:
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
        current_ts = round(time.time() * 10)
        return current_ts - self.time_stamp

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
            self.mask_deletion_enabled = False

    def toggle_polygon_drawing(self):
        self.reset_manual_annotation()
        self.polygon_drawing_enabled = not self.polygon_drawing_enabled
        if self.polygon_drawing_enabled:
            self.manual_annotation_enabled = False
            self.mask_deletion_enabled = False

    def toggle_mask_deletion(self):
        self.reset_manual_annotation()
        self.mask_deletion_enabled = not self.mask_deletion_enabled
        if self.mask_deletion_enabled:
            self.previoius_toggle_state = {
                "manual": self.manual_annotation_enabled,
                "polygon": self.polygon_drawing_enabled,
            }
            self.manual_annotation_enabled = False
            self.polygon_drawing_enabled = False
        else:
            self.manual_annotation_enabled = self.previoius_toggle_state["manual"]
            self.polygon_drawing_enabled = self.previoius_toggle_state["polygon"]
            self.previoius_toggle_state = None

    def reset_manual_annotation(self):
        self.annotation.preview_mask = None
        self.annotation.mask_visualizations.img_sam_preview = None
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

    def prepare_amg(self, bbox_tracker: PanoImageAligner = None):
        if self.annotation is None:
            raise ValueError("No annotation object found.")

        self.mask_idx = 0
        self.sam.amg.set_visualization_img(self.annotation.img)

        if bbox_tracker is not None:
            tracked_bboxes = bbox_tracker.track(self.annotation)
            input_bboxes = self.sam.transform_bboxes(
                tracked_bboxes, self.annotation.img.shape[:2]
            )
            prop_mask_out_torch = self.sam.predict_batch(bboxes=input_bboxes)
            prop_mask_out = self.sam._torch_to_npmasks(prop_mask_out_torch)
            prop_mask_objs = [
                MaskData(
                    mid=self.mask_id_handler.set_id(),
                    mask=mask,
                    origin="Panorama_tracking",
                    time_stamp=1,
                )
                for mask in prop_mask_out
            ]
            prop_mask_objs = self.convey_color_to_next_annot(prop_mask_objs)
            self.annotation.add_masks(prop_mask_objs, decision=True)

        start_setting_masks = time.time()
        if self.annotation.masks:
            for dec, mobj in zip(self.annotation.mask_decisions, self.annotation.masks):
                if dec:
                    mobj.time_stamp = 1
                    self.annotation.good_masks.append(mobj)
                    if self.mask_id_handler._id == mobj.mid:
                        raise ValueError("Mask ID is not unique and set correctly")
                    self.mask_idx += 1
                else:
                    raise ValueError(
                        "Tracked annotations have not been annotated Correctly. Mask decision 'False' received"
                    )
        print(f"Setting masks time: {time.time() - start_setting_masks}")

    def automatic_mask_generation(self):
        start_custom_amg = time.time()
        mask_objs, annotated_image = self.sam.amg()

        print(f"Custom AMG time: {time.time() - start_custom_amg}")
        return mask_objs, annotated_image

    def process_amg_masks(self, mask_objs: list[MaskData], annotated_image: np.ndarray):
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

    def convey_color_to_next_annot(self, next_mask_objs: list[MaskData]):
        for mobj in self.annotation.good_masks:
            mid = mobj.mid
            for next_mobj in next_mask_objs:
                if next_mobj.mid == mid:
                    next_mobj.color_idx = mobj.color_idx
        return next_mask_objs

    def good_mask(self, time_stamp: int | None = None):
        annot = self.annotation
        if self.manual_annotation_enabled:
            origin = (
                "Sam1_interactive"
                if isinstance(self.sam, SamInference)
                else "Sam2_interactive"
            )
            if annot.preview_mask is None:
                return "Mask not ready"

            mask_to_store = MaskData(
                mid=self.mask_id_handler.set_id(),
                mask=annot.preview_mask,
                origin=origin,
                time_stamp=self._get_time_stamp(),
            )
            annot.masks.insert(self.mask_idx, mask_to_store)
            annot.mask_decisions.insert(self.mask_idx, True)
            self.reset_manual_annotation()

        elif self.polygon_drawing_enabled:
            if annot.preview_mask is None:
                return "No polygon provided"
            mask_to_store = MaskData(
                mid=self.mask_id_handler.set_id(),
                mask=annot.preview_mask,
                origin="Polygon_drawing",
                time_stamp=self._get_time_stamp(),
            )
            annot.masks.insert(self.mask_idx, mask_to_store)
            annot.mask_decisions.insert(self.mask_idx, True)
            self.reset_manual_annotation()

        elif len(annot.masks) > self.mask_idx:
            mask_obj = annot.masks[self.mask_idx]
            if time_stamp is None:
                time_stamp = self._get_time_stamp()
            mask_to_store = MaskData(
                mid=(
                    self.mask_id_handler.set_id()
                    if mask_obj.mid is None
                    else mask_obj.mid
                ),
                mask=mask_obj.mask,
                origin=mask_obj.origin,
                color_idx=mask_obj.color_idx,
                center=mask_obj.center,
                contour=mask_obj.contour,
                time_stamp=time_stamp,
            )
            annot.mask_decisions[self.mask_idx] = True

        else:
            return None
        if mask_to_store.mask is None:
            print("No mask to store")
            return (0, 0)

        annot.good_masks.append(mask_to_store)
        self.mask_idx += 1

        self.update_collections(annot)
        if self.mask_idx >= len(annot.masks):
            next_mask_center = None  # all masks have been labeled
        elif self.manual_annotation_enabled or self.polygon_drawing_enabled:
            next_mask_center = ""
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
        mcenter = ""

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
            mcenter = self.good_mask(time_stamp=1)
        return mcenter

    def _recycle_mask_meta_data(self, popped_mobj: MaskData):
        for i, mobj in enumerate(self.annotation.masks):
            if mobj.mid == popped_mobj.mid:
                if mobj.center is None:
                    mobj.center = popped_mobj.center
                if mobj.contour is None:
                    mobj.contour = popped_mobj.contour
                if mobj.color_idx is None:
                    mobj.color_idx = popped_mobj.color_idx

    def _clear_unfinished_polygon(self):
        self.manual_mask_points = []
        self.manual_mask_point_labels = []
        self.annotation.preview_mask = None
        pass

    def step_back(self):
        annot = self.annotation

        if self.polygon_drawing_enabled and len(self.manual_mask_points) > 0:
            self._clear_unfinished_polygon()
            return

        if self.mask_idx == 0:
            return

        # if len(annot.good_masks) == 0:
        # return

        if not (
            self.manual_annotation_enabled
            and not "interactive" in annot.good_masks[-1].origin
        ) and not (
            self.polygon_drawing_enabled
            and not "Polygon" in annot.good_masks[-1].origin
        ):

            if annot.mask_decisions[self.mask_idx - 1] and len(annot.good_masks) > 0:
                popped_mobj = annot.good_masks.pop()
                self._recycle_mask_meta_data(popped_mobj)

            annot.mask_decisions[self.mask_idx - 1] = False
            self.mask_idx -= 1
            return annot.masks[self.mask_idx].center

    def highlight_mask_at_point(self, position: tuple[int]):
        xindx, yindx = position
        if (
            xindx >= self.annotation.img.shape[1]
            or yindx >= self.annotation.img.shape[0]
        ):
            return None
        elif len(self.annotation.good_masks) == 0:
            self.annotation.preview_mask = None
            return None

        for mobj in self.annotation.good_masks:
            if mobj.mask[yindx, xindx] > 0:
                self.annotation.preview_mask = mobj.mask
                break
        self.update_collections(self.annotation)
        return mobj.mid

    def delete_mask(self, midtopop: int):
        startdeltime = time.time()
        annot = self.annotation
        for i, mobj in enumerate(annot.good_masks):
            if mobj.mid == midtopop:
                popped_mobj = annot.good_masks.pop(i)
                self._recycle_mask_meta_data(popped_mobj)
                mask_dec_idx = [
                    i for i, m in enumerate(annot.masks) if m.mid == midtopop
                ]
                assert len(mask_dec_idx) <= 1
                if len(mask_dec_idx) == 0:
                    return
                self.annotation.mask_decisions[mask_dec_idx[0]] = False
                self.annotation.preview_mask = None
                break
        print(f"Delete mask time: {time.time() - startdeltime}")

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
                self.mask_id_handler.set_id(),
                cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE),
                origin,
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

    def update_collections(self, annot: AnnotationObject):

        mask_vis = self.annotation.mask_visualizer
        mask_vis.set_annotation(annotation=annot)

        mvis_data: MaskVisualizationData = self.annotation.mask_visualizations

        if self.manual_annotation_enabled:
            img_sam_preview = mask_vis.get_sam_preview(
                self.manual_mask_points, self.manual_mask_point_labels
            )
            mvis_data.img_sam_preview = img_sam_preview
        elif self.polygon_drawing_enabled:
            img_sam_preview = mask_vis.get_polygon_preview(self.manual_mask_points)
            mvis_data.img_sam_preview = img_sam_preview
        elif self.mask_deletion_enabled:
            img_sam_preview = mask_vis.get_mask_deletion_preview()
            mvis_data.img_sam_preview = img_sam_preview

        masked_img = mask_vis.get_masked_img()  # masks get contours
        mask_collection = mask_vis.get_mask_collection()

        if (
            len(annot.masks) > self.mask_idx
            and not self.manual_annotation_enabled
            and not self.polygon_drawing_enabled
            and not self.mask_deletion_enabled
        ):
            mask_obj = annot.masks[self.mask_idx]
            if mask_obj.contour is None:
                mask_vis.set_contour(mask_obj)
            cnt = mask_obj.contour
            maskinrgb = mask_vis.get_maskinrgb(mask_obj)

        else:
            # after all proposed masks have been labeled
            maskinrgb = mvis_data.img
            cnt = None

        masked_img_cnt = mask_vis.get_masked_img_cnt(cnt)
        mask_collection_cnt = mask_vis.get_mask_collection_cnt(cnt)

        if len(annot.masks) > self.mask_idx:
            self.annotation.set_current_mask(self.mask_idx)
        mvis_data.maskinrgb = maskinrgb
        mvis_data.masked_img = masked_img
        mvis_data.mask_collection = mask_collection
        mvis_data.masked_img_cnt = masked_img_cnt
        mvis_data.mask_collection_cnt = mask_collection_cnt

        self.annotation.good_masks = mask_vis.mask_objs

    def save_annotations(self, save_path: Path, save_suffix: str = None) -> bool:
        output_masks_exist = False
        img_id = os.path.splitext(self.annotation.img_name)[0]
        annots_path = os.path.join(save_path, f"{img_id}_annots")
        if save_suffix is not None:
            annots_path = f"{annots_path}_{save_suffix}"

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

        good_masks_log_dict, total_time = self._log_and_save_masks(
            self.annotation.good_masks, mask_dir
        )
        all_masks_log_dict, _ = self._log_and_save_masks(self.annotation.masks)
        log_dict = {
            "All_masks": all_masks_log_dict,
            "Selected_masks": good_masks_log_dict,
            "Total_time": total_time,
        }

        log_path = os.path.join(annots_path, "log.json")

        with open(log_path, "w") as json_file:
            json.dump(log_dict, json_file, indent=4)

        print(f"Annotations saved to {annots_path}")
        return output_masks_exist

    def _log_and_save_masks(self, mask_objs: list[MaskData], mask_dir: str = None):
        """Masks are only saved if mask_dir is provided"""
        log_dict = {key: 0 for key in self.origin_codes.keys()}
        latest_ts = 0
        for i, m in enumerate(mask_objs):

            if m.origin not in self.origin_codes.keys():
                raise ValueError(f"Origin code not found for {m.origin}")
            log_dict[m.origin] += 1
            mask_code = self.origin_codes[m.origin]

            if mask_dir is not None:
                mask_ts = m.time_stamp
                if mask_ts > latest_ts:
                    latest_ts = mask_ts
                mask_name = f"mask_{mask_code}_{mask_ts}_{i}.png"
                mask_dest_path = os.path.join(mask_dir, mask_name)
                cv2.imwrite(mask_dest_path, m.mask)

        total_mask_n = len(mask_objs)
        log_dict["Total_masks"] = total_mask_n

        if mask_dir:
            return log_dict, latest_ts
        else:
            return log_dict, None
