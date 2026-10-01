"""Mask decisions, interactive drawing, deletion, and undo."""
import cv2
import numpy as np
from aramsam_annotator.run_sam import SamInference
from aramsam_annotator.mask_visualizations import MaskData


class AnnotationEditor:
    def __init__(self, context):
        self.context = context
        self.preselecting = False

    def reset_toggles(self, toggles_only=False):
        context = self.context
        context.manual_annotation_enabled = False
        context.polygon_drawing_enabled = False
        context.mask_deletion_enabled = False
        if toggles_only:
            return
        context.reset_manual_annotation()

    def toggle_manual_annotation(self):
        context = self.context
        context.reset_manual_annotation()
        if context.sam is None or (not context.manual_annotation_enabled and not context.sam.predictor.is_image_set):
            print("Embed image before manually annotation")
            return
        context.manual_annotation_enabled = not context.manual_annotation_enabled
        if context.manual_annotation_enabled:
            context.polygon_drawing_enabled = False
            context.mask_deletion_enabled = False

    def toggle_polygon_drawing(self):
        context = self.context
        context.reset_manual_annotation()
        context.polygon_drawing_enabled = not context.polygon_drawing_enabled
        if context.polygon_drawing_enabled:
            context.manual_annotation_enabled = False
            context.mask_deletion_enabled = False

    def toggle_mask_deletion(self):
        context = self.context
        context.reset_manual_annotation()
        context.mask_deletion_enabled = not context.mask_deletion_enabled
        if context.mask_deletion_enabled:
            context.previoius_toggle_state = {
                "manual": context.manual_annotation_enabled,
                "polygon": context.polygon_drawing_enabled,
            }
            context.manual_annotation_enabled = False
            context.polygon_drawing_enabled = False
        else:
            context.manual_annotation_enabled = context.previoius_toggle_state["manual"]
            context.polygon_drawing_enabled = context.previoius_toggle_state["polygon"]
            context.previoius_toggle_state = None

    def reset_manual_annotation(self):
        context = self.context
        if context.annotation is not None:
            context.annotation.preview_mask = None
            context.annotation.mask_visualizations.img_sam_preview = None
        context.manual_mask_points = []
        context.manual_mask_point_labels = []

    def predict_sam_manually(self, position: tuple[int]):
        context = self.context
        if context.manual_annotation_enabled:
            # create live mask preview
            context.annotation.preview_mask = context.sam.predict(
                pts=np.array(
                    [[position[0], position[1]], *context.manual_mask_points],
                    dtype=np.float32,
                ),
                pts_labels=np.array(
                    [1, *context.manual_mask_point_labels], dtype=np.int32
                ),
            )
            context.update_collections(context.annotation)

    def mask_from_drawing(self, mouse_pos: tuple[int] = None):
        context = self.context
        if not context.polygon_drawing_enabled:
            return

        # bounding box should only be drawn if there is already exactly one point
        if mouse_pos is not None and len(context.manual_mask_points) != 1:
            return

        if len(context.manual_mask_points) > 1:
            context.annotation.preview_mask = np.zeros(
                context.annotation.img.shape[:2], dtype=np.uint8
            )

        if len(context.manual_mask_points) == 2:
            context.annotation.preview_mask = cv2.rectangle(
                context.annotation.preview_mask,
                tuple(context.manual_mask_points[0]),
                tuple(context.manual_mask_points[1]),
                255,
                1,
            )

        elif len(context.manual_mask_points) > 2:
            polypts = np.array(context.manual_mask_points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(context.annotation.preview_mask, [polypts], 255)
        context.update_collections(context.annotation, current_mouse_pos=mouse_pos)

    def update_mask_idx(self, new_idx: int = 0):
        context = self.context
        if new_idx < 0:
            new_idx = 0
            print("Mask index cannot be negative. Setting to 0.")
        context.mask_idx = new_idx
        context.annotation.set_current_mask(context.mask_idx)

    def good_mask(self, time_stamp: int | None = None, class_id: int = None):
        context = self.context
        annot = context.annotation
        if context.manual_annotation_enabled:
            origin = (
                "Sam1_interactive"
                if isinstance(context.sam, SamInference)
                else "Sam2_interactive"
            )
            if annot.preview_mask is None:
                return "Mask not ready"

            mask_to_store = MaskData(
                mid=context.mask_id_handler.get_id(),
                mask=annot.preview_mask,
                origin=origin,
                time_stamp=context._get_time_stamp(),
            )
            annot.masks.insert(context.mask_idx, mask_to_store)
            annot.mask_decisions.insert(context.mask_idx, True)
            context.reset_manual_annotation()

        elif context.polygon_drawing_enabled:
            if annot.preview_mask is None:
                return "No polygon provided"
            mask_to_store = MaskData(
                mid=context.mask_id_handler.get_id(),
                mask=annot.preview_mask,
                origin="Polygon_drawing",
                time_stamp=context._get_time_stamp(),
            )
            annot.masks.insert(context.mask_idx, mask_to_store)
            annot.mask_decisions.insert(context.mask_idx, True)
            context.reset_manual_annotation()

        elif len(annot.masks) > context.mask_idx:
            mask_obj = annot.masks[context.mask_idx]
            if time_stamp is None:
                time_stamp = context._get_time_stamp()
            mask_to_store = MaskData(
                mid=(
                    context.mask_id_handler.get_id()
                    if mask_obj.mid is None
                    else mask_obj.mid
                ),
                mask=mask_obj.mask,
                bbox=mask_obj.bbox,
                origin=mask_obj.origin,
                color_idx=mask_obj.color_idx,
                center=mask_obj.center,
                contour=mask_obj.contour,
                time_stamp=time_stamp,
            )
            annot.mask_decisions[context.mask_idx] = True

        else:
            return None
        if mask_to_store.mask is None and mask_to_store.bbox is None:
            print("No mask to store")
            return (0, 0)

        mask_to_store.class_id = class_id
        annot.good_masks.append(mask_to_store)
        context.mask_idx += 1

        context.update_collections(annot)
        if context.mask_idx >= len(annot.masks):
            next_mask_center = None  # all masks have been labeled
        elif context.manual_annotation_enabled or context.polygon_drawing_enabled:
            next_mask_center = ""
        else:
            context.annotation.set_current_mask(context.mask_idx)
            if context.preselect_mask() is None:
                return None
            next_mask_center = context.annotation.masks[context.mask_idx].center
        return next_mask_center

    def bad_mask(self):
        context = self.context
        annot = context.annotation
        if context.mask_idx >= len(annot.masks):
            return None

        annot.mask_decisions[context.mask_idx] = False
        context.mask_idx += 1

        context.update_collections(annot)
        if context.mask_idx >= len(annot.masks):
            next_mask_center = None  # all masks have been labeled
        else:
            context.annotation.set_current_mask(context.mask_idx)
            if context.preselect_mask() is None:
                return None
            next_mask_center = context.annotation.masks[context.mask_idx].center
        return next_mask_center

    def preselect_mask(self, max_overlap_ratio: float = 0.4):
        context = self.context
        if self.preselecting:
            return ""
        self.preselecting = True
        try:
            annot = context.annotation
            while context.mask_idx < len(annot.masks):
                obj = annot.masks[context.mask_idx]
                reject = False
                if obj.mask is not None:
                    size = np.count_nonzero(obj.mask)
                    collection = annot.mask_visualizations.mask_collection
                    overlap = 0
                    if collection is not None:
                        accepted = np.any(collection != 0, axis=-1)
                        overlap = np.count_nonzero(accepted & (obj.mask != 0))
                    reject = size == 0 or overlap / size > max_overlap_ratio
                if reject:
                    context.bad_mask()
                elif "tracking" in obj.origin or obj.origin == "Yolo_prediction":
                    context.good_mask(time_stamp=1, class_id=obj.class_id)
                else:
                    return ""
            return None
        finally:
            self.preselecting = False

    def _recycle_mask_meta_data(self, popped_mobj: MaskData):
        context = self.context
        for i, mobj in enumerate(context.annotation.masks):
            if mobj.mid == popped_mobj.mid:
                if mobj.center is None:
                    mobj.center = popped_mobj.center
                if mobj.contour is None:
                    mobj.contour = popped_mobj.contour
                if mobj.color_idx is None:
                    mobj.color_idx = popped_mobj.color_idx

    def _clear_unfinished_polygon(self):
        context = self.context
        context.manual_mask_points = []
        context.manual_mask_point_labels = []
        context.annotation.preview_mask = None

    def step_back(self):
        context = self.context
        annot = context.annotation

        # If a polygon is being drawn or interactive propmting is active
        # and there are manual points, clear the points.
        point_annotation_condition = (
            context.manual_annotation_enabled or context.polygon_drawing_enabled
        )
        if point_annotation_condition and context.manual_mask_points:
            context._clear_unfinished_polygon()
            return

        # Keep the cursor within this image's decision history, including when
        # recovering an annotation created before the image-transition reset.
        context.mask_idx = max(0, min(context.mask_idx, len(annot.masks), len(annot.mask_decisions)))
        if context.mask_idx == 0:
            return

        previous_idx = context.mask_idx - 1
        previous_mask = annot.masks[previous_idx]
        if context.manual_annotation_enabled and "interactive" not in previous_mask.origin:
            return
        if context.polygon_drawing_enabled and "Polygon" not in previous_mask.origin:
            return

        if annot.mask_decisions[previous_idx]:
            for idx in range(len(annot.good_masks) - 1, -1, -1):
                if annot.good_masks[idx].mid == previous_mask.mid:
                    popped_mobj = annot.good_masks.pop(idx)
                    context._recycle_mask_meta_data(popped_mobj)
                    break

        annot.mask_decisions[previous_idx] = False
        context.mask_idx = previous_idx
        return previous_mask.center

    def get_preview_object_id(self, position: tuple[int]):
        context = self.context
        xindx, yindx = position
        if (
            xindx < 0 or yindx < 0
            or xindx >= context.annotation.img.shape[1]
            or yindx >= context.annotation.img.shape[0]
        ):
            return None
        elif len(context.annotation.good_masks) == 0:
            context.annotation.preview_mask = None
            return None

        obj_id, context.annotation.preview_mask = (
            context.annotation.mask_visualizer.highlight_mask_at_point(position)
        )
        if obj_id != context.preview_obj_id:
            context.update_collections(context.annotation)
        context.preview_obj_id = obj_id
        return obj_id

    def delete_mask(self, midtopop: int):
        context = self.context
        annot = context.annotation
        for i, mobj in enumerate(annot.good_masks):
            if mobj.mid == midtopop:
                popped_mobj = annot.good_masks.pop(i)
                context._recycle_mask_meta_data(popped_mobj)
                mask_dec_idx = [
                    i for i, m in enumerate(annot.masks) if m.mid == midtopop
                ]
                assert len(mask_dec_idx) <= 1
                if len(mask_dec_idx) == 0:
                    return
                context.annotation.mask_decisions[mask_dec_idx[0]] = False
                context.annotation.preview_mask = None
                break

