"""Translate existing UI gestures into annotation edits."""
import time


class InteractionController:
    def __init__(self, context):
        self.context = context

    def add_good_mask(self):
        context = self.context
        if context.annotator.annotation is None:
            return
        if context.sam_gen == 2 and context.annotator.next_annotation is not None:
            context.start_mask_batch_thread()
        new_center = context.annotator.good_mask(class_id=context.ui.get_selected_class())
        if new_center is None:
            if (
                context.experiment_mode == "structured"
                and context.annotator.polygon_drawing_enabled == False
                and context.annotator.manual_annotation_enabled == False
            ):
                context.ui.create_message_box(
                    False,
                    "All proposed masks are done. Press the Next button if you want to continue with selecting masks interactively.",
                )
            context.update_ui_imgs()

        else:
            context.update_ui_imgs(center=new_center)

    def add_bad_mask(self):
        context = self.context
        if context.annotator.annotation is None:
            return
        new_center = context.annotator.bad_mask()
        if new_center is None:
            if (
                context.experiment_mode == "structured"
                and context.annotator.polygon_drawing_enabled == False
                and context.annotator.manual_annotation_enabled == False
            ):
                context.ui.create_message_box(
                    False,
                    "All proposed masks are done. Press the Next button if you want to continue with selecting masks interactively.",
                )
            context.update_ui_imgs()
        else:
            context.update_ui_imgs(center=new_center)

    def previous_mask(self):
        context = self.context
        if context.annotator.annotation is None:
            return
        new_center = context.annotator.step_back()
        context.annotator.update_collections(context.annotator.annotation)
        context.update_ui_imgs(center=new_center)

    def proposed_masks_instructions(self):
        context = self.context
        context.ui.experiment_instructions_label.setText(
            "pan with left-click, zoom with mouse wheel"
        )

    def manual_annotation(self):
        context = self.context
        if context.annotator.annotation is None:
            return
        context.annotator.toggle_manual_annotation()
        context.ui.draw_button.setChecked(False)
        context.ui.delete_button.setChecked(False)
        context.ui.set_cursor(context.annotator.manual_annotation_enabled)
        if (
            context.annotator.manual_annotation_enabled
            and context.experiment_mode is not None
        ):
            context.ui.experiment_instructions_label.setText(
                "positive point ('a'), negative point ('s'), undo point ('d')"
            )
        elif context.experiment_mode is not None:
            context.proposed_masks_instructions()
        context.annotator.update_collections(context.annotator.annotation)
        context.update_ui_imgs()

    def draw_polygon(self):
        context = self.context
        if context.annotator.annotation is None:
            return
        context.annotator.toggle_polygon_drawing()
        context.ui.manual_annotation_button.setChecked(False)
        context.ui.delete_button.setChecked(False)
        context.ui.set_cursor(context.annotator.polygon_drawing_enabled)
        if context.annotator.polygon_drawing_enabled and context.experiment_mode is not None:
            context.ui.experiment_instructions_label.setText(
                "positive point ('a'/'right-click'), undo point ('d')"
            )
        elif context.experiment_mode is not None:
            context.proposed_masks_instructions()
        context.annotator.update_collections(context.annotator.annotation)
        context.update_ui_imgs()

    def select_masks_to_delete(self):

        # restoring previous state after mask deletion
        context = self.context
        if context.annotator.annotation is None:
            return
        if context.annotator.previoius_toggle_state is not None:
            man_state = context.annotator.previoius_toggle_state["manual"]
            poly_state = context.annotator.previoius_toggle_state["polygon"]
        else:
            man_state, poly_state = False, False
        context.ui.manual_annotation_button.setChecked(man_state)
        context.ui.draw_button.setChecked(poly_state)
        context.annotator.toggle_mask_deletion()
        if context.annotator.mask_deletion_enabled and context.experiment_mode is not None:
            context.ui.experiment_instructions_label.setText(
                "right-click on mask to delete it"
            )
        elif context.experiment_mode is not None:
            context.proposed_masks_instructions()
        context.annotator.update_collections(context.annotator.annotation)
        context.update_ui_imgs()

    def manage_mouse_move(self, point: tuple[int]):
        context = self.context
        if context.annotator.annotation is None:
            return
        height, width = context.annotator.annotation.img.shape[:2]
        if point[0] < 0 or point[0] >= width:
            return
        if point[1] < 0 or point[1] >= height:
            return

        current_time = time.time_ns()
        delta = current_time - context.last_sam_preview_time_stamp
        context.mouse_pos = point
        if delta * 1e-9 > 1 / context.manual_sam_preview_updates_per_sec:
            context.last_sam_preview_time_stamp = current_time
            if context.annotator.manual_annotation_enabled:
                if not context.mutex.tryLock(0):
                    return
                try:
                    context.annotator.predict_sam_manually(point)
                finally:
                    context.mutex.unlock()
            elif context.annotator.mask_deletion_enabled:
                context.annotator.get_preview_object_id(point)
            elif context.annotator.polygon_drawing_enabled:
                context.annotator.mask_from_drawing(point)
            else:
                return

            context.update_ui_imgs()

    def manage_mouse_action(self, label: int):
        context = self.context
        if context.annotator.annotation is None:
            return
        if (
            context.annotator.manual_annotation_enabled
            or context.annotator.polygon_drawing_enabled
        ):
            context.add_sam_preview_annotation_point(label)
        elif context.annotator.mask_deletion_enabled:
            context.delete_mask_at_point(label)
        else:
            return

    def add_sam_preview_annotation_point(self, label: int):
        context = self.context
        if getattr(context, "mouse_pos", None) is None:
            return
        if context.annotator.annotation is None:
            return
        if label == -1:
            if context.annotator.manual_mask_points:
                context.annotator.manual_mask_points.pop()
                context.annotator.manual_mask_point_labels.pop()
        else:
            context.annotator.manual_mask_points.append(context.mouse_pos)
            context.annotator.manual_mask_point_labels.append(label)
            if (
                len(context.annotator.manual_mask_points) == 1
                and context.experiment_mode == "polygon"
            ):
                context.annotator.init_time_stamp()

        if context.annotator.manual_annotation_enabled:
            if context.mutex.tryLock(100):
                try:
                    context.annotator.predict_sam_manually(context.mouse_pos)
                finally:
                    context.mutex.unlock()
        elif context.annotator.polygon_drawing_enabled:
            context.annotator.mask_from_drawing()

        context.update_ui_imgs()

    def delete_mask_at_point(self, label: int):
        context = self.context
        if getattr(context, "mouse_pos", None) is None:
            return
        if context.annotator.annotation is None:
            return
        mid = context.annotator.get_preview_object_id(context.mouse_pos)
        if mid is None:
            return
        elif label == 1:
            context.annotator.delete_mask(mid)
            context.annotator.update_collections(context.annotator.annotation)
        context.update_ui_imgs()

