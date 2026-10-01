"""Preserve structured experiment and tutorial progression."""


class ExperimentController:
    def __init__(self, context):
        self.context = context

    def next_indicated_polygon_img(self):
        context = self.context
        print("Next polygon img")
        if len(context.annotator.annotation.good_masks) != 3:
            context.ui.create_message_box(False, "Please select exactly 3 masks")
            return
        context.select_next_img()
        # reset
        context.ui.draw_button.setDisabled(False)
        context.ui.draw_button.click()
        context.ui.draw_button.setDisabled(True)
        if context.tutorial_flag and context.experiment_step == 1:
            context.ui.start_tutorial("polygon_drawing_texts")
            context.experiment_step += 1

    def next_method(self):
        context = self.context
        print("Next method")
        if context.experiment_step == 0:
            context.experiment_step = 1
            context.ui.performing_embedding_label.setText(
                f"Step 1/3: Select the good proposed masks"
            )
            # reset
            context.ui.manual_annotation_button.setDisabled(False)
            context.ui.manual_annotation_button.click()
            context.ui.manual_annotation_button.click()
            context.ui.manual_annotation_button.setDisabled(True)
            context.ui.good_mask_button.setDisabled(False)
            context.ui.bad_mask_button.setDisabled(False)
            context.ui.back_button.setDisabled(False)
            context.ui.delete_button.setDisabled(False)
            if context.tutorial_flag:
                context.ui.start_tutorial("proposed_masks_texts")
        elif context.experiment_step == 1:
            context.experiment_step = 2
            context.ui.performing_embedding_label.setText(
                f"Step 2/3: Annotate masks interactively with SAM{context.sam_gen}"
            )
            context.ui.manual_annotation_button.setDisabled(False)
            context.ui.manual_annotation_button.click()
            context.ui.manual_annotation_button.setDisabled(True)
            if context.tutorial_flag:
                context.ui.start_tutorial("interactive_annotation_texts")
        elif context.experiment_step == 2:
            context.experiment_step = 3
            context.ui.performing_embedding_label.setText(
                f"Step 3/3: Draw polygon masks for remaining objects"
            )
            context.ui.draw_button.setDisabled(False)
            context.ui.draw_button.click()
            context.ui.draw_button.setDisabled(True)
            if context.tutorial_flag:
                context.ui.start_tutorial("polygon_drawing_texts")
        elif context.experiment_step == 3:
            context.experiment_step = 0
            if not context.img_fnames:
                context.select_next_img()
                return
            else:
                context.select_next_img()
            context.ui.performing_embedding_label.setText(
                f"Step 0/3: Check whether masks have been propagated correctly"
            )
            context.ui.good_mask_button.setDisabled(True)
            context.ui.bad_mask_button.setDisabled(True)
            context.ui.back_button.setDisabled(True)
            context.ui.delete_button.click()
            context.ui.delete_button.setDisabled(True)
            if context.tutorial_flag:
                context.ui.start_tutorial("mask_deletion_texts")

