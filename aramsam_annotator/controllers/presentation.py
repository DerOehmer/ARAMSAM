"""Refresh the existing views from annotation state."""


class PresentationController:
    def __init__(self, context):
        self.context = context

    def start_user_annotation(self):
        context = self.context
        if context.annotator.annotation is None:
            return
        if context.experiment_mode is None:
            context.ui.manual_annotation_button.setEnabled(context.ui.auto_embed_box.isChecked())
            # Restore the mode that was active on the previous image
            context.navigation.restore_annotation_mode()
        context.annotator.init_time_stamp()
        context.annotator.update_collections(context.annotator.annotation)
        context.update_ui_imgs()

    def _ui_config_changed(self, fields: list[str]):
        context = self.context
        assert (
            len(fields) == 4
        ), f"Too many fields selected for visualization ({len(fields)}) expected 4"
        context.fields = fields
        context.update_ui_imgs()

    def update_ui_imgs(self, center: tuple | None | str = None):
        context = self.context
        if context.annotator.annotation is None:
            return
        mviss = context.annotator.annotation.mask_visualizations
        for idx, field in enumerate(context.fields):
            context.ui.update_main_pix_map(idx=idx, img=getattr(mviss, field))

        if type(center) == tuple:
            context.ui.center_all_annotation_visualizers(center)

