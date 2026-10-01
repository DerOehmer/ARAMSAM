"""Build the existing visualization outputs for an annotation."""
from aramsam_annotator.mask_visualizations import AnnotationObject, MaskVisualizationData


class AnnotationRenderer:
    def __init__(self, context):
        self.context = context

    def update_collections(self, annot: AnnotationObject, current_mouse_pos=None):
        context = self.context
        mask_vis = annot.mask_visualizer
        mask_vis.set_annotation(annotation=annot)

        mvis_data: MaskVisualizationData = annot.mask_visualizations

        if context.manual_annotation_enabled:
            img_sam_preview = mask_vis.get_sam_preview(
                context.manual_mask_points, context.manual_mask_point_labels
            )
            mvis_data.img_sam_preview = img_sam_preview
        elif context.polygon_drawing_enabled:
            img_sam_preview = mask_vis.get_drawing_preview(
                context.manual_mask_points, current_mouse_pos
            )
            mvis_data.img_sam_preview = img_sam_preview
        elif context.mask_deletion_enabled:
            img_sam_preview = mask_vis.get_mask_deletion_preview()
            mvis_data.img_sam_preview = img_sam_preview

        # TODO only compute visualizations that are currently selecetd in the UI
        masked_img = mask_vis.get_masked_img()  # masks get contours
        mask_collection = mask_vis.get_mask_collection()
        bbox_img = mask_vis.get_bbox_img()

        if (
            len(annot.masks) > context.mask_idx
            and not context.manual_annotation_enabled
            and not context.polygon_drawing_enabled
            and not context.mask_deletion_enabled
        ):
            mask_obj = annot.masks[context.mask_idx]
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
        bbox_img_cnt = mask_vis.get_bbox_img_cnt()

        if len(annot.masks) > context.mask_idx:
            annot.set_current_mask(context.mask_idx)
        mvis_data.maskinrgb = maskinrgb
        mvis_data.masked_img = masked_img
        mvis_data.mask_collection = mask_collection
        mvis_data.masked_img_cnt = masked_img_cnt
        mvis_data.mask_collection_cnt = mask_collection_cnt
        mvis_data.bbox_img = bbox_img
        mvis_data.bbox_img_cnt = bbox_img_cnt

        annot.good_masks = mask_vis.mask_objs

