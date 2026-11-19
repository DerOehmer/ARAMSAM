"""Names shared by configuration, the layout form, and control-image exports."""

VISUALIZATION_LABELS = {
    'img': 'Original image',
    'img_sam_preview': 'Interactive annotation preview',
    'mask': 'Current mask',
    'maskinrgb': 'Current masked object',
    'masked_img': 'Filled masks with instance IDs',
    'mask_collection': 'Mask collection and overlaps',
    'masked_img_cnt': 'Filled masks with current contour',
    'mask_collection_cnt': 'Mask collection with current contour',
    'bbox_img': 'Bounding boxes',
    'bbox_img_cnt': 'Bounding boxes with current contour',
    'instance_contours': 'Instance contours with ID circles',
}

CONTROL_IMAGE_OPTIONS = {
    'auto': 'Automatic (filled masks or bounding boxes)',
    **{key: VISUALIZATION_LABELS[key] for key in (
        'img', 'masked_img', 'bbox_img', 'mask_collection', 'instance_contours',
    )},
}


def default_layout(bbox_mode=False):
    """Panel order follows the UI: top left, bottom left, top right, bottom right."""
    return ['img', 'img_sam_preview',
            'bbox_img_cnt' if bbox_mode else 'masked_img_cnt', 'mask_collection_cnt']
