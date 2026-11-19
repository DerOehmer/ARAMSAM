"""Bounding-box prompts shared by all SAM generations."""
import numpy as np

from aramsam_annotator.backend.annotations import MaskData


def merge_propagated_masks(annotation, masks, source_ids, active_source_ids=None):
    """Refresh source objects without duplicating earlier or reloaded results."""
    masks = list({mask.mid: mask for mask in masks}.values())
    removed_ids = set(source_ids)
    active_source_ids = set(source_ids if active_source_ids is None else active_source_ids)

    def match_score(existing, prediction):
        if (existing.class_id != prediction.class_id or existing.mask is None
                or prediction.mask is None or existing.mask.shape != prediction.mask.shape):
            return 0
        if np.array_equal(existing.mask, prediction.mask):
            return 1
        # Reloading allocates fresh IDs, and YOLO polygon export can change the
        # outline slightly. Match near-identical saved results, not manual edits
        # or objects whose identities are still present in the source image.
        if existing.origin not in ('Yolo_prediction', 'Sam1_box_propagation',
                                    'Sam2_box_propagation', 'Sam3_box_propagation',
                                    'Sam2_tracking', 'Sam3_tracking', 'Panorama_tracking'):
            return 0
        old, new = existing.mask > 0, prediction.mask > 0
        intersection = np.count_nonzero(old & new)
        union = np.count_nonzero(old) + np.count_nonzero(new) - intersection
        return intersection / union if union else 0

    for prediction in masks:
        candidates = [(match_score(existing, prediction), existing.mid)
                      for existing in annotation.masks
                      if existing.mid not in removed_ids and existing.mid not in active_source_ids]
        if candidates:
            score, mid = max(candidates, key=lambda candidate: candidate[0])
            if score >= 0.9:
                removed_ids.add(mid)

    retained = [(mask, decision) for mask, decision in zip(annotation.masks, annotation.mask_decisions)
                if mask.mid not in removed_ids]
    annotation.masks = [mask for mask, _ in retained]
    annotation.mask_decisions = [decision for _, decision in retained]
    annotation.good_masks = [mask for mask in annotation.good_masks if mask.mid not in removed_ids]
    annotation.add_masks(masks, decision=True)


def predict_image_boxes(predictor, image, boxes, image_is_set=False):
    """Use an independent SAM1/SAM2 image predictor; retain empty output slots."""
    if not image_is_set:
        predictor.set_image(np.ascontiguousarray(image[..., ::-1]))
    result = []
    for box in boxes:
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        mask = (masks[0] > 0).astype(np.uint8) * 255
        result.append(mask if np.any(mask) else None)
    return result


def propagate_boxes(sam, current_annotation, next_annotation, masks, generation):
    """Extract tight XYXY boxes, then segment their locations in the next image."""
    source_height, source_width = current_annotation.img.shape[:2]
    height, width = next_annotation.img.shape[:2]
    scale = np.array([width / source_width, height / source_height] * 2, np.float32)
    sources, boxes = [], []
    for source in masks:
        if source.mask is None:
            continue
        ys, xs = np.nonzero(source.mask)
        if not len(xs):
            continue
        # Exclusive upper bounds also give single-pixel masks a nonzero box.
        box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32) * scale
        box = np.clip(box, 0, [width, height, width, height]).astype(np.float32)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        sources.append(source)
        boxes.append(box)
    if not boxes:
        return []
    predictions = sam.predict_boxes(next_annotation.img, np.asarray(boxes, dtype=np.float32))
    if len(predictions) != len(sources):
        raise ValueError("Box predictions must return one result per prompt")
    return [MaskData(mid=source.mid, mask=mask, origin=f"Sam{generation}_box_propagation",
                     class_id=source.class_id, color_idx=source.color_idx, time_stamp=1)
            for source, mask in zip(sources, predictions) if mask is not None and np.any(mask)]
