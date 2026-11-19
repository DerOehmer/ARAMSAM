"""Bounding-box prompts shared by all SAM generations."""
import numpy as np
from contextlib import nullcontext

from aramsam_annotator.backend.annotations import MaskData


def excluded_track_ids(annotation):
    return (set(getattr(annotation, 'deleted_track_ids', ()))
            | set(getattr(annotation, 'suppressed_track_ids', ())))


def prune_excluded_masks(annotation):
    with getattr(annotation, '_propagation_lock', nullcontext()):
        excluded = excluded_track_ids(annotation)
        if not excluded:
            return
        retained = [(mask, decision) for mask, decision in zip(annotation.masks, annotation.mask_decisions)
                    if mask.track_id not in excluded]
        annotation.masks = [mask for mask, _ in retained]
        annotation.mask_decisions = [decision for _, decision in retained]
        annotation.good_masks = [mask for mask in annotation.good_masks if mask.track_id not in excluded]


def sync_propagation_exclusions(source, destination):
    """Replace inherited exclusions so an explicit reacceptance can propagate again."""
    with getattr(destination, '_propagation_lock', nullcontext()):
        destination.suppressed_track_ids = excluded_track_ids(source)
        prune_excluded_masks(destination)


def merge_propagated_masks(annotation, masks, source_ids, active_source_ids=None,
                           source_track_ids=(), active_track_ids=()):
    """Refresh source objects without duplicating earlier or reloaded results."""
    with getattr(annotation, '_propagation_lock', nullcontext()):
        excluded = excluded_track_ids(annotation)
        masks = [mask for mask in masks if mask.track_id not in excluded]
        _merge_propagated_masks(annotation, masks, source_ids, active_source_ids,
                                source_track_ids, active_track_ids)
        prune_excluded_masks(annotation)


def _merge_propagated_masks(annotation, masks, source_ids, active_source_ids,
                            source_track_ids, active_track_ids):
    masks = list({mask.mid: mask for mask in masks}.values())
    masks = list({mask.track_id: mask for mask in masks}.values())
    removed_ids = set(source_ids)
    refreshed_tracks = set(source_track_ids) | {mask.track_id for mask in masks}
    removed_ids.update(existing.mid for existing in annotation.masks
                       if existing.track_id in refreshed_tracks)
    active_source_ids = set(source_ids if active_source_ids is None else active_source_ids)
    active_track_ids = set(active_track_ids) | refreshed_tracks

    def match_score(existing, prediction):
        # Known identities are authoritative even when masks overlap or change
        # completely. Geometry is only a compatibility fallback for old files.
        if (not existing.track_id.startswith('legacy:')
                or existing.class_id != prediction.class_id or existing.mask is None
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
                      if existing.mid not in removed_ids and existing.mid not in active_source_ids
                      and existing.track_id not in active_track_ids]
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


def inherit_tracking_identity(predictions, sources):
    """Keep identity and annotation metadata independent of a model's output format."""
    sources = {source.mid: source for source in sources}
    for prediction in predictions:
        source = sources[prediction.mid]
        prediction.track_id = source.track_id
        prediction.class_id = source.class_id
        prediction.color_idx = source.color_idx
        prediction.time_stamp = prediction.time_stamp or 1
    return predictions


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
    sync_propagation_exclusions(current_annotation, next_annotation)
    excluded = excluded_track_ids(next_annotation)
    source_height, source_width = current_annotation.img.shape[:2]
    height, width = next_annotation.img.shape[:2]
    scale = np.array([width / source_width, height / source_height] * 2, np.float32)
    sources, boxes = [], []
    for source in masks:
        if source.track_id in excluded or source.mask is None:
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
                     class_id=source.class_id, color_idx=source.color_idx, time_stamp=1,
                     track_id=source.track_id)
            for source, mask in zip(sources, predictions) if mask is not None and np.any(mask)]
