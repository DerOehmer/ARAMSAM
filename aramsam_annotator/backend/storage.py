"""Annotation persistence, independent of Qt and model inference.

Existing image/mask/log and YOLO layouts are preserved. A metadata sidecar in
native exports retains class IDs and origins when an annotation is reloaded.
"""
import json
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from aramsam_annotator.backend.annotations import MaskData, legacy_track_id
from aramsam_annotator.backend.propagation import prune_excluded_masks

ORIGIN_CODES = {
    "Sam1_proposed": "s1p", "Sam2_proposed": "s2p",
    "Sam1_interactive": "s1i", "Sam2_interactive": "s2i",
    "Polygon_drawing": "plg", "Sam2_tracking": "s2t",
    "Panorama_tracking": "pat", "Yolo_prediction": "yol",
    "Sam3_proposed": "s3p", "Sam3_interactive": "s3i",
    "Sam3_tracking": "s3t", "Sam3_text": "s3x",
    "Sam1_box_propagation": "s1b", "Sam2_box_propagation": "s2b",
    "Sam3_box_propagation": "s3b",
}


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported annotation metadata type: {type(value).__name__}")


def write_image(path, image):
    """Fail explicitly instead of reporting success after an OpenCV write failure."""
    if image is None or not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def atomic_text(path, text):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class AnnotationRepository:
    def __init__(self, settings, origin_codes=None):
        self.settings = settings
        self.origin_codes = ORIGIN_CODES if origin_codes is None else origin_codes

    @property
    def yolo(self):
        return (self.settings.save_masks and self.settings.mask_style == 'yolo'
                or self.settings.save_bboxes and self.settings.bbox_style == 'yolo')

    def exists(self, directory, image):
        if image is None:
            return True
        if directory is None:
            return False
        directory, image = Path(directory), Path(image)
        if self.yolo:
            return ((directory / 'images' / image.name).is_file()
                    and (directory / 'labels' / f'{image.stem}.txt').is_file())
        return (directory / f'{image.stem}_annots' / 'log.json').is_file()

    def save(self, annotation, directory, suffix=None):
        if annotation is None or not self.settings.do_save:
            return False
        if self.settings.save_masks:
            if self.settings.mask_style == 'default':
                return self.save_native(annotation, directory, suffix)
            if self.settings.mask_style == 'yolo':
                return self.save_yolo(annotation, directory, segmentation=True)
        elif self.settings.save_bboxes and self.settings.bbox_style == 'yolo':
            return self.save_yolo(annotation, directory, segmentation=False)
        raise NotImplementedError('Unsupported annotation output format')

    def log_masks(self, masks, directory=None):
        counts = dict.fromkeys(self.origin_codes, 0)
        latest = 0
        for index, mask in enumerate(masks):
            origin = mask.origin or 'Polygon_drawing'
            if origin not in self.origin_codes:
                raise ValueError(f'Origin code not found for {origin}')
            counts[origin] += 1
            if directory is not None and mask.mask is not None:
                timestamp = mask.time_stamp or 0
                latest = max(latest, timestamp)
                name = f'mask_{self.origin_codes[origin]}_{timestamp}_{index}.png'
                write_image(Path(directory) / name, mask.mask)
        counts['Total_masks'] = len(masks)
        return counts, latest if directory is not None else None

    def control_image(self, annotation, segmentation=True):
        """Render accepted annotations freshly, independently of UI previews or layout."""
        from aramsam_annotator.mask_visualizations import MaskVisualization

        selection = self.settings.control_image_visualization
        if selection == 'auto':
            selection = 'masked_img' if segmentation else 'bbox_img'
        if selection == 'img':
            return annotation.img
        visualizer = MaskVisualization()
        # Empty predictions are omitted from segmentation labels and have no
        # centroid to draw in a control image.
        drawable = [obj for obj in annotation.good_masks
                    if (obj.mask is not None and np.any(obj.mask))
                    or (obj.mask is None and obj.bbox is not None)]
        visualizer.set_annotation(img=annotation.img, mask_objs=drawable)
        renderers = {
            'masked_img': visualizer.get_masked_img,
            'bbox_img': visualizer.get_bbox_img,
            'mask_collection': visualizer.get_mask_collection,
            'instance_contours': visualizer.get_instance_contours,
        }
        return renderers[selection]()

    @staticmethod
    def _propagation_state(annotation):
        return {'deleted_track_ids': sorted(annotation.deleted_track_ids),
                'suppressed_track_ids': sorted(annotation.suppressed_track_ids)}

    @staticmethod
    def _restore_propagation_state(annotation, state):
        for name in ('deleted_track_ids', 'suppressed_track_ids'):
            values = state.get(name, []) if isinstance(state, dict) else []
            setattr(annotation, name, {value for value in values if isinstance(value, str)}
                    if isinstance(values, list) else set())

    def save_native(self, annotation, directory, suffix=None):
        if annotation is None:
            return False
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        name = f'{Path(annotation.img_name).stem}_annots'
        if suffix is not None:
            name += f'_{suffix}'
        destination = directory / name
        existed = destination.exists()
        # Build the complete replacement first. Existing masks survive write failures.
        with tempfile.TemporaryDirectory(dir=directory, prefix=f'.{name}.') as work:
            staged = Path(work) / 'new'
            staged.mkdir()
            masks_dir = staged / 'masks'
            masks_dir.mkdir()
            write_image(staged / 'img.jpg', annotation.img)
            write_image(staged / 'annotations.jpg', self.control_image(annotation))
            selected, total_time = self.log_masks(annotation.good_masks, masks_dir)
            all_masks, _ = self.log_masks(annotation.masks)
            (staged / 'log.json').write_text(json.dumps({
                'All_masks': all_masks, 'Selected_masks': selected, 'Total_time': total_time,
            }, indent=4, default=json_value), encoding='utf-8')
            metadata = []
            for index, mask in enumerate(annotation.good_masks):
                origin = mask.origin or 'Polygon_drawing'
                metadata.append({
                    'file': (f'mask_{self.origin_codes[origin]}_{mask.time_stamp or 0}_{index}.png'
                             if mask.mask is not None else None),
                    'origin': origin, 'time_stamp': mask.time_stamp,
                    'class_id': mask.class_id, 'color_idx': mask.color_idx,
                    'track_id': mask.track_id,
                    'bbox': list(mask.bbox) if mask.bbox is not None else None,
                })
            (staged / 'annotations.json').write_text(json.dumps(metadata, indent=2, default=json_value), encoding='utf-8')
            (staged / 'propagation.json').write_text(json.dumps(self._propagation_state(annotation)),
                                                   encoding='utf-8')
            backup = Path(work) / 'old'
            if existed:
                os.replace(destination, backup)
            try:
                os.replace(staged, destination)
            except BaseException:
                if existed:
                    os.replace(backup, destination)
                raise
        return existed

    def save_yolo(self, annotation, directory, segmentation):
        directory = Path(directory)
        for folder in ('images', 'labels', 'control_images', 'metadata'):
            (directory / folder).mkdir(parents=True, exist_ok=True)
        height, width = annotation.img.shape[:2]
        lines = []
        instances = []
        for obj in annotation.good_masks:
            class_id = obj.class_id if obj.class_id is not None else 0
            if segmentation:
                if obj.mask is None:
                    continue
                contours, _ = cv2.findContours(obj.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)
                contour = cv2.approxPolyDP(contour, .002 * cv2.arcLength(contour, True), True)
                if len(contour) < 3:
                    continue
                values = np.clip(contour.reshape(-1, 2) / [width, height], 0, 1).ravel()
            else:
                if obj.bbox is None:
                    continue
                x0, y0, x1, y1 = obj.bbox
                values = ((x0 + x1) / (2 * width), (y0 + y1) / (2 * height),
                          (x1 - x0) / width, (y1 - y0) / height)
            lines.append(f'{class_id} ' + ' '.join(f'{v:.6f}' for v in values))
            instances.append({'track_id': obj.track_id, 'origin': obj.origin,
                              'color_idx': obj.color_idx, 'time_stamp': obj.time_stamp})
        stem = Path(annotation.img_name).stem
        label_text = '\n'.join(lines)
        metadata = {'version': 1, 'labels_sha256': hashlib.sha256(label_text.encode('utf-8')).hexdigest(),
                    'instances': instances, **self._propagation_state(annotation)}
        # Encode before replacing any output, and commit the label file last.
        with tempfile.TemporaryDirectory(dir=directory, prefix='.yolo.') as work:
            image_path = Path(work) / annotation.img_name
            # Annotation/model images are downsampled. The dataset must retain
            # the original file (or the generated tile), including its size.
            shutil.copyfile(annotation.filepath, image_path)
            control = self.control_image(annotation, segmentation)
            control_path = Path(work) / 'control.jpg'
            write_image(control_path, annotation.img if control is None else control)
            os.replace(image_path, directory / 'images' / annotation.img_name)
            os.replace(control_path, directory / 'control_images' / f'{stem}_control.jpg')
            atomic_text(directory / 'metadata' / f'{stem}.json', json.dumps(metadata, default=json_value))
            atomic_text(directory / 'labels' / f'{stem}.txt', label_text)
        return False

    def load(self, annotation, directory, id_handler):
        """Load accepted objects; allocate fresh IDs to avoid collisions in a session."""
        directory = Path(directory)
        self._restore_propagation_state(annotation, {})
        if self.yolo:
            masks = self._load_yolo(annotation, directory, id_handler)
        else:
            native = directory / f'{Path(annotation.img_name).stem}_annots'
            state_path = native / 'propagation.json'
            if state_path.is_file():
                self._restore_propagation_state(annotation, json.loads(state_path.read_text(encoding='utf-8')))
            metadata = native / 'annotations.json'
            if not metadata.exists():
                annotation.load_masks_from_dir(native / 'masks', id_handler)
                annotation.proposals_ready = True
                prune_excluded_masks(annotation)
                return
            masks = []
            for index, record in enumerate(json.loads(metadata.read_text(encoding='utf-8'))):
                filename = record.pop('file')
                if not record.get('track_id'):
                    record['track_id'] = legacy_track_id(annotation.filepath, filename or index)
                mask = None
                if filename is not None:
                    if Path(filename).name != filename:
                        raise ValueError('Invalid mask filename in annotation metadata')
                    mask = cv2.imread(str(native / 'masks' / filename), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise OSError(f'Could not read mask: {filename}')
                masks.append(MaskData(mid=id_handler.get_id(), mask=mask, **record))
        annotation.proposals_ready = True
        annotation.good_masks = masks
        annotation.masks = list(masks)
        annotation.mask_decisions = [True] * len(masks)
        prune_excluded_masks(annotation)

    def _load_yolo(self, annotation, directory, id_handler):
        path = directory / 'labels' / f'{Path(annotation.img_name).stem}.txt'
        label_text = path.read_text(encoding='utf-8')
        lines = [line for line in label_text.splitlines() if line.strip()]
        records = []
        metadata_path = directory / 'metadata' / f'{Path(annotation.img_name).stem}.json'
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                if (metadata.get('version') == 1 and metadata.get('labels_sha256')
                        == hashlib.sha256(label_text.encode('utf-8')).hexdigest()
                        and isinstance(metadata.get('instances'), list)
                        and len(metadata['instances']) == len(lines)):
                    records = metadata['instances']
                    self._restore_propagation_state(annotation, metadata)
            except (OSError, ValueError, TypeError, AttributeError):
                # A missing/stale sidecar must not prevent importing ordinary YOLO labels.
                pass
        height, width = annotation.img.shape[:2]
        masks = []
        for index, line in enumerate(lines):
            parts = line.split()
            class_id, coords = int(parts[0]), np.asarray(parts[1:], dtype=float)
            if not np.all(np.isfinite(coords)) or np.any(coords < 0) or np.any(coords > 1):
                raise ValueError(f'Invalid normalized coordinates in {path}')
            record = records[index] if records and isinstance(records[index], dict) else {}
            track_id = record.get('track_id')
            if not isinstance(track_id, str) or not track_id:
                track_id = legacy_track_id(annotation.filepath, index)
            origin = record.get('origin', 'Yolo_prediction')
            if not isinstance(origin, str) or origin not in self.origin_codes:
                origin = 'Yolo_prediction'
            obj = MaskData(mid=id_handler.get_id(), origin=origin, class_id=class_id,
                           track_id=track_id, color_idx=record.get('color_idx'),
                           time_stamp=record.get('time_stamp'))
            if self.settings.save_masks:
                if len(coords) < 6 or len(coords) % 2:
                    raise ValueError(f'Invalid YOLO polygon in {path}')
                points = np.rint(coords.reshape(-1, 2) * [width, height]).astype(np.int32)
                obj.mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(obj.mask, [points], 255)
            else:
                if len(coords) != 4:
                    raise ValueError(f'Invalid YOLO box in {path}')
                x, y, w, h = coords * [width, height, width, height]
                obj.bbox = tuple(int(round(v)) for v in (x-w/2, y-h/2, x+w/2, y+h/2))
            masks.append(obj)
        return masks
