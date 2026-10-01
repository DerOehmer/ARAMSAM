"""Annotation persistence, independent of Qt and model inference.

Existing image/mask/log and YOLO layouts are preserved. A metadata sidecar in
native exports retains class IDs and origins when an annotation is reloaded.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from aramsam_annotator.backend.annotations import MaskData

ORIGIN_CODES = {
    "Sam1_proposed": "s1p", "Sam2_proposed": "s2p",
    "Sam1_interactive": "s1i", "Sam2_interactive": "s2i",
    "Polygon_drawing": "plg", "Sam2_tracking": "s2t",
    "Panorama_tracking": "pat", "Yolo_prediction": "yol",
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
            if annotation.mask_visualizations.masked_img is not None:
                write_image(staged / 'annotations.jpg', annotation.mask_visualizations.masked_img)
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
                    'bbox': list(mask.bbox) if mask.bbox is not None else None,
                })
            (staged / 'annotations.json').write_text(json.dumps(metadata, indent=2, default=json_value), encoding='utf-8')
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
        for folder in ('images', 'labels', 'control_images'):
            (directory / folder).mkdir(parents=True, exist_ok=True)
        height, width = annotation.img.shape[:2]
        lines = []
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
        stem = Path(annotation.img_name).stem
        # Encode before replacing any output, and commit the label file last.
        with tempfile.TemporaryDirectory(dir=directory, prefix='.yolo.') as work:
            image_path = Path(work) / annotation.img_name
            write_image(image_path, annotation.img)
            control = (annotation.mask_visualizations.masked_img if segmentation
                       else annotation.mask_visualizations.bbox_img)
            control_path = Path(work) / 'control.jpg'
            write_image(control_path, annotation.img if control is None else control)
            os.replace(image_path, directory / 'images' / annotation.img_name)
            os.replace(control_path, directory / 'control_images' / f'{stem}_control.jpg')
            atomic_text(directory / 'labels' / f'{stem}.txt', '\n'.join(lines))
        return False

    def load(self, annotation, directory, id_handler):
        """Load accepted objects; allocate fresh IDs to avoid collisions in a session."""
        directory = Path(directory)
        if self.yolo:
            masks = self._load_yolo(annotation, directory, id_handler)
        else:
            native = directory / f'{Path(annotation.img_name).stem}_annots'
            metadata = native / 'annotations.json'
            if not metadata.exists():
                annotation.load_masks_from_dir(native / 'masks', id_handler)
                annotation.proposals_ready = True
                return
            masks = []
            for record in json.loads(metadata.read_text(encoding='utf-8')):
                filename = record.pop('file')
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

    def _load_yolo(self, annotation, directory, id_handler):
        path = directory / 'labels' / f'{Path(annotation.img_name).stem}.txt'
        height, width = annotation.img.shape[:2]
        masks = []
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            parts = line.split()
            class_id, coords = int(parts[0]), np.asarray(parts[1:], dtype=float)
            if not np.all(np.isfinite(coords)) or np.any(coords < 0) or np.any(coords > 1):
                raise ValueError(f'Invalid normalized coordinates in {path}')
            obj = MaskData(mid=id_handler.get_id(), origin='Yolo_prediction', class_id=class_id)
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
