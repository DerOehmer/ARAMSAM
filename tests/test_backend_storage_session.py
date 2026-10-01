import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from aramsam_annotator.annotator import Annotator
from aramsam_annotator.backend.annotations import AnnotationObject, MaskData, MaskIdHandler
from aramsam_annotator.backend.session import ImageQueue
from aramsam_annotator.backend.storage import AnnotationRepository
from aramsam_annotator.configs import AramsamConfigs, ImgTiles, SaveData
from aramsam_annotator.img_tiling import split_image_into_tiles
from aramsam_annotator.tracker import PanoImageAligner


class BackendStorageSessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / 'image.png'
        cv2.imwrite(str(self.image), np.zeros((32, 48, 3), np.uint8))
        self.annotation = AnnotationObject(self.image)
        mask = np.zeros((32, 48), np.uint8)
        mask[3:15, 4:20] = 255
        self.obj = MaskData(mid=1, origin='Polygon_drawing', mask=mask,
                            class_id=3, time_stamp=12, color_idx=2)
        self.annotation.good_masks = [self.obj]
        self.annotation.add_masks([self.obj], decision=True)
        self.output = self.root / 'output'

    def native_repository(self):
        return AnnotationRepository(SaveData(mask_style='default'))

    def test_native_round_trip_keeps_masks_classes_and_origins(self):
        repository = self.native_repository()
        self.assertFalse(repository.save(self.annotation, self.output))
        restored = AnnotationObject(self.image)
        repository.load(restored, self.output, MaskIdHandler())
        np.testing.assert_array_equal(restored.good_masks[0].mask, self.obj.mask)
        self.assertEqual(restored.good_masks[0].class_id, 3)
        self.assertEqual(restored.good_masks[0].origin, self.obj.origin)
        self.assertEqual(restored.good_masks[0].time_stamp, 12)
        self.assertEqual(restored.mask_decisions, [True])
        self.assertTrue(repository.exists(self.output, self.image))

    def test_resaving_replaces_masks_and_deletions(self):
        repository = self.native_repository()
        repository.save(self.annotation, self.output)
        self.annotation.good_masks.clear()
        self.assertTrue(repository.save(self.annotation, self.output))
        folder = self.output / 'image_annots'
        self.assertEqual(list((folder / 'masks').glob('*.png')), [])
        self.assertEqual(json.loads((folder / 'log.json').read_text())['Selected_masks']['Total_masks'], 0)

    def test_failed_resave_preserves_previous_complete_output(self):
        repository = self.native_repository()
        repository.save(self.annotation, self.output)
        original = (self.output / 'image_annots' / 'log.json').read_bytes()
        with patch('aramsam_annotator.backend.storage.cv2.imwrite', return_value=False):
            with self.assertRaises(OSError):
                repository.save(self.annotation, self.output)
        self.assertEqual((self.output / 'image_annots' / 'log.json').read_bytes(), original)
        self.assertEqual(len(list((self.output / 'image_annots' / 'masks').glob('*.png'))), 1)

    def test_legacy_mask_directory_can_be_loaded_and_resaved(self):
        repository = self.native_repository()
        repository.save(self.annotation, self.output)
        (self.output / 'image_annots' / 'annotations.json').unlink()
        restored = AnnotationObject(self.image)
        repository.load(restored, self.output, MaskIdHandler())
        self.assertEqual(restored.good_masks[0].origin, 'Polygon_drawing')
        repository.save(restored, self.output)

    def test_yolo_mask_and_bbox_round_trip(self):
        for segmentation in (True, False):
            with self.subTest(segmentation=segmentation):
                settings = SaveData(save_masks=segmentation, save_bboxes=not segmentation, bbox_style='yolo')
                repository = AnnotationRepository(settings)
                self.obj.bbox = (4, 3, 20, 15)
                repository.save(self.annotation, self.output)
                restored = AnnotationObject(self.image)
                repository.load(restored, self.output, MaskIdHandler())
                self.assertEqual(restored.good_masks[0].class_id, 3)
                if segmentation:
                    np.testing.assert_array_equal(restored.good_masks[0].mask, self.obj.mask)
                else:
                    self.assertEqual(restored.good_masks[0].bbox, self.obj.bbox)
                self.assertTrue(repository.exists(self.output, self.image))
                (self.output / 'labels' / 'image.txt').unlink()
                self.assertFalse(repository.exists(self.output, self.image))

    def test_invalid_image_and_empty_mask_selection(self):
        with self.assertRaises(OSError):
            AnnotationObject(self.root / 'absent.png')
        empty = AnnotationObject(self.image)
        empty.set_current_mask(0)
        self.assertIsNone(empty.mask_visualizations.mask)

    def test_prefetch_requires_matching_full_path(self):
        annotator = Annotator(AramsamConfigs())
        other = self.root / 'other'
        other.mkdir()
        second = other / self.image.name
        cv2.imwrite(str(second), self.annotation.img)
        annotator.create_new_annotation(self.image, second)
        prefetched = annotator.next_annotation
        prefetched.features = object()
        embed, _ = annotator.create_new_annotation(second)
        self.assertFalse(embed)
        self.assertIs(annotator.annotation, prefetched)
        annotator.next_annotation = self.annotation
        annotator.create_new_annotation(second)
        self.assertIs(annotator.annotation, prefetched)
        self.assertIsNone(annotator.next_annotation)

    def test_failed_pair_transition_preserves_previous_annotation(self):
        annotator = Annotator(AramsamConfigs())
        annotator.create_new_annotation(self.image)
        current = annotator.annotation
        with self.assertRaises(OSError):
            annotator.create_new_annotation(self.image, self.root / 'missing.png')
        self.assertIs(annotator.annotation, current)

    def test_queue_preserves_order_and_rolls_back_failed_tiling(self):
        paths = ['first.png', 'second.png']
        queue = ImageQueue(paths, self.root, ImgTiles(do_tiling=False))
        self.assertEqual(queue.pop_pair(), (Path('second.png'), Path('first.png')))
        self.assertEqual(paths, ['first.png'])
        failing = ImageQueue(paths, self.root, ImgTiles())
        with self.assertRaises(OSError):
            failing.pop_pair()
        self.assertEqual(paths, ['first.png'])

    def test_tiles_cover_edges_and_validate_settings(self):
        tile_dir = self.root / 'tiles'
        tile_dir.mkdir()
        tiles = split_image_into_tiles(str(self.image), str(tile_dir), ImgTiles(tile_size=20))
        self.assertEqual(len(tiles), 6)
        self.assertTrue(all(cv2.imread(path).shape[:2] == (20, 20) for path in tiles))
        for config in (ImgTiles(tile_size=0), ImgTiles(tile_overlap=1)):
            with self.assertRaises(ValueError):
                split_image_into_tiles(str(self.image), str(tile_dir), config)

    def test_auto_selection_is_iterative_and_handles_empty_masks(self):
        annotator = Annotator(AramsamConfigs())
        annotator.create_new_annotation(self.image)
        annotator.update_collections = Mock()
        boxes = [MaskData(mid=i, origin='Yolo_prediction', bbox=(1, 1, 5, 5), class_id=7)
                 for i in range(1500)]
        annotator.annotation.add_masks(boxes)
        annotator.preselect_mask()
        self.assertEqual(annotator.mask_idx, 1500)
        self.assertTrue(all(obj.class_id == 7 for obj in annotator.annotation.good_masks))
        annotator.annotation.add_masks([MaskData(mid=1500, origin='Sam1_proposed',
                                                mask=np.zeros((32, 48), np.uint8))])
        annotator.preselect_mask()
        self.assertEqual(annotator.mask_idx, 1501)

    def test_tracker_handles_images_without_features(self):
        tracker = PanoImageAligner()
        tracker.add_annotation(self.annotation)
        self.assertEqual(tracker.track(AnnotationObject(self.image)), [])


if __name__ == '__main__':
    unittest.main()
