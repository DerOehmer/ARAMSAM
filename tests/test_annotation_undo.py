import unittest
from unittest.mock import Mock, patch

import numpy as np

from aramsam_annotator.annotator import Annotator
from aramsam_annotator.configs import AramsamConfigs
from aramsam_annotator.mask_visualizations import AnnotationObject, MaskData


class AnnotationUndoTests(unittest.TestCase):
    def setUp(self):
        self.loader = patch.object(AnnotationObject, '_load_img', return_value=np.zeros((16, 16, 3), dtype=np.uint8))
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.annotator = Annotator(AramsamConfigs())
        self.annotator.update_collections = Mock()
        self.annotator.create_new_annotation('first.jpg')

    def polygon(self):
        a = self.annotator
        a.polygon_drawing_enabled = True
        a.annotation.preview_mask = np.ones((16, 16), dtype=np.uint8) * 255
        a.good_mask(class_id=2)

    def test_new_image_polygon_can_be_undone_with_or_without_prefetch(self):
        for prefetched in (False, True):
            with self.subTest(prefetched=prefetched):
                a = self.annotator
                a.create_new_annotation('first.jpg')
                self.polygon()
                self.polygon()
                if prefetched:
                    a.next_annotation = AnnotationObject('second.jpg')
                a.create_new_annotation('second.jpg')
                self.assertEqual(a.mask_idx, 0)
                self.polygon()
                a.step_back()
                self.assertEqual(a.mask_idx, 0)
                self.assertEqual(a.annotation.mask_decisions, [False])
                self.assertEqual(a.annotation.good_masks, [])
                a.step_back()
                self.assertEqual(a.mask_idx, 0)

    def test_repeated_polygon_undo_and_new_polygon_keep_history_aligned(self):
        a = self.annotator
        self.polygon()
        self.polygon()
        a.step_back()
        self.polygon()
        self.assertEqual(len(a.annotation.masks), len(a.annotation.mask_decisions))
        self.assertEqual(len(a.annotation.good_masks), 2)
        a.step_back()
        a.step_back()
        self.assertEqual(a.annotation.good_masks, [])
        self.assertEqual(a.mask_idx, 0)

    def test_unfinished_polygon_is_cleared_before_undoing_saved_polygon(self):
        a = self.annotator
        self.polygon()
        a.manual_mask_points = [(1, 1), (4, 4)]
        a.annotation.preview_mask = np.ones((16, 16), dtype=np.uint8)
        a.step_back()
        self.assertEqual(a.manual_mask_points, [])
        self.assertIsNone(a.annotation.preview_mask)
        self.assertEqual(len(a.annotation.good_masks), 1)
        a.step_back()
        self.assertEqual(a.annotation.good_masks, [])

    def test_rejected_mask_can_be_undone_without_any_good_masks(self):
        a = self.annotator
        mask = MaskData(mid=10, origin='Sam1_proposed', center=(3, 4))
        a.annotation.add_masks([mask])
        a.bad_mask()
        self.assertEqual(a.step_back(), (3, 4))
        self.assertEqual(a.mask_idx, 0)

    def test_mode_guard_checks_previous_decision_not_last_accepted_mask(self):
        a = self.annotator
        self.polygon()
        a.polygon_drawing_enabled = False
        a.annotation.add_masks([MaskData(mid=10, origin='Sam1_proposed')])
        a.bad_mask()
        a.polygon_drawing_enabled = True
        a.step_back()
        self.assertEqual(a.mask_idx, 2)
        self.assertEqual(len(a.annotation.good_masks), 1)
        a.polygon_drawing_enabled = False
        a.step_back()
        a.step_back()
        self.assertEqual(a.annotation.good_masks, [])

    def test_stale_index_is_recovered_without_crashing(self):
        a = self.annotator
        self.polygon()
        a.mask_idx = 15
        a.step_back()
        self.assertEqual(a.mask_idx, 0)
        self.assertEqual(a.annotation.good_masks, [])

    def test_undo_removes_matching_accepted_mask_and_preserves_metadata(self):
        a = self.annotator
        original = MaskData(mid=10, origin='Sam1_proposed')
        accepted = MaskData(mid=10, origin='Sam1_proposed', center=(3, 4), color_idx=2)
        unrelated = MaskData(mid=11, origin='Sam1_proposed')
        a.annotation.add_masks([original], decision=True)
        a.annotation.good_masks = [accepted, unrelated]
        a.mask_idx = 1
        self.assertEqual(a.step_back(), (3, 4))
        self.assertEqual(a.annotation.good_masks, [unrelated])
        self.assertEqual(original.color_idx, 2)


if __name__ == '__main__':
    unittest.main()
