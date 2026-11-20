import unittest
from unittest.mock import patch

import numpy as np

from aramsam_annotator.annotator import Annotator
from aramsam_annotator.backend.annotations import AnnotationObject, MaskData
from aramsam_annotator.backend.propagation import merge_propagated_masks
from aramsam_annotator.configs import AramsamConfigs


class PropagationVisualizationTests(unittest.TestCase):
    def setUp(self):
        loader = patch.object(AnnotationObject, '_load_img',
                              return_value=np.zeros((30, 40, 3), np.uint8))
        loader.start()
        self.addCleanup(loader.stop)
        self.annotator = Annotator(AramsamConfigs())
        self.annotator.create_new_annotation('source.png', 'destination.png')

    def mask(self, mid, x, origin='Sam2_box_propagation'):
        pixels = np.zeros((30, 40), np.uint8)
        pixels[10:15, x:x + 5] = 255
        return MaskData(mid=mid, origin=origin, mask=pixels, color_idx=mid, class_id=0)

    def render_destination(self, masks):
        target = self.annotator.next_annotation
        target.add_masks(masks, decision=True)
        target.good_masks = list(masks)
        self.annotator.update_collections(target)
        return target

    def assert_render_matches_masks(self, target):
        expected = np.zeros((30, 40, 3), np.uint8)
        occupied = np.zeros((30, 40), bool)
        for obj in target.good_masks:
            pixels = obj.mask > 0
            overlap = pixels & occupied
            expected[pixels] = 255
            expected[overlap] = (0, 0, 255)
            occupied |= pixels
        np.testing.assert_array_equal(target.mask_visualizations.mask_collection, expected)

    def test_refreshing_one_mask_does_not_draw_cached_masks_twice(self):
        for generation in (1, 2, 3):
            with self.subTest(generation=generation):
                self.annotator.create_new_annotation('source.png', f'destination{generation}.png')
                first = self.mask(1, 3, f'Sam{generation}_box_propagation')
                second = self.mask(2, 25, 'Polygon_drawing')
                target = self.render_destination([first, second])
                refreshed = self.mask(1, 3, first.origin)
                merge_propagated_masks(target, [refreshed], {first.mid})
                self.annotator.create_new_annotation(target.filepath)
                self.annotator.update_collections(target)
                self.assertEqual(len(target.good_masks), 2)
                self.assertEqual(len({obj.mid for obj in target.good_masks}), 2)
                self.assert_render_matches_masks(target)
                # Further view refreshes must also be idempotent.
                self.annotator.update_collections(target)
                self.assert_render_matches_masks(target)

    def test_refreshing_same_ids_replaces_old_geometry_in_every_view(self):
        first = self.mask(1, 3)
        target = self.render_destination([first])
        refreshed = self.mask(1, 25)
        merge_propagated_masks(target, [refreshed], {first.mid})
        self.annotator.create_new_annotation(target.filepath)
        self.annotator.update_collections(target)
        self.assert_render_matches_masks(target)
        reference = AnnotationObject('reference.png')
        reference.add_masks([refreshed], decision=True)
        reference.good_masks = [refreshed]
        self.annotator.update_collections(reference)
        for name in ('masked_img', 'bbox_img', 'mask_collection', 'instance_contours'):
            with self.subTest(view=name):
                np.testing.assert_array_equal(getattr(target.mask_visualizations, name),
                                              getattr(reference.mask_visualizations, name))

    def test_reordering_existing_masks_does_not_create_false_overlap(self):
        first, second = self.mask(1, 3), self.mask(2, 25)
        target = self.render_destination([first, second])
        target.good_masks.reverse()
        self.annotator.update_collections(target)
        self.assert_render_matches_masks(target)

    def test_replacing_mask_pixels_without_changing_the_object_refreshes_views(self):
        first = self.mask(1, 3)
        target = self.render_destination([first])
        first.mask = self.mask(1, 25).mask
        first.bbox = None
        self.annotator.update_collections(target)
        self.assert_render_matches_masks(target)

    def test_real_overlap_remains_visible_after_a_refresh(self):
        first, second = self.mask(1, 3), self.mask(2, 5)
        target = self.render_destination([first, second])
        refreshed = self.mask(1, 3)
        merge_propagated_masks(target, [refreshed], {first.mid})
        self.annotator.create_new_annotation(target.filepath)
        self.annotator.update_collections(target)
        self.assert_render_matches_masks(target)
        np.testing.assert_array_equal(target.mask_visualizations.mask_collection[12, 6], (0, 0, 255))


if __name__ == '__main__':
    unittest.main()
