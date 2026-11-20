import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

from aramsam_annotator.app import App
from aramsam_annotator.backend.annotations import AnnotationObject, MaskData
from aramsam_annotator.backend.storage import AnnotationRepository
from aramsam_annotator.configs import AramsamConfigs, ImgTiles, SaveData
from aramsam_annotator.tracker import PanoImageAligner


class IdentityPanoramaAligner(PanoImageAligner):
    """Use known alignment while exercising the real SAM1 propagation coordinator."""
    def match_and_align(self):
        self.tracked_sources = [obj for obj in self.matching_masks[0] if np.any(obj.mask)]
        return self._masks_to_bboxes_np([obj.mask for obj in self.tracked_sources])


class RevisitedPropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = [self.root / f'image{index}.png' for index in range(4)]
        for path in self.images:
            cv2.imwrite(str(path), np.zeros((12, 20, 3), np.uint8))

    def make_app(self, generation, method='bounding_box', style='default'):
        configs = AramsamConfigs(mask_propagation_method=method,
                                 img_tiles=ImgTiles(do_tiling=False),
                                 save_data=SaveData(mask_style=style))
        with patch('aramsam_annotator.gui.get_monitors', return_value=[]):
            app = App(configs=configs)
        self.addCleanup(app.ui.deleteLater)
        self.addCleanup(app.ui.hide)
        self.addCleanup(app.shutdown)
        app.sam_gen = generation
        app.output_dir = str(self.root / f'output-{generation}-{method}-{style}')
        Path(app.output_dir).mkdir()
        app.annotator.sam = Mock(device='cuda')
        app.embed_img = Mock()
        app.embed_img_pair = Mock()

        def predict_boxes(image, boxes):
            result = []
            for x0, y0, x1, y1 in boxes.astype(int):
                mask = np.zeros(image.shape[:2], np.uint8)
                mask[y0:y1, x0:x1] = 255
                result.append(mask)
            return result

        def track(masks):
            return [MaskData(mid=mask.mid, origin=f'Sam{generation}_tracking',
                             mask=mask.mask.copy(), class_id=mask.class_id, color_idx=mask.color_idx)
                    for mask in masks]

        app.annotator.sam.predict_boxes.side_effect = predict_boxes
        app.annotator.sam.prop_thread_func.side_effect = track
        if generation == 1 and method == 'tracking':
            aligner = patch('aramsam_annotator.controllers.inference.PanoImageAligner', IdentityPanoramaAligner)
            aligner.start()
            self.addCleanup(aligner.stop)
            app.embed_img = Mock(side_effect=lambda _: app.propose_masks())
            app.annotator.sam.transform_bboxes.side_effect = lambda boxes, shape: np.asarray(boxes, np.float32)
            app.annotator.sam.predict_batch.side_effect = lambda bboxes: np.asarray(
                predict_boxes(app.annotator.annotation.img, bboxes))[:, None] > 0
            app.annotator.sam._torch_to_npmasks.side_effect = lambda masks: [
                np.asarray(mask[0], np.uint8) * 255 for mask in masks if np.any(mask)]
        app.img_fnames = [str(path) for path in reversed(self.images)]
        app.select_next_img()
        return app

    def add_mask(self, app, class_id, x):
        mask = np.zeros((12, 20), np.uint8)
        mask[2:6, x:x + 3] = 255
        obj = MaskData(mid=app.annotator.mask_id_handler.get_id(), origin='Polygon_drawing',
                       mask=mask, class_id=class_id, color_idx=class_id, time_stamp=1)
        app.annotator.annotation.add_masks([obj], decision=True)
        app.annotator.annotation.good_masks.append(obj)
        return obj

    def assert_classes(self, annotation, expected):
        self.assertCountEqual([mask.class_id for mask in annotation.good_masks], expected)
        self.assertEqual(len(annotation.masks), len(expected))
        self.assertEqual(annotation.mask_decisions, [True] * len(expected))
        self.assertEqual(len({mask.track_id for mask in annotation.good_masks}), len(expected))

    def test_new_masks_propagate_after_going_back_two_images_in_all_model_paths(self):
        for method, generations in (('bounding_box', (1, 2, 3)), ('tracking', (1, 2, 3))):
            for generation in generations:
                for style in ('default', 'yolo'):
                    with self.subTest(generation=generation, method=method, style=style):
                        app = self.make_app(generation, method, style)
                        self.add_mask(app, class_id=3, x=1)
                        app.select_next_img()
                        self.add_mask(app, class_id=9, x=14)
                        app.select_next_img()
                        self.assert_classes(app.annotator.annotation, [3, 9])
                        app.select_previous_img()
                        app.select_previous_img()
                        self.assertEqual(app.annotator.annotation.filepath, str(self.images[0]))
                        self.add_mask(app, class_id=6, x=8)
                        app.select_next_img()
                        self.assertEqual(app.annotator.annotation.filepath, str(self.images[1]))
                        self.assert_classes(app.annotator.annotation, [3, 6, 9])
                        app.select_next_img()
                        self.assertEqual(app.annotator.annotation.filepath, str(self.images[2]))
                        self.assert_classes(app.annotator.annotation, [3, 6, 9])

    def test_next_refreshes_all_good_masks_even_if_they_were_already_batched(self):
        for method, generations in (('bounding_box', (1, 2, 3)), ('tracking', (2, 3))):
            for generation in generations:
                with self.subTest(generation=generation, method=method):
                    app = self.make_app(generation, method)
                    first = self.add_mask(app, class_id=3, x=1)
                    app.mask_track_batch_size = 1
                    app.start_mask_batch_thread()
                    first.mask[2:6, 4:6] = 255  # Update the already-propagated object.
                    second = self.add_mask(app, class_id=6, x=8)
                    app.propagated_mids.add(second.mid)  # Stale bookkeeping must not suppress Next.
                    app.annotator.sam.prop_thread_func.reset_mock()
                    app.select_next_img()
                    self.assert_classes(app.annotator.annotation, [3, 6])
                    refreshed = next(mask for mask in app.annotator.annotation.good_masks if mask.class_id == 3)
                    np.testing.assert_array_equal(refreshed.mask, first.mask)
                    model = app.annotator.sam
                    if method == 'bounding_box':
                        self.assertEqual(len(model.predict_boxes.call_args.args[1]), 2)
                    else:
                        self.assertEqual({mask.mid for call in model.prop_thread_func.call_args_list
                                          for mask in call.args[0]},
                                         {first.mid, second.mid})

    def test_revisit_replaces_changed_predictions_instead_of_adding_duplicates(self):
        for method, generations in (('bounding_box', (1, 2, 3)), ('tracking', (1, 2, 3))):
            for generation in generations:
                for style in ('default', 'yolo'):
                    with self.subTest(generation=generation, method=method, style=style):
                        app = self.make_app(generation, method, style)
                        if method == 'tracking' and generation == 1:
                            original_batch = app.annotator.sam.predict_batch.side_effect

                            def changed_panorama(bboxes):
                                return np.roll(original_batch(bboxes), int(len(bboxes) > 1), axis=3)

                            app.annotator.sam.predict_batch.side_effect = changed_panorama
                        elif method == 'tracking':
                            def changed_tracking(masks):
                                return [MaskData(mid=obj.mid, origin=f'Sam{generation}_tracking',
                                                 mask=np.roll(obj.mask, int(len(masks) > 1), axis=1))
                                        for obj in masks]
                            app.annotator.sam.prop_thread_func.side_effect = changed_tracking
                        else:
                            original_predict = app.annotator.sam.predict_boxes.side_effect

                            def changed_boxes(image, boxes):
                                return [np.roll(mask, int(len(boxes) > 1), axis=1)
                                        for mask in original_predict(image, boxes)]
                            app.annotator.sam.predict_boxes.side_effect = changed_boxes
                        self.add_mask(app, class_id=0, x=1)
                        app.select_next_img()
                        self.assert_classes(app.annotator.annotation, [0])
                        app.select_previous_img()
                        self.add_mask(app, class_id=0, x=8)
                        app.select_next_img()
                        # Both masks have the same class, and the old prediction
                        # changes by enough to defeat the previous 90% IoU rule.
                        self.assert_classes(app.annotator.annotation, [0, 0])
                        app.select_next_img()
                        self.assert_classes(app.annotator.annotation, [0, 0])

    def test_saved_destination_is_not_skipped_or_overwritten_when_advancing(self):
        app = self.make_app(2)
        destination = AnnotationObject(self.images[1])
        independent = MaskData(mid=99, origin='Polygon_drawing', class_id=9,
                               mask=np.zeros((12, 20), np.uint8))
        independent.mask[2:6, 14:17] = 255
        destination.good_masks = [independent]
        repository = AnnotationRepository(app.configs.save_data)
        repository.save(destination, app.output_dir)
        repository.save(AnnotationObject(self.images[2]), app.output_dir)
        # Reopen the pair so its destination is loaded before propagation.
        app.annotator.annotation = None
        app.annotator.next_annotation = None
        app.img_fnames = [str(path) for path in reversed(self.images)]
        app.select_next_img()
        self.assertEqual(app.annotator.annotation.filepath, str(self.images[0]))
        self.add_mask(app, class_id=3, x=1)
        app.select_next_img()
        self.assertEqual(app.annotator.annotation.filepath, str(self.images[1]))
        self.assert_classes(app.annotator.annotation, [3, 9])

    def test_switching_between_tracking_and_box_prompts_keeps_one_copy_per_object(self):
        app = self.make_app(2, 'tracking', 'yolo')
        self.add_mask(app, class_id=0, x=1)
        app.select_next_img()
        app.select_previous_img()
        app.configs.mask_propagation_method = 'bounding_box'
        self.add_mask(app, class_id=0, x=8)
        app.select_next_img()
        self.assert_classes(app.annotator.annotation, [0, 0])
        self.assertEqual(len({mask.track_id for mask in app.annotator.annotation.good_masks}), 2)
        app.select_previous_img()
        app.configs.mask_propagation_method = 'tracking'
        self.add_mask(app, class_id=0, x=14)
        app.select_next_img()
        self.assert_classes(app.annotator.annotation, [0, 0, 0])
        self.assertEqual(len({mask.track_id for mask in app.annotator.annotation.good_masks}), 3)

    def test_deleted_mask_does_not_return_from_saved_following_images(self):
        for method in ('bounding_box', 'tracking'):
            for generation in (1, 2, 3):
                for style in ('default', 'yolo'):
                    with self.subTest(generation=generation, method=method, style=style):
                        app = self.make_app(generation, method, style)
                        deleted = self.add_mask(app, class_id=0, x=1)
                        self.add_mask(app, class_id=0, x=8)
                        app.select_next_img()
                        self.add_mask(app, class_id=9, x=14)
                        app.select_next_img()
                        app.select_previous_img()
                        app.select_previous_img()
                        restored = next(mask for mask in app.annotator.annotation.good_masks
                                        if mask.track_id == deleted.track_id)
                        app.annotator.delete_mask(restored.mid)
                        self.assertNotIn(deleted.track_id,
                                         {mask.track_id for mask in app.annotator.next_annotation.masks})
                        app.select_next_img()
                        self.assert_classes(app.annotator.annotation, [0, 9])
                        app.select_next_img()
                        self.assert_classes(app.annotator.annotation, [0, 9])
                        self.assertNotIn(deleted.track_id,
                                         {mask.track_id for mask in app.annotator.annotation.good_masks})
                        loaded = AnnotationObject(self.images[1])
                        app.navigation.repository.load(loaded, app.output_dir, app.annotator.mask_id_handler)
                        self.assertIn(deleted.track_id, loaded.suppressed_track_ids)

    def test_deleting_last_mask_clears_tracking_and_does_not_restore_saved_copies(self):
        for method in ('bounding_box', 'tracking'):
            for generation in (1, 2, 3):
                for style in ('default', 'yolo'):
                    with self.subTest(generation=generation, method=method, style=style):
                        app = self.make_app(generation, method, style)
                        source = self.add_mask(app, class_id=0, x=1)
                        app.select_next_img()
                        app.select_previous_img()
                        current = app.annotator.annotation.good_masks[0]
                        app.annotator.delete_mask(current.mid)
                        model = app.annotator.sam
                        model.predict_boxes.reset_mock()
                        model.prop_thread_func.reset_mock()
                        model.predict_batch.reset_mock()
                        app.select_next_img()
                        self.assert_classes(app.annotator.annotation, [])
                        self.assertIn(source.track_id, app.annotator.annotation.suppressed_track_ids)
                        app.select_next_img()
                        self.assert_classes(app.annotator.annotation, [])
                        model.predict_boxes.assert_not_called()
                        model.prop_thread_func.assert_not_called()
                        model.predict_batch.assert_not_called()
                        if generation in (2, 3):
                            self.assertTrue(model.reset_tracking.called)

    def test_deletion_removes_results_already_propagated_in_background(self):
        for method, generations in (('bounding_box', (1, 2, 3)), ('tracking', (2, 3))):
            for generation in generations:
                with self.subTest(generation=generation, method=method):
                    app = self.make_app(generation, method)
                    deleted = self.add_mask(app, class_id=0, x=1)
                    self.add_mask(app, class_id=0, x=8)
                    app.mask_track_batch_size = 1
                    app.start_mask_batch_thread()
                    self.assertEqual(len(app.annotator.next_annotation.masks), 2)
                    app.annotator.delete_mask(deleted.mid)
                    self.assertEqual(len(app.annotator.next_annotation.masks), 1)
                    app.select_next_img()
                    self.assert_classes(app.annotator.annotation, [0])
                    self.assertNotIn(deleted.track_id,
                                     {mask.track_id for mask in app.annotator.annotation.good_masks})

    def test_explicit_reacceptance_allows_a_deleted_object_to_propagate_again(self):
        for method in ('bounding_box', 'tracking'):
            for generation in (1, 2, 3):
                with self.subTest(generation=generation, method=method):
                    app = self.make_app(generation, method)
                    source = self.add_mask(app, class_id=0, x=1)
                    app.annotator.delete_mask(source.mid)
                    app.annotator.mask_idx = 0
                    app.annotator.good_mask(class_id=0)
                    self.assertNotIn(source.track_id, app.annotator.annotation.deleted_track_ids)
                    app.select_next_img()
                    self.assert_classes(app.annotator.annotation, [0])
                    self.assertEqual(app.annotator.annotation.good_masks[0].track_id, source.track_id)


if __name__ == '__main__':
    unittest.main()
