import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QThread
from PyQt6.QtWidgets import QApplication

from aramsam_annotator.app import App
from aramsam_annotator.backend.annotations import AnnotationObject, MaskData, MaskIdHandler
from aramsam_annotator.backend.propagation import propagate_boxes, merge_propagated_masks
from aramsam_annotator.backend.storage import AnnotationRepository
from aramsam_annotator.configs import (
    AramsamConfigs, ImgTiles, SaveData, load_configs_from_yaml, save_configs_to_yaml,
)
from aramsam_annotator.run_sam import SamInference, Sam2Inference
from aramsam_annotator.run_sam3 import Sam3Inference
from aramsam_annotator.welcome import WelcomeWindow
from aramsam_annotator.workers import BoxPropagationWorker


def source_mask(mid=7):
    mask = np.zeros((10, 20), np.uint8)
    mask[2:6, 3:9] = 255
    return MaskData(mid=mid, mask=mask, origin='Polygon_drawing', class_id=4, color_idx=2,
                    bbox=(0, 0, 1, 1))  # Extraction must use mask pixels, not a stale box.


class BoxPropagationTests(unittest.TestCase):
    def merge_target(self, masks):
        with patch.object(AnnotationObject, '_load_img', return_value=np.zeros((10, 20, 3), np.uint8)):
            target = AnnotationObject('destination.png')
        target.add_masks(masks, decision=True)
        target.good_masks = list(masks)
        return target

    def test_merge_replaces_reloaded_masks_with_slightly_different_outlines(self):
        for origin in ('Yolo_prediction', 'Sam1_box_propagation', 'Sam2_box_propagation',
                       'Sam3_box_propagation', 'Sam2_tracking', 'Sam3_tracking'):
            with self.subTest(origin=origin):
                saved = source_mask(99)
                saved.origin = origin
                target = self.merge_target([saved])
                predicted = source_mask(7)
                predicted.mask[2, 3] = 0
                merge_propagated_masks(target, [predicted], {7})
                self.assertEqual([mask.mid for mask in target.masks], [7])
                self.assertEqual(target.mask_decisions, [True])
                self.assertEqual(target.good_masks, [])
                np.testing.assert_array_equal(target.masks[0].mask, predicted.mask)

    def test_merge_keeps_distinct_objects_and_classes_and_manual_edits(self):
        for change in ('different_class', 'manual_edit', 'partly_overlapping'):
            with self.subTest(change=change):
                saved = source_mask(99)
                saved.origin = 'Yolo_prediction'
                saved.mask[2, 3] = 0
                if change == 'different_class':
                    saved.class_id = 9
                elif change == 'manual_edit':
                    saved.origin = 'Polygon_drawing'
                else:
                    saved.mask[:, 3:6] = 0
                target = self.merge_target([saved])
                predicted = source_mask(7)
                merge_propagated_masks(target, [predicted], {7})
                self.assertEqual([mask.mid for mask in target.masks], [99, 7])
                self.assertEqual(target.good_masks, [saved])

    def test_merge_matches_saved_masks_once_and_deduplicates_repeated_source_ids(self):
        saved = source_mask(99)
        saved.origin = 'Yolo_prediction'
        saved.mask[2, 3] = 0
        target = self.merge_target([saved])
        first, second = source_mask(7), source_mask(8)
        second.mask[2, 4] = 0
        merge_propagated_masks(target, [first, second], {7, 8})
        self.assertEqual([mask.mid for mask in target.masks], [7, 8])
        replacement = source_mask(7)
        merge_propagated_masks(target, [first, replacement], {7})
        self.assertEqual([mask.mid for mask in target.masks], [8, 7])
        self.assertIs(target.masks[-1], replacement)

    def test_merge_preserves_other_active_source_objects_even_if_masks_overlap(self):
        existing = source_mask(8)
        existing.origin = 'Sam2_box_propagation'
        target = self.merge_target([existing])
        predicted = source_mask(7)
        merge_propagated_masks(target, [predicted], {7}, active_source_ids={7, 8})
        self.assertEqual([mask.mid for mask in target.masks], [8, 7])
        self.assertEqual(target.good_masks, [existing])

    def test_merge_handles_actual_yolo_polygon_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'circle.png'
            cv2.imwrite(str(image), np.zeros((128, 128, 3), np.uint8))
            pixels = np.zeros((128, 128), np.uint8)
            cv2.circle(pixels, (64, 64), 40, 255, -1)
            prediction = MaskData(mid=7, origin='Sam2_box_propagation', mask=pixels, class_id=4)
            saved = AnnotationObject(image)
            saved.good_masks = [prediction]
            repository = AnnotationRepository(SaveData(mask_style='yolo'))
            repository.save(saved, root / 'output')
            target = AnnotationObject(image)
            repository.load(target, root / 'output', MaskIdHandler())
            self.assertNotEqual(target.masks[0].mid, prediction.mid)
            self.assertFalse(np.array_equal(target.masks[0].mask, prediction.mask))
            merge_propagated_masks(target, [prediction], {7})
            self.assertEqual([mask.mid for mask in target.masks], [7])
            np.testing.assert_array_equal(target.masks[0].mask, pixels)

    def test_extracts_and_scales_boxes_and_preserves_metadata_for_all_models(self):
        for generation in (1, 2, 3):
            with self.subTest(generation=generation):
                source = source_mask()
                current = SimpleNamespace(img=np.zeros((10, 20, 3), np.uint8))
                following = SimpleNamespace(img=np.zeros((20, 10, 3), np.uint8))
                predicted = np.ones((20, 10), np.uint8) * 255
                model = Mock()
                model.predict_boxes.return_value = [predicted]
                result = propagate_boxes(model, current, following, [source], generation)
                args = model.predict_boxes.call_args.args
                self.assertIs(args[0], following.img)
                np.testing.assert_array_equal(args[1], [[1.5, 4, 4.5, 12]])
                self.assertEqual(args[1].dtype, np.float32)
                self.assertEqual(len(result), 1)
                self.assertIs(result[0].mask, predicted)
                self.assertEqual((result[0].mid, result[0].class_id, result[0].color_idx), (7, 4, 2))
                self.assertEqual(result[0].origin, f'Sam{generation}_box_propagation')

    def test_empty_masks_and_predictions_do_not_shift_object_ids(self):
        current = SimpleNamespace(img=np.zeros((10, 20, 3), np.uint8))
        empty = MaskData(mid=1, origin='Polygon_drawing', mask=np.zeros((10, 20), np.uint8))
        no_mask = MaskData(mid=2, origin='Yolo_prediction', bbox=(1, 2, 3, 4))
        first, second = source_mask(7), source_mask(8)
        model = Mock()
        self.assertEqual(propagate_boxes(model, current, current, [empty, no_mask], 1), [])
        model.predict_boxes.assert_not_called()
        model.predict_boxes.return_value = [None, second.mask]
        result = propagate_boxes(model, current, current, [empty, first, no_mask, second], 1)
        self.assertEqual([mask.mid for mask in result], [8])

    def test_single_pixel_and_edge_masks_have_valid_boxes(self):
        current = SimpleNamespace(img=np.zeros((10, 20, 3), np.uint8))
        source = source_mask()
        source.mask[:] = 0
        source.mask[-1, -1] = 255
        model = Mock()
        model.predict_boxes.return_value = [source.mask]
        propagate_boxes(model, current, current, [source], 2)
        np.testing.assert_array_equal(model.predict_boxes.call_args.args[1], [[19, 9, 20, 10]])

    def test_sam1_and_sam2_use_independent_image_predictors_and_keep_empty_slots(self):
        image = np.zeros((10, 20, 3), np.uint8)
        image[:, :, 0] = 17
        boxes = np.array([[3, 2, 9, 6], [0, 0, 1, 1]], np.float32)
        prediction = np.ones((1, 10, 20), np.float32)
        for generation, cls in ((1, SamInference), (2, Sam2Inference)):
            with self.subTest(generation=generation):
                sam = cls.__new__(cls)
                sam.device = 'cpu'
                sam.sam = Mock()
                sam.predictor = Mock()
                sam.img_predictor = Mock()
                sam._image_entry = Mock(return_value={'features': (None, object())})
                sam._set_img_predictor_features = Mock()
                active = sam.predictor
                predictor = Mock()
                predictor.predict.side_effect = [(prediction, None, None),
                                                 (np.zeros_like(prediction), None, None)]
                factory = Mock(return_value=predictor)
                modules = {'sam2.sam2_image_predictor': SimpleNamespace(SAM2ImagePredictor=factory)}
                with patch('aramsam_annotator.run_sam.SamPredictor', factory), \
                        patch.dict(sys.modules, modules):
                    result = sam.predict_boxes(image, boxes)
                factory.assert_called_once_with(sam.sam if generation == 1 else active)
                if generation == 1:
                    predictor.set_image.assert_called_once()
                    np.testing.assert_array_equal(predictor.set_image.call_args.args[0], image[..., ::-1])
                else:
                    predictor.set_image.assert_not_called()
                    sam._image_entry.assert_called_once_with(image, image_format='BGR')
                    self.assertIs(sam._set_img_predictor_features.call_args.kwargs['predictor'], predictor)
                np.testing.assert_array_equal(predictor.predict.call_args_list[0].kwargs['box'], boxes[0])
                np.testing.assert_array_equal(result[0], prediction[0].astype(np.uint8) * 255)
                self.assertIsNone(result[1])
                active.set_image.assert_not_called()
                sam.img_predictor.set_image.assert_not_called()

    def test_sam3_uses_destination_state_without_changing_active_state(self):
        sam = Sam3Inference.__new__(Sam3Inference)
        sam.device = 'cpu'
        sam.state = active = object()
        destination = object()
        image = np.zeros((10, 20, 3), np.uint8)
        sam.embed_img = Mock(return_value=(destination, (10, 20), (10, 20)))
        sam.model = Mock()
        sam.model.predict_inst.return_value = (np.ones((1, 10, 20)), None, None)
        boxes = np.array([[3, 2, 9, 6]], np.float32)
        result = sam.predict_boxes(image, boxes)
        sam.embed_img.assert_called_once_with(image, image_format='BGR')
        args, kwargs = sam.model.predict_inst.call_args
        self.assertIs(args[0], destination)
        np.testing.assert_array_equal(kwargs['box'], boxes[0])
        self.assertFalse(kwargs['multimask_output'])
        self.assertIs(sam.state, active)
        self.assertEqual(result[0].dtype, np.uint8)
        self.assertTrue(np.all(result[0] == 255))

    def test_worker_failure_leaves_destination_unchanged_and_releases_mutex(self):
        model = Mock()
        model.predict_boxes.side_effect = RuntimeError('prediction failed')
        current = SimpleNamespace(img=np.zeros((10, 20, 3), np.uint8))
        following = SimpleNamespace(img=current.img.copy(), add_masks=Mock())
        mutex = QMutex()
        worker = BoxPropagationWorker(model, current, following, [source_mask()], 1, mutex)
        with self.assertRaisesRegex(RuntimeError, 'prediction failed'):
            worker.execute()
        following.add_masks.assert_not_called()
        self.assertTrue(mutex.tryLock())
        mutex.unlock()


class BoxPropagationUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = [self.root / f'image{index}.png' for index in range(3)]
        for image in self.images:
            cv2.imwrite(str(image), np.zeros((10, 20, 3), np.uint8))

    def make_app(self, generation):
        configs = AramsamConfigs(mask_propagation_method='bounding_box',
                                 img_tiles=ImgTiles(do_tiling=False),
                                 save_data=SaveData(auto_save=False, do_save=False))
        with patch('aramsam_annotator.gui.get_monitors', return_value=[]):
            app = App(configs=configs)
        app.sam_gen = generation
        app.annotator.annotation = AnnotationObject(self.images[0])
        app.annotator.next_annotation = AnnotationObject(self.images[1])
        app.annotator.sam = Mock(device='cpu')
        app.ui.open_save_annots_box = Mock()
        app.embed_img = Mock()
        app.embed_img_pair = Mock()
        self.addCleanup(app.ui.deleteLater)
        self.addCleanup(app.ui.hide)
        self.addCleanup(app.shutdown)
        return app

    def test_batched_propagation_and_next_image_preserve_objects_without_duplicates(self):
        for generation in (1, 2, 3):
            for batch_size in (None, 1):
                with self.subTest(generation=generation, batch_size=batch_size):
                    app = self.make_app(generation)
                    source = source_mask()
                    app.annotator.annotation.good_masks = [source]
                    threads = []

                    def predict(image, boxes):
                        threads.append(QThread.currentThread())
                        return [source.mask.copy()]

                    app.annotator.sam.predict_boxes.side_effect = predict
                    app.mask_track_batch_size = batch_size
                    if batch_size == 1:
                        app.start_mask_batch_thread()
                    app.img_fnames = [str(self.images[2]), str(self.images[1])]
                    app.select_next_img()
                    self.assertEqual(app.annotator.sam.predict_boxes.call_count,
                                     2 if batch_size == 1 else 1)
                    self.assertNotEqual(threads[0], self.application.thread())
                    self.assertEqual([mask.mid for mask in app.annotator.annotation.good_masks], [7])
                    self.assertEqual(app.annotator.annotation.mask_decisions, [True])
                    self.assertEqual(app.annotator.annotation.good_masks[0].class_id, 4)
                    app.annotator.sam.prop_thread_func.assert_not_called()
                    self.assertIsNone(app.bbox_tracker)

    def test_disabled_propagation_and_no_next_image_do_not_predict(self):
        app = self.make_app(1)
        app.annotator.annotation.good_masks = [source_mask()]
        app.configs.mask_propagation = False
        app.propagate_good_masks()
        app.start_mask_batch_thread(track_remaining=True)
        app.configs.mask_propagation = True
        app.annotator.next_annotation = None
        app.propagate_good_masks()
        app.start_mask_batch_thread(track_remaining=True)
        app.annotator.sam.predict_boxes.assert_not_called()

    def test_disabling_propagation_removes_box_results_for_each_model(self):
        for generation in (1, 2, 3):
            with self.subTest(generation=generation):
                app = self.make_app(generation)
                source = source_mask()
                app.annotator.annotation.good_masks = [source]
                app.annotator.sam.predict_boxes.return_value = [source.mask.copy()]
                app.mask_track_batch_size = 1
                app.start_mask_batch_thread()
                app.ui.mask_propagation_box.click()
                self.assertFalse(app.configs.mask_propagation)
                self.assertEqual(app.annotator.next_annotation.masks, [])
                self.assertEqual(app.annotator.next_annotation.mask_decisions, [])

    def test_failed_box_propagation_can_be_retried(self):
        app = self.make_app(2)
        source = source_mask()
        app.annotator.annotation.good_masks = [source]
        app.mask_track_batch_size = 1
        app.print_thread_error = Mock()
        app.annotator.sam.predict_boxes.side_effect = RuntimeError('prediction failed')
        with patch('aramsam_annotator.workers.traceback.print_exc'):
            app.start_mask_batch_thread()
        app.print_thread_error.assert_called_once()
        self.assertEqual(app.propagated_mids, set())
        self.assertEqual(app.annotator.next_annotation.masks, [])
        app.annotator.sam.predict_boxes.side_effect = None
        app.annotator.sam.predict_boxes.return_value = [source.mask.copy()]
        app.start_mask_batch_thread()
        self.assertEqual([mask.mid for mask in app.annotator.next_annotation.masks], [7])

    def test_config_selection_round_trips_and_rejects_unknown_method(self):
        path = self.root / 'settings.yaml'
        path.write_text('{}')
        self.assertEqual(load_configs_from_yaml(path).mask_propagation_method, 'tracking')
        dialog = WelcomeWindow(AramsamConfigs(), path)
        self.addCleanup(dialog.deleteLater)
        control = dialog.controls['mask_propagation_method']
        control.setCurrentIndex(control.findData('bounding_box'))
        configs = dialog.read_configs()
        self.assertEqual(configs.mask_propagation_method, 'bounding_box')
        save_configs_to_yaml(configs, path)
        self.assertEqual(load_configs_from_yaml(path).mask_propagation_method, 'bounding_box')
        path.write_text('mask_propagation_method: unknown')
        with self.assertRaisesRegex(ValueError, 'propagation method'):
            load_configs_from_yaml(path)

    def test_box_results_can_be_saved_and_loaded_for_all_models(self):
        settings = SaveData(mask_style='default')
        repository = AnnotationRepository(settings)
        for generation in (1, 2, 3):
            with self.subTest(generation=generation):
                annotation = AnnotationObject(self.images[1])
                result = source_mask()
                result.origin = f'Sam{generation}_box_propagation'
                result.time_stamp = 1
                annotation.good_masks = [result]
                repository.save(annotation, self.root)
                restored = AnnotationObject(self.images[1])
                repository.load(restored, self.root, MaskIdHandler())
                self.assertEqual(restored.good_masks[0].origin, result.origin)
                self.assertEqual(restored.good_masks[0].class_id, result.class_id)
                np.testing.assert_array_equal(restored.good_masks[0].mask, result.mask)


if __name__ == '__main__':
    unittest.main()
