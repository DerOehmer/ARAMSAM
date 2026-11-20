from collections import OrderedDict
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
import unittest

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QEventLoop, QThread, QTimer
from PyQt6.QtWidgets import QApplication

from aramsam_annotator.app import App
from aramsam_annotator.backend.annotations import MaskData
from aramsam_annotator.configs import AramsamConfigs, ImgTiles, SamConfigs, SaveData
from aramsam_annotator.controllers.inference import InferenceController
from aramsam_annotator.controllers.presentation import PresentationController
from aramsam_annotator.run_sam import Sam2Inference


def cached_sam():
    """Exercise the real adapter/cache with only the expensive model stubbed."""
    sam = Sam2Inference.__new__(Sam2Inference)
    sam.device = 'cpu'
    sam.predictor = Mock(is_image_set=False)
    sam.img_predictor = SimpleNamespace(_features=None)
    sam.imgs = None
    sam.inference_state = None
    sam._current_entry = None
    sam._tracking_images = None
    sam._image_cache = OrderedDict()
    sam._cache_lock = Lock()

    def init_state(video_path, **kwargs):
        assert torch.is_inference_mode_enabled()
        image = video_path[0]
        pixels = torch.zeros((1, 3, 8, 8))
        return {'images': pixels, 'num_frames': 1,
                'cached_features': {0: (pixels, object())},
                'video_height': image.shape[0], 'video_width': image.shape[1]}

    def activate(backbone, image=None, predictor=None):
        predictor = sam.img_predictor if predictor is None else predictor
        predictor._features = backbone

    sam.predictor.init_state.side_effect = init_state
    sam._set_img_predictor_features = Mock(side_effect=activate)
    return sam


class Sam2CacheTests(unittest.TestCase):
    def setUp(self):
        self.sam = cached_sam()
        self.images = [np.zeros((8, 8, 3), np.uint8) for _ in range(3)]

    def test_current_is_ready_without_encoding_the_next_image(self):
        sam = self.sam
        sam.set_features(self.images[:2], image_format='BGR')
        self.assertEqual(sam.predictor.init_state.call_count, 1)
        self.assertTrue(sam.predictor.is_image_set)
        self.assertTrue(sam.has_cached_image(self.images[0]))
        self.assertFalse(sam.has_cached_image(self.images[1]))
        self.assertEqual(sam.inference_state['num_frames'], 1)

    def test_prefetch_preserves_active_predictor_and_reuses_embedding_on_next(self):
        sam = self.sam
        sam.set_features(self.images[:2], image_format='BGR')
        active_features, state = sam.img_predictor._features, sam.inference_state
        sam.prefetch_image(self.images[1])
        sam.prefetch_image(self.images[1])
        self.assertEqual(sam.predictor.init_state.call_count, 2)
        self.assertIs(sam.img_predictor._features, active_features)
        self.assertIs(sam.inference_state, state)
        next_features = sam._image_cache[(id(self.images[1]), 'BGR')]['features'][1]
        sam.set_features(self.images[1:], image_format='BGR')
        self.assertIs(sam.img_predictor._features, next_features)
        self.assertEqual(sam.predictor.init_state.call_count, 2)
        sam.prefetch_image(self.images[2])
        self.assertEqual(len(sam._image_cache), 2)
        self.assertFalse(sam.has_cached_image(self.images[0]))
        self.assertTrue(sam.has_cached_image(self.images[1]))
        self.assertTrue(sam.has_cached_image(self.images[2]))

    def test_tracking_reseeds_both_cached_frames_without_reencoding(self):
        sam = self.sam
        sam.set_features(self.images[:2], image_format='BGR')
        sam.prefetch_image(self.images[1])
        source = MaskData(mid=7, origin='Polygon_drawing', mask=np.ones((8, 8), np.uint8))
        sam.set_masks([source])
        pixels = sam.inference_state['images']
        self.assertEqual(sam.inference_state['num_frames'], 2)
        self.assertEqual(set(sam.inference_state['cached_features']), {0, 1})
        sam.inference_state['cached_features'].pop(0)
        sam.set_masks([source])
        self.assertEqual(set(sam.inference_state['cached_features']), {0, 1})
        self.assertIs(sam.inference_state['images'], pixels)
        self.assertEqual(sam.predictor.init_state.call_count, 2)

    def test_tracking_encodes_next_on_demand_without_prefetch(self):
        sam = self.sam
        sam.set_features(self.images[:2], image_format='BGR')
        sam.set_masks([MaskData(mid=7, origin='Polygon_drawing', mask=np.ones((8, 8), np.uint8))])
        self.assertEqual(sam.predictor.init_state.call_count, 2)
        self.assertEqual(sam.inference_state['num_frames'], 2)

    def test_deleted_objects_are_removed_from_tracking_memory_and_empty_tracking_is_safe(self):
        sam = self.sam
        sam.set_features(self.images[:2], image_format='BGR')
        sam.inference_state['obj_ids'] = [7, 8]
        sam.predictor.reset_state.side_effect = lambda state: state['obj_ids'].clear()
        sam.predictor.add_new_mask.side_effect = lambda state, frame, mid, mask: state['obj_ids'].append(mid)
        features = sam.img_predictor._features
        remaining = MaskData(mid=8, origin='Polygon_drawing', mask=np.ones((8, 8), np.uint8))
        sam.set_masks([remaining])
        self.assertEqual(sam.inference_state['obj_ids'], [8])
        encoded = sam.predictor.init_state.call_count
        self.assertEqual(sam.prop_thread_func([]), [])
        self.assertEqual(sam.inference_state['obj_ids'], [])
        self.assertEqual(sam._tracking_objects, {})
        self.assertIs(sam.img_predictor._features, features)
        self.assertEqual(sam.predictor.init_state.call_count, encoded)
        sam.predictor.propagate_in_video.assert_not_called()

    def test_bgr_is_converted_once_and_public_rgb_input_is_preserved(self):
        image = self.images[0]
        image[:, :, 0] = 17
        self.sam.prefetch_image(image)
        submitted = self.sam.predictor.init_state.call_args.kwargs['video_path'][0]
        np.testing.assert_array_equal(submitted, image[..., ::-1])
        self.sam.set_features([image])
        submitted = self.sam.predictor.init_state.call_args.kwargs['video_path'][0]
        np.testing.assert_array_equal(submitted, image)

    def test_failed_prefetch_is_not_cached_and_can_be_retried(self):
        sam = self.sam
        sam.set_features(self.images[:2], image_format='BGR')
        active_features = sam.img_predictor._features
        original = sam.predictor.init_state.side_effect
        sam.predictor.init_state.side_effect = RuntimeError('encoding failed')
        with self.assertRaisesRegex(RuntimeError, 'encoding failed'):
            sam.prefetch_image(self.images[1])
        self.assertFalse(sam.has_cached_image(self.images[1]))
        self.assertIs(sam.img_predictor._features, active_features)
        sam.predictor.init_state.side_effect = original
        sam.prefetch_image(self.images[1])
        self.assertTrue(sam.has_cached_image(self.images[1]))

    def test_box_prompts_reuse_prefetched_features_and_leave_active_features_unchanged(self):
        sam = self.sam
        sam.set_features(self.images[:2], image_format='BGR')
        sam.prefetch_image(self.images[1])
        active_features = sam.img_predictor._features
        predictor = Mock()
        predictor.predict.return_value = (np.ones((1, 8, 8), np.float32), None, None)
        with patch('sam2.sam2_image_predictor.SAM2ImagePredictor', return_value=predictor):
            masks = sam.predict_boxes(self.images[1], np.array([[1, 1, 5, 5]], np.float32))
        self.assertEqual(sam.predictor.init_state.call_count, 2)
        predictor.set_image.assert_not_called()
        self.assertIs(sam.img_predictor._features, active_features)
        self.assertEqual(len(masks), 1)
        np.testing.assert_array_equal(masks[0], np.full((8, 8), 255, np.uint8))


class Sam2PrefetchControllerTests(unittest.TestCase):
    def make_controller(self):
        target = SimpleNamespace(img=np.zeros((8, 8, 3), np.uint8), img_name='next.png')
        model = Mock()
        model.has_cached_image.return_value = False
        context = SimpleNamespace(
            experiment_mode=None, configs=AramsamConfigs(), ui=Mock(),
            annotator=SimpleNamespace(sam=model, next_annotation=target),
            print_thread_error=Mock(),
        )
        context.ui.auto_embed_box.isChecked.return_value = True
        controller = InferenceController(context)
        controller._submit = Mock()
        return controller, context

    def test_prefetch_respects_settings_and_does_not_repeat_pending_or_cached_work(self):
        for skip in ('background_disabled', 'auto_embed_disabled', 'experiment', 'last', 'cached'):
            with self.subTest(skip=skip):
                controller, context = self.make_controller()
                if skip == 'background_disabled':
                    context.configs.sam_background_embedding = False
                elif skip == 'auto_embed_disabled':
                    context.ui.auto_embed_box.isChecked.return_value = False
                elif skip == 'experiment':
                    context.experiment_mode = 'structured'
                elif skip == 'last':
                    context.annotator.next_annotation = None
                else:
                    context.annotator.sam.has_cached_image.return_value = True
                controller.prefetch_sam2_next_image()
                controller._submit.assert_not_called()
        controller, _ = self.make_controller()
        controller.prefetch_sam2_next_image()
        controller.prefetch_sam2_next_image()
        controller._submit.assert_called_once()

    def test_prefetch_errors_allow_retry_without_disabling_current_annotation(self):
        controller, context = self.make_controller()
        controller.prefetch_sam2_next_image()
        error = (RuntimeError, RuntimeError('encoding failed'), 'traceback')
        controller._submit.call_args.kwargs['error'](error)
        context.print_thread_error.assert_called_once_with(error)
        self.assertIsNone(controller.sam2_prefetch_target)
        controller.prefetch_sam2_next_image()
        self.assertEqual(controller._submit.call_count, 2)
        context.ui.manual_annotation_button.setEnabled.assert_not_called()

    def test_current_image_is_displayed_before_next_prefetch_is_scheduled(self):
        order = []
        context = SimpleNamespace(
            sam_gen=2, experiment_mode=None, configs=AramsamConfigs(), ui=Mock(),
            annotator=Mock(), navigation=Mock(), inference=Mock(),
            run_backend_task=lambda operation, *args: operation(),
            update_ui_imgs=lambda: order.append('display'),
        )
        context.annotator.update_collections.side_effect = lambda _: order.append('render')
        context.inference.prefetch_sam2_next_image.side_effect = lambda: order.append('prefetch')
        PresentationController(context).start_user_annotation()
        self.assertEqual(order, ['render', 'display', 'prefetch'])


class Sam2PrefetchUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_current_image_stays_usable_while_next_image_is_being_encoded(self):
        with tempfile.TemporaryDirectory() as directory:
            images = [Path(directory) / f'image{index}.png' for index in range(3)]
            for image in images:
                cv2.imwrite(str(image), np.zeros((8, 8, 3), np.uint8))
            config = AramsamConfigs(sam_configs=SamConfigs(gen=2), mask_propagation=False,
                                   img_tiles=ImgTiles(do_tiling=False),
                                   save_data=SaveData(do_save=False, auto_save=False))
            with patch('aramsam_annotator.gui.get_monitors', return_value=[]):
                app = App(configs=config)
            app.sam_gen = 2
            sam = cached_sam()
            app.annotator.sam = sam
            started, release = Event(), Event()
            threads, locks_available = [], []
            original_prefetch = sam.prefetch_image

            def prefetch(image):
                threads.append(QThread.currentThread())
                available = app.mutex.tryLock()
                locks_available.append(available)
                if available:
                    app.mutex.unlock()
                started.set()
                if not release.wait(3):
                    raise RuntimeError('Current image did not become usable during prefetch')
                original_prefetch(image)

            sam.prefetch_image = prefetch
            try:
                app.img_fnames = [str(image) for image in reversed(images)]
                app.select_next_img()
                loop = QEventLoop()
                timer = QTimer()
                timer.timeout.connect(lambda: loop.quit() if started.is_set() else None)
                timer.start(5)
                deadline = QTimer()
                deadline.setSingleShot(True)
                deadline.timeout.connect(loop.quit)
                deadline.start(2000)
                loop.exec()
                timer.stop()
                deadline.stop()
                self.assertTrue(started.is_set())
                self.assertTrue(app.ui.manual_annotation_button.isEnabled())
                self.assertTrue(sam.predictor.is_image_set)
                self.assertEqual(sam.predictor.init_state.call_count, 1)
                self.assertTrue(app.tasks.pending)
                self.assertEqual(locks_available, [True])
                self.assertNotEqual(threads[0], self.application.thread())
                release.set()
                app.wait_for_workers()
                self.assertEqual(sam.predictor.init_state.call_count, 2)
                app.ui.open_save_annots_box = Mock()
                app.select_next_img()
                app.wait_for_workers()
                self.assertEqual(app.annotator.annotation.filepath, str(images[1]))
                # First, second and third images are each encoded once.
                self.assertEqual(sam.predictor.init_state.call_count, 3)
            finally:
                release.set()
                app.shutdown()
                app.ui.hide()
                app.ui.deleteLater()


if __name__ == '__main__':
    unittest.main()
