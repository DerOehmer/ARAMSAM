import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QMutex, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QPushButton

from aramsam_annotator.app import App
from aramsam_annotator.annotator import Annotator
from aramsam_annotator.backend.annotations import AnnotationObject, MaskData, MaskIdHandler
from aramsam_annotator.backend.storage import AnnotationRepository
from aramsam_annotator.configs import AramsamConfigs, SamConfigs, SaveData, ImgTiles, load_configs_from_yaml
from aramsam_annotator.run_sam import DefaultAMG
from aramsam_annotator.run_sam3 import Sam3Inference, Sam3AutomaticMaskGenerator
from aramsam_annotator.workers import Sam1EmbeddingWorker, TextPromptWorker


def binary_mask():
    mask = np.zeros((12, 16), np.uint8)
    mask[2:8, 3:10] = 255
    return mask


class FakeProcessor:
    def __init__(self):
        self.image = None
        self.grad_enabled = None
        self.transform = lambda pixels: pixels.float() / 255
        self.confidence = 0.5

    def set_image(self, image):
        self.image = np.array(image)
        self.grad_enabled = torch.is_grad_enabled()
        return {"original_height": image.height, "original_width": image.width,
                "backbone_out": {"sam2_backbone_out": {"backbone_fpn": [torch.zeros(1)]}}}

    def reset_all_prompts(self, state):
        state["backbone_out"].pop("language_features", None)
        state.pop("geometric_prompt", None)

    def set_confidence_threshold(self, value):
        self.confidence = value

    def set_text_prompt(self, state, prompt):
        state["backbone_out"]["language_features"] = prompt
        state["geometric_prompt"] = prompt
        return {"masks": torch.tensor(np.stack([binary_mask() > 0, np.zeros((12, 16), bool)]))[:, None]}


def fake_sam():
    sam = Sam3Inference.__new__(Sam3Inference)
    sam.device = "cpu"
    sam.mask_id_handler = MaskIdHandler()
    sam.processor = FakeProcessor()
    sam.tracker = Mock()
    sam.tracker.sam_prompt_encoder.mask_input_size = (8, 8)
    sam.model = Mock()
    sam.model.predict_inst.return_value = (
        (binary_mask() > 0)[None], np.ones(1), np.ones((1, 8, 8)))
    sam.predictor = sam
    sam.state = None
    sam.is_image_set = False
    sam.inference_state = None
    sam._next_annotation = None
    sam.background_embedding = True
    state, height, size = sam.embed_img(np.zeros((12, 16, 3), np.uint8), "BGR")
    sam.set_features(state, height, size)
    return sam


class Sam3AdapterTests(unittest.TestCase):
    def test_builder_enables_interactivity_and_uses_local_checkpoint_and_vocab(self):
        builder, processor = MagicMock(), Mock()
        modules = {
            'sam3.model_builder': SimpleNamespace(build_sam3_image_model=builder),
            'sam3.model.sam3_image_processor': SimpleNamespace(Sam3Processor=processor),
        }
        with patch.dict(sys.modules, modules):
            sam = Sam3Inference(MaskIdHandler(), "local.pt", device="cuda", bpe_path="vocab.gz")
        self.assertTrue(builder.call_args.kwargs['enable_inst_interactivity'])
        self.assertFalse(builder.call_args.kwargs['load_from_HF'])
        self.assertEqual(builder.call_args.kwargs['checkpoint_path'], 'local.pt')
        self.assertEqual(builder.call_args.kwargs['bpe_path'], 'vocab.gz')
        self.assertEqual(processor.call_args.kwargs['device'], 'cuda')
        self.assertIs(sam.predictor, sam)

    def test_cpu_reports_installed_package_requirement(self):
        with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
            Sam3Inference(MaskIdHandler(), device="cpu")

    def test_config_loads_generation_three_and_text_options(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('sam_configs:\n  gen: 3\n  model_ckpt_p: sam3.pt\n  bpe_path: vocab.gz\n  confidence_threshold: 0.4\n')
            config = load_configs_from_yaml(path)
        self.assertEqual(config.sam_configs.gen, 3)
        self.assertEqual(config.sam_configs.confidence_threshold, 0.4)
        annotator = Annotator(config)
        with patch('aramsam_annotator.annotator.Sam3Inference') as model:
            annotator.set_sam_version(3, False)
        self.assertEqual(model.call_args.kwargs['bpe_path'], 'vocab.gz')
        self.assertFalse(model.call_args.kwargs['background_embedding'])

    def test_background_embedding_keeps_active_state_and_converts_bgr(self):
        sam = fake_sam()
        active = sam.state
        image = np.zeros((5, 9, 3), np.uint8)
        image[:, :, 0] = 27
        features, original, size = sam.embed_img(image, 'BGR')
        self.assertIs(sam.state, active)
        self.assertEqual(original, (5, 9))
        self.assertEqual(size, (5, 9))
        self.assertEqual(features['original_width'], 9)
        self.assertTrue(np.all(sam.processor.image[:, :, 2] == 27))
        self.assertFalse(sam.processor.grad_enabled)
        self.assertFalse(features['tracking_image'].requires_grad)
        sam.set_features(features)
        self.assertIs(sam.state, features)
        sam.reset_img()
        with self.assertRaisesRegex(ValueError, 'embedding'):
            sam.predict_text('cup')

    def test_point_box_mask_and_batch_predictions_keep_image_coordinates(self):
        sam = fake_sam()
        points, labels = np.array([[3, 4], [5, 6]], np.float32), np.array([1, 0], np.int32)
        np.testing.assert_array_equal(sam.predict(points, labels), binary_mask())
        arguments = sam.model.predict_inst.call_args.kwargs
        np.testing.assert_array_equal(arguments['point_coords'], points)
        np.testing.assert_array_equal(arguments['point_labels'], labels)
        boxes = np.array([[1, 2, 8, 9], [2, 3, 9, 10]], np.float32)
        output = sam.predict_batch(bboxes=sam.transform_bboxes(boxes, (12, 16)))
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].shape, (1, 12, 16))
        np.testing.assert_array_equal(sam.model.predict_inst.call_args.kwargs['box'], boxes[1])
        sam.predict_batch(mask_input=[binary_mask()])
        mask_input = sam.model.predict_inst.call_args.kwargs['mask_input']
        self.assertEqual(mask_input.shape, (1, 8, 8))
        self.assertEqual(mask_input.dtype, np.float32)
        self.assertLess(mask_input.min(), 0)
        with self.assertRaisesRegex(ValueError, 'same length'):
            sam.predict_batch(pts=points[:, None], pts_labels=labels[:1, None])
        with self.assertRaisesRegex(ValueError, 'together'):
            sam.predict(points)

    def test_text_filters_empty_masks_and_never_mutates_image_cache(self):
        sam = fake_sam()
        for text in (' cup ', 'seed'):
            output = sam.predict_text(text, 0.25)
            self.assertEqual(len(output), 1)
            np.testing.assert_array_equal(output[0], binary_mask())
            self.assertEqual(output[0].dtype, np.uint8)
            self.assertNotIn('language_features', sam.state['backbone_out'])
            self.assertNotIn('geometric_prompt', sam.state)
        self.assertEqual(sam.processor.confidence, 0.25)
        with self.assertRaises(ValueError):
            sam.predict_text(' ')
        with self.assertRaises(ValueError):
            sam.predict_text('cup', 1.5)

    def test_tracking_reuses_backbone_preserves_ids_and_resizes_next_masks(self):
        sam = fake_sam()
        following = SimpleNamespace(img=np.zeros((24, 32, 3), np.uint8), features=sam.state)
        sam.set_tracking_pair(None, following)
        obj = MaskData(mid=72, mask=binary_mask(), origin='Sam3_text', class_id=3, color_idx=2)
        sam.device = 'cuda'
        with patch.object(sam, '_precision', side_effect=nullcontext):
            sam.set_masks([obj])
            state = sam.tracker.init_state.call_args.kwargs
            self.assertEqual(state['num_frames'], 2)
            self.assertEqual(set(state['cached_features']), {0, 1})
            self.assertIs(state['cached_features'][0][1], sam.state['backbone_out']['sam2_backbone_out'])
            self.assertIsInstance(sam.tracker.add_new_mask.call_args.args[3], torch.Tensor)
            sam.tracker.propagate_in_video.return_value = iter([
                (0, [72], None, (binary_mask() > 0)[None, None], None),
                (1, [72], None, (binary_mask() > 0)[None, None], None)])
            result = sam.propagate_to_next_img()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].mid, 72)
        self.assertEqual(result[0].origin, 'Sam3_tracking')
        self.assertEqual(result[0].class_id, 3)
        self.assertEqual(result[0].color_idx, 2)
        self.assertEqual(result[0].mask.shape, (24, 32))
        sam.tracker.propagate_in_video.assert_called_once_with(
            sam.inference_state, start_frame_idx=0, max_frame_num_to_track=1,
            reverse=False, propagate_preflight=True)

    def test_amg_filters_low_quality_and_duplicate_masks(self):
        sam = fake_sam()
        mask = binary_mask() > 0
        logits = np.where(mask, 4.0, -4.0)
        sam.model.predict_inst.return_value = (
            np.stack([mask, mask, ~mask]), np.array([0.95, 0.94, 0.1]),
            np.stack([logits, logits, -logits]))
        generator = Sam3AutomaticMaskGenerator(sam, points_per_side=1)
        output = generator.generate(np.zeros((12, 16, 3), np.uint8))
        self.assertEqual(len(output), 1)
        np.testing.assert_array_equal(output[0]['segmentation'], mask)
        wrapper = DefaultAMG(sam, generator)
        wrapper.set_visualization_img(np.zeros((12, 16, 3), np.uint8))
        objects, image = wrapper()
        self.assertEqual(objects[0].origin, 'Sam3_proposed')
        self.assertEqual(image.shape, (12, 16, 3))

    def test_embedding_and_text_workers_release_lock_on_success_and_failure(self):
        sam, mutex = fake_sam(), QMutex()
        worker = Sam1EmbeddingWorker(sam, np.zeros((12, 16, 3), np.uint8), 'next.png', mutex)
        received = []
        worker.signals.result.connect(received.append)
        active = sam.state
        worker.run()
        self.assertIs(sam.state, active)
        self.assertEqual(received[0][-1], 'next.png')
        for fail in (False, True):
            worker = TextPromptWorker(sam, 'cup', 0.5, mutex)
            results, errors, finished = [], [], []
            worker.signals.result.connect(results.append)
            worker.signals.error.connect(errors.append)
            worker.signals.finished.connect(finished.append)
            with patch.object(sam, 'predict_text', side_effect=RuntimeError('failed') if fail else None,
                              return_value=[binary_mask()]), patch('aramsam_annotator.workers.traceback.print_exc'):
                worker.run()
            self.assertEqual(len(results), int(not fail))
            self.assertEqual(len(errors), int(fail))
            self.assertEqual(finished, ['done'])
            self.assertTrue(mutex.tryLock())
            mutex.unlock()


class Sam3UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config = AramsamConfigs(sam_configs=SamConfigs(gen=3, model_ckpt_p='sam3.pt'),
                               img_tiles=ImgTiles(do_tiling=False), sam_background_embedding=False,
                               save_data=SaveData(mask_style='default'))
        with patch('aramsam_annotator.gui.get_monitors', return_value=[]):
            self.app = App(configs=config)
        self.addCleanup(self.cleanup_app)
        self.app.sam_gen = 3
        self.app.annotator.sam = fake_sam()
        self.image = self.root / 'sample.png'
        cv2.imwrite(str(self.image), np.zeros((12, 16, 3), np.uint8))
        self.app.annotator.create_new_annotation(self.image)
        self.app.annotator.annotation.set_sam_parameters(self.app.annotator.sam.state, (12, 16), (12, 16))
        self.app.ui.enable_push_buttons()
        self.app.ui.create_message_box = Mock()
        self.application.processEvents()

    def cleanup_app(self):
        self.app.shutdown()
        self.app.ui.hide()
        self.app.ui.deleteLater()

    def test_dropdown_hide_resize_history_enter_and_shortcut_isolation(self):
        ui = self.app.ui
        self.assertTrue(ui.text_prompt_panel.isHidden())
        ui.text_prompt_toggle.click()
        self.assertTrue(ui.text_prompt_panel.isVisible())
        ui.resize(1000, 700)
        self.application.processEvents()
        self.assertLessEqual(ui.text_prompt_panel.geometry().right(), ui.width())
        actions = []
        ui.preview_annotation_point_signal.connect(actions.append)
        submitted = []
        ui.text_prompt_signal.connect(lambda prompt, confidence: submitted.append((prompt, confidence)))
        edit = ui.text_prompt_input.lineEdit()
        edit.setFocus()
        QTest.keyClicks(edit, 'banana seed')
        self.assertEqual(actions, [])
        self.assertFalse(self.app.annotator.manual_annotation_enabled)
        QTest.keyClick(edit, Qt.Key.Key_Return)
        self.app.wait_for_workers()
        self.assertEqual(submitted, [('banana seed', 0.5)])
        self.assertEqual(ui.text_prompt_input.itemText(0), 'banana seed')
        next(button for button in ui.text_prompt_panel.findChildren(QPushButton)
             if button.text() == 'Hide').click()
        self.assertTrue(ui.text_prompt_panel.isHidden())
        ui.text_prompt_toggle.click()
        self.assertEqual(ui.text_prompt_input.currentText(), 'banana seed')
        ui.set_text_prompt_available(False)
        self.assertTrue(ui.text_prompt_panel.isHidden())
        self.assertFalse(ui.text_prompt_toggle.isEnabled())

    def test_text_masks_review_accept_undo_and_native_round_trip(self):
        self.app.start_text_prompt('cup', 0.5)
        self.assertFalse(self.app.ui.text_prompt_run.isEnabled())
        self.app.wait_for_workers()
        annotator = self.app.annotator
        self.assertTrue(self.app.ui.text_prompt_run.isEnabled())
        self.assertEqual(len(annotator.annotation.masks), 1)
        self.assertEqual(annotator.annotation.masks[0].origin, 'Sam3_text')
        self.assertEqual(annotator.annotation.mask_decisions, [False])
        self.app.ui.good_mask_button.click()
        self.assertEqual(len(annotator.annotation.good_masks), 1)
        self.app.ui.back_button.click()
        self.assertEqual(len(annotator.annotation.good_masks), 0)
        self.app.ui.good_mask_button.click()
        for origin in ('Sam3_text', 'Sam3_interactive', 'Sam3_tracking', 'Sam3_proposed'):
            annotator.annotation.good_masks[0].origin = origin
            output = self.root / origin
            annotator.save_annotations(output)
            restored = AnnotationObject(self.image)
            AnnotationRepository(self.app.configs.save_data).load(restored, output, MaskIdHandler())
            self.assertEqual(restored.good_masks[0].origin, origin)
            np.testing.assert_array_equal(restored.good_masks[0].mask, binary_mask())

    def test_interactive_acceptance_uses_sam3_origin(self):
        self.app.ui.manual_annotation_button.click()
        self.app.annotator.predict_sam_manually((4, 5))
        self.app.ui.good_mask_button.click()
        self.assertEqual(self.app.annotator.annotation.good_masks[0].origin, 'Sam3_interactive')

    def test_empty_or_failed_text_results_preserve_annotations_and_allow_retry(self):
        self.app.inference.receive_text_masks([binary_mask()])
        objects = list(self.app.annotator.annotation.masks)
        with patch.object(self.app.annotator.sam, 'predict_text', return_value=[]):
            self.app.start_text_prompt('absent', 0.5)
            self.app.wait_for_workers()
        self.assertEqual(self.app.annotator.annotation.masks, objects)
        self.assertIn('No objects found', self.app.ui.text_prompt_status.text())
        with patch.object(self.app.annotator.sam, 'predict_text', side_effect=RuntimeError('failure')), \
                patch('aramsam_annotator.workers.traceback.print_exc'):
            self.app.start_text_prompt('cup', 0.5)
            self.app.wait_for_workers()
        self.assertTrue(self.app.ui.text_prompt_run.isEnabled())
        self.assertFalse(self.app.inference.text_prompt_running)
        self.assertEqual(self.app.annotator.annotation.masks, objects)
        self.app.ui.create_message_box.assert_called_once()

    def test_navigation_activates_cached_sam3_state(self):
        self.app.configs.sam_background_embedding = True
        state = self.app.annotator.sam.state
        self.app.annotator.annotation.set_sam_parameters(state, (12, 16), (12, 16))
        self.app.propose_masks = Mock()
        self.app.embed_img = Mock()
        self.app.navigation._activate_image(self.image, None, False, False)
        self.assertIs(self.app.annotator.sam.state, state)
        self.app.propose_masks.assert_called_once()
        self.app.embed_img.assert_not_called()

    def test_text_prompt_waits_until_active_embedding_matches_current_image(self):
        model = self.app.annotator.sam
        model.predict_text = Mock()
        for features in (None, {'another_image': True}):
            self.app.annotator.annotation.features = features
            self.app.start_text_prompt('cup', 0.5)
        model.predict_text.assert_not_called()
        self.assertFalse(self.app.inference.text_prompt_running)
        self.assertIn('embedding', self.app.ui.text_prompt_status.text())

    def test_switching_model_accepts_sam3_pt_and_updates_prompt_ui(self):
        with patch('aramsam_annotator.annotator.Sam3Inference', return_value=fake_sam()):
            self.app.changed_sam_model(str(self.root / 'sam3.pt'))
            self.app.wait_for_workers()
        self.assertEqual(self.app.sam_gen, 3)
        self.assertTrue(self.app.ui.text_prompt_toggle.isEnabled())
        self.app.ui.create_message_box.assert_not_called()


if __name__ == '__main__':
    unittest.main()
