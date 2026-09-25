import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PyQt6.QtCore import QMutex

from aramsam_annotator.app import App
from aramsam_annotator.configs import AramsamConfigs, SamConfigs
from aramsam_annotator.run_sam import BackgroundThreadSamPredictor, MainThreadSamPredictor
from aramsam_annotator.workers import Sam1EmbeddingWorker


class TinyEncoder(torch.nn.Module):
    img_size = 8

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, image):
        return image * self.weight


class TinySam(torch.nn.Module):
    image_format = 'RGB'

    def __init__(self):
        super().__init__()
        self.image_encoder = TinyEncoder()

    @property
    def device(self):
        return self.image_encoder.weight.device

    def preprocess(self, image):
        return image.float()


def annotation(name):
    obj = SimpleNamespace(img_name=name, img=np.zeros((8, 8, 3), dtype=np.uint8),
                          features=None, original_size=None, input_size=None)

    def set_parameters(**kwargs):
        obj.__dict__.update(kwargs)

    obj.set_sam_parameters = set_parameters
    return obj


class BackgroundEmbeddingTests(unittest.TestCase):
    def make_app(self, enabled=True):
        app = App.__new__(App)
        app.configs = AramsamConfigs(sam_background_embedding=enabled)
        app.experiment_mode = None
        app.ui = Mock()
        app.ui.auto_embed_box.isChecked.return_value = True
        app.threadpool = Mock()
        app.threadpool.activeThreadCount.return_value = 1
        app.mutex = QMutex()
        app.annotator = Mock()
        app.annotator.annotation = annotation('current.jpg')
        app.annotator.next_annotation = annotation('next.jpg')
        app.annotator.get_annotation_img_name.return_value = 'current.jpg'
        app.annotator.get_next_annotation_img_name.return_value = 'next.jpg'
        app.propose_masks = Mock()
        return app

    def test_configuration_preserves_both_modes_for_both_generations(self):
        for gen in (1, 2):
            for enabled in (False, True):
                with self.subTest(gen=gen, enabled=enabled):
                    config = AramsamConfigs(sam_configs=SamConfigs(gen=gen),
                                            sam_background_embedding=enabled)
                    self.assertEqual(config.sam_background_embedding, enabled)

    def test_background_encoding_preserves_active_state_and_has_no_gradients(self):
        predictor = BackgroundThreadSamPredictor(TinySam())
        old_features = torch.zeros(1)
        predictor.set_features(old_features, (4, 4), (8, 8))
        img = np.zeros((4, 8, 3), dtype=np.uint8)
        img[:, :, 0] = 17
        features, original_size, input_size = predictor.embed_img(img, image_format='BGR')
        self.assertIs(predictor.features, old_features)
        self.assertEqual(predictor.original_size, (4, 4))
        self.assertTrue(predictor.is_image_set)
        self.assertFalse(features.requires_grad)
        self.assertIsNone(features.grad_fn)
        self.assertEqual(original_size, (4, 8))
        self.assertEqual(input_size, (4, 8))
        self.assertTrue(torch.all(features[:, 2] == 17))
        predictor.set_features(features, original_size, input_size)
        self.assertIs(predictor.features, features)
        self.assertEqual(predictor.original_size, (4, 8))

    def test_first_embedding_activates_current_and_prefetches_next(self):
        app = self.make_app()
        app.embed_img = Mock()
        features = torch.ones(1)
        app.receive_embedding_from_thread((features, (8, 8), (8, 8), 'current.jpg'))
        self.assertIs(app.annotator.annotation.features, features)
        app.annotator.update_sam_features_to_current_annotation.assert_called_once()
        app.propose_masks.assert_called_once()
        app.embed_img.assert_called_once_with('next.jpg')

    def test_prefetch_only_caches_next_without_touching_current_ui(self):
        app = self.make_app()
        features = torch.ones(1)
        app.receive_embedding_from_thread((features, (8, 8), (8, 8), 'next.jpg'))
        self.assertIs(app.annotator.next_annotation.features, features)
        app.annotator.update_sam_features_to_current_annotation.assert_not_called()
        app.propose_masks.assert_not_called()
        app.ui.close_basic_loading_window.assert_not_called()

    def test_no_prefetch_when_disabled_or_last_image_or_experiment(self):
        for mode in ('disabled', 'last', 'structured', 'cached'):
            with self.subTest(mode=mode):
                app = self.make_app(enabled=mode != 'disabled')
                app.embed_img = Mock()
                if mode == 'last':
                    app.annotator.next_annotation = None
                elif mode == 'structured':
                    app.experiment_mode = mode
                elif mode == 'cached':
                    app.annotator.next_annotation.features = torch.ones(1)
                app.receive_embedding_from_thread((torch.ones(1), (8, 8), (8, 8), 'current.jpg'))
                app.embed_img.assert_not_called()

    def test_loading_dialog_only_for_current_image(self):
        for name in ('current.jpg', 'next.jpg'):
            with self.subTest(name=name):
                app = self.make_app()
                app.embed_img(name)
                app.threadpool.start.assert_called_once()
                self.assertEqual(app.ui.create_basic_loading_window.call_count,
                                 int(name == 'current.jpg'))

    def test_navigation_reuses_cached_embedding_or_retries_missing_embedding(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                app = self.make_app()
                app.sam_gen = 1
                app.img_fnames = ['next.jpg', 'current.jpg']
                app.threadpool.activeThreadCount.return_value = 0
                app._pop_img_fnames = Mock(return_value=('current.jpg', 'next.jpg'))
                app.check_annotations_done = Mock(return_value=(False, False))
                app.propagate_good_masks = Mock()
                app.annotator.create_new_annotation.return_value = (False, True)
                if cached:
                    app.annotator.annotation.features = torch.ones(1)
                app.embed_img = Mock()
                app.select_next_img()
                if cached:
                    app.annotator.update_sam_features_to_current_annotation.assert_called_once()
                    app.propose_masks.assert_called_once()
                    app.embed_img.assert_called_once_with('next.jpg')
                else:
                    app.annotator.update_sam_features_to_current_annotation.assert_not_called()
                    app.embed_img.assert_called_once_with('current.jpg')

    def test_prefetch_completion_preserves_annotation_timer(self):
        app = self.make_app()
        app.threadpool.activeThreadCount.return_value = 0
        app.update_ui_imgs = Mock()
        app.embedding_done('next.jpg')
        app.annotator.init_time_stamp.assert_not_called()

    def test_worker_background_leaves_interaction_mutex_available(self):
        predictor = BackgroundThreadSamPredictor(TinySam())
        mutex = QMutex()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        worker = Sam1EmbeddingWorker(predictor, image, 'next.jpg', mutex)
        results = []
        worker.signals.result.connect(results.append)
        original_embed = predictor.embed_img

        def embed(*args, **kwargs):
            self.assertTrue(mutex.tryLock())
            mutex.unlock()
            return original_embed(*args, **kwargs)

        with patch.object(predictor, 'embed_img', side_effect=embed):
            worker.run()
        self.assertEqual(results[0][-1], 'next.jpg')
        self.assertFalse(predictor.is_image_set)

    def test_worker_releases_mutex_and_reports_failure(self):
        predictor = MainThreadSamPredictor(TinySam())
        mutex = QMutex()
        worker = Sam1EmbeddingWorker(predictor, np.zeros((8, 8, 3), dtype=np.uint8),
                                     'current.jpg', mutex)
        errors, finished = [], []
        worker.signals.error.connect(errors.append)
        worker.signals.finished.connect(finished.append)
        with patch.object(predictor, 'embed_img', side_effect=RuntimeError('failed')), \
                patch('aramsam_annotator.workers.traceback.print_exc'):
            worker.run()
        self.assertEqual(errors[0][0], RuntimeError)
        self.assertEqual(finished, ['current.jpg'])
        self.assertTrue(mutex.tryLock())
        mutex.unlock()


if __name__ == '__main__':
    unittest.main()
