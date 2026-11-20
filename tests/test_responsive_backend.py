from functools import partial
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
import unittest

import cv2
import numpy as np
from PyQt6.QtCore import QThread, QTimer
from PyQt6.QtWidgets import QApplication

from aramsam_annotator.app import App
from aramsam_annotator.backend.annotations import AnnotationObject, MaskData
from aramsam_annotator.backend.storage import AnnotationRepository
from aramsam_annotator.configs import AramsamConfigs, ImgTiles, SaveData
from aramsam_annotator.workers import BackendWorker


class ResponsiveBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config = AramsamConfigs(img_tiles=ImgTiles(do_tiling=False),
                               save_data=SaveData(mask_style='default'))
        with patch('aramsam_annotator.gui.get_monitors', return_value=[]):
            self.app = App(configs=config)
        self.addCleanup(self.cleanup_app)
        self.first, self.second = self.root / 'first.png', self.root / 'second.png'
        for path in (self.first, self.second):
            cv2.imwrite(str(path), np.zeros((12, 16, 3), np.uint8))

    def cleanup_app(self):
        self.app.shutdown()
        self.app.ui.hide()
        self.app.ui.deleteLater()

    def hold_until_dialog_is_painted(self, operation):
        """Hold the worker until GUI timer ticks observe its loading dialog."""
        started, release = Event(), Event()
        threads, messages = [], []
        timer = QTimer()

        def wrapped():
            threads.append(QThread.currentThread())
            started.set()
            if not release.wait(3):
                raise RuntimeError('GUI did not display a loading window while backend was running')
            return operation()

        def tick():
            window = self.app.ui.basic_loading_window
            if started.is_set() and window is not None and window.isVisible():
                messages.append((QThread.currentThread(), window.text))
                release.set()
                timer.stop()

        timer.timeout.connect(tick)
        timer.start(10)
        self.addCleanup(timer.stop)
        self.addCleanup(release.set)
        return wrapped, threads, messages

    def test_backend_work_shows_delayed_dialog_and_keeps_ui_timers_alive(self):
        operation, threads, messages = self.hold_until_dialog_is_painted(lambda: 42)
        value = self.app.run_backend_task(operation, 'Preparing images...', delay_ms=0)
        self.assertEqual(value, 42)
        self.assertNotEqual(threads[0], self.application.thread())
        self.assertEqual(messages, [(self.application.thread(), 'Preparing images...')])
        self.assertIsNone(self.app.ui.basic_loading_window)
        self.assertFalse(self.app.tasks.pending)

    def test_fast_work_does_not_flash_a_loading_window(self):
        with patch.object(self.app.ui, 'create_basic_loading_window') as loading:
            result = self.app.run_backend_task(lambda: 'ready', delay_ms=10000)
        self.assertEqual(result, 'ready')
        loading.assert_not_called()

    def test_backend_failure_closes_dialog_and_allows_retry(self):
        def fail():
            raise ValueError('cannot read input')

        operation, _, messages = self.hold_until_dialog_is_painted(fail)
        with patch('aramsam_annotator.workers.traceback.print_exc'):
            with self.assertRaisesRegex(ValueError, 'cannot read input'):
                self.app.run_backend_task(operation, 'Loading images...', delay_ms=0)
        self.assertTrue(messages)
        self.assertIsNone(self.app.ui.basic_loading_window)
        self.assertFalse(self.app.tasks.pending)
        self.assertEqual(self.app.run_backend_task(lambda: 'retry'), 'retry')

    def test_responsive_wait_includes_nested_backend_work_and_followup_jobs(self):
        app = self.app
        results = []

        def received(_):
            results.append(app.run_backend_task(lambda: 'nested'))
            app.tasks.submit(BackendWorker(lambda: 'following'), result=lambda value: results.append(value[0]))

        app.tasks.submit(BackendWorker(lambda: 'initial'), result=received)
        app.wait_for_workers()
        self.assertEqual(results, ['nested', 'following'])
        self.assertFalse(app.tasks.pending)

    def test_embedding_callbacks_can_prepare_views_and_prefetch_without_blocking_each_other(self):
        app = self.app
        app.sam_gen = 1
        predictor = Mock()
        predictor.embed_img.return_value = (np.zeros((1,), np.float32), (12, 16), (12, 16))
        app.annotator.sam = SimpleNamespace(predictor=predictor)
        app.output_dir = str(self.root)
        app.img_fnames = [str(self.second), str(self.first)]
        app.select_next_img()
        app.wait_for_workers()
        self.assertIsNotNone(app.annotator.annotation.features)
        self.assertIsNotNone(app.annotator.next_annotation.features)
        self.assertFalse(app.tasks.pending)
        self.assertIsNone(app.ui.basic_loading_window)

    def test_batched_and_unbatched_sam2_and_sam3_propagation_are_off_ui_thread(self):
        app = self.app
        for generation in (2, 3):
            for batch_size in (None, 1):
                with self.subTest(generation=generation, batch_size=batch_size):
                    app.annotator.annotation = AnnotationObject(self.first)
                    app.annotator.next_annotation = AnnotationObject(self.second)
                    mask = np.zeros((12, 16), np.uint8)
                    mask[2:7, 3:9] = 255
                    source = MaskData(mid=7, origin='Sam3_text', mask=mask)
                    app.annotator.annotation.good_masks = [source]
                    propagated = MaskData(mid=7, origin=f'Sam{generation}_tracking', mask=mask)
                    operation, threads, messages = self.hold_until_dialog_is_painted(lambda: [propagated])
                    tracking_threads = []
                    model = SimpleNamespace(
                        device='cuda', prop_thread_func=Mock(side_effect=lambda batch: operation()),
                        set_tracking_pair=Mock(side_effect=lambda *args: tracking_threads.append(QThread.currentThread())))
                    app.annotator.sam = model
                    app.sam_gen = generation
                    app.propagated_mids = set()
                    app.mask_track_batch_size = batch_size
                    original_wait = App.wait_for_workers.__get__(app)
                    app.wait_for_workers = lambda **kwargs: original_wait(delay_ms=0, **kwargs)
                    app.start_mask_batch_thread(track_remaining=True)
                    self.assertNotEqual(threads[0], self.application.thread())
                    self.assertTrue(messages)
                    self.assertIn('Propagating masks', messages[0][1])
                    self.assertEqual(app.annotator.next_annotation.masks, [propagated])
                    self.assertEqual(app.annotator.next_annotation.mask_decisions, [True])
                    self.assertIsNone(app.ui.basic_loading_window)
                    if generation == 3:
                        self.assertEqual(tracking_threads, threads)

    def test_output_chooser_stays_on_ui_while_checks_and_image_preparation_use_workers(self):
        app = self.app
        app.run_backend_task = partial(app.run_backend_task, delay_ms=0)
        app.sam_gen = 1
        app.annotator.sam = Mock()
        app.ui.auto_embed_box.setChecked(False)
        app.img_fnames = [str(self.second), str(self.first)]
        chooser_threads, check_threads, image_threads = [], [], []

        def choose_output():
            chooser_threads.append(QThread.currentThread())
            app.change_output_dir(str(self.root))

        def check_saved(*args):
            self.assertEqual(app.output_dir, str(self.root))
            check_threads.append(QThread.currentThread())
            return False

        def load_image(path):
            self.assertTrue(chooser_threads)
            image_threads.append(QThread.currentThread())
            return AnnotationObject(path)

        app.ui.open_ouput_dir_selection = Mock(side_effect=choose_output)
        app.annotator.session.annotation_factory = load_image
        with patch.object(AnnotationRepository, 'exists', side_effect=check_saved):
            app.select_next_img()
        self.assertEqual(chooser_threads, [self.application.thread()])
        self.assertEqual(len(check_threads), 2)
        self.assertEqual(len(image_threads), 2)
        self.assertTrue(all(thread != self.application.thread() for thread in check_threads + image_threads))
        self.assertEqual(app.annotator.annotation.img_name, 'first.png')

    def test_saving_after_output_selection_is_off_ui_thread(self):
        app = self.app
        app.run_backend_task = partial(app.run_backend_task, delay_ms=0)
        app.annotator.annotation = AnnotationObject(self.first)
        app.annotator.annotation.good_masks = [MaskData(mid=1, origin='Polygon_drawing',
                                                      mask=np.ones((12, 16), np.uint8) * 255)]
        app.ui.open_ouput_dir_selection = Mock(side_effect=lambda: app.change_output_dir(str(self.root)))
        original_save = app.annotator.save_annotations
        operation, threads, messages = self.hold_until_dialog_is_painted(lambda: original_save(app.output_dir))
        app.annotator.save_annotations = Mock(side_effect=lambda *args, **kwargs: operation())
        app.save_output()
        self.assertNotEqual(threads[0], self.application.thread())
        self.assertIn('Saving annotations', messages[0][1])
        self.assertTrue((self.root / 'first_annots' / 'log.json').exists())

    def test_image_tiling_is_off_ui_thread(self):
        app = self.app
        app.configs.img_tiles.do_tiling = True
        app.img_fnames = [str(self.first)]
        threads = []

        def tile(image, directory, settings):
            threads.append(QThread.currentThread())
            return [image]

        with patch('aramsam_annotator.backend.session.split_image_into_tiles', side_effect=tile):
            self.assertEqual(app._pop_img_fnames(), (self.first, None))
        self.assertNotEqual(threads[0], self.application.thread())


if __name__ == '__main__':
    unittest.main()
