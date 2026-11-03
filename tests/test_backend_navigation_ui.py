import tempfile
import unittest
from threading import Event
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QThreadPool, Qt, QThread, QTimer
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget

from aramsam_annotator.app import App
from aramsam_annotator.annotator import Annotator
from aramsam_annotator.backend.annotations import AnnotationObject, MaskData
from aramsam_annotator.configs import AramsamConfigs, ImgTiles, SaveData
from aramsam_annotator.gui import BasicLoadingWindow


class NavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def make_app(self):
        app = App.__new__(App)
        app.run_backend_task = Mock(side_effect=lambda operation, *args, **kwargs: operation())
        app.configs = AramsamConfigs(img_tiles=ImgTiles(do_tiling=False))
        app.annotator = Mock()
        app.annotator.annotation = None
        app.annotator.next_annotation = None
        app.experiment_mode = None
        app.experiment_progress = None
        app.tutorial_flag = False
        app.ui = Mock()
        app.ui.open_img_load_file_dialog.return_value = ''
        app.ui.open_load_folder_dialog.return_value = ''
        app.output_dir = None
        app.threadpool = Mock()
        app.threadpool.activeThreadCount.return_value = 0
        app.img_fnames = []
        app.temp_dir = tempfile.gettempdir()
        return app

    def test_cancelled_load_dialogs_do_not_initialize_model_or_change_queue(self):
        for method in ('load_img', 'load_img_folder'):
            app = self.make_app()
            app.set_sam = Mock()
            app.img_fnames = ['keep.png']
            getattr(app, method)(None)
            app.set_sam.assert_not_called()
            app.ui.create_basic_loading_window.assert_not_called()
            self.assertEqual(app.img_fnames, ['keep.png'])

    def test_cancelled_save_dialog_does_not_recurse(self):
        app = self.make_app()
        app.annotator.annotation = Mock()
        app.save_output()
        app.ui.open_ouput_dir_selection.assert_called_once()
        app.ui.save.assert_not_called()
        app.annotator.save_annotations.assert_not_called()

    def test_completed_folder_does_not_recurse_or_create_annotations(self):
        app = self.make_app()
        app.img_fnames = [f'{index}.jpg' for index in range(2000)]
        app.check_annotations_done = Mock(return_value=(True, True))
        app.select_next_img()
        self.assertEqual(app.img_fnames, [])
        app.annotator.create_new_annotation.assert_not_called()
        app.ui.disable_push_buttons.assert_called_once()

    def test_failed_annotation_creation_restores_queue(self):
        app = self.make_app()
        app.img_fnames = ['first.jpg', 'second.jpg']
        app.check_annotations_done = Mock(return_value=(False, False))
        app.annotator.create_new_annotation.side_effect = OSError('unreadable image')
        app.propagate_good_masks = Mock()
        with self.assertRaises(OSError):
            app.select_next_img()
        self.assertEqual(app.img_fnames, ['first.jpg', 'second.jpg'])
        self.assertFalse(app.navigation.changing_image)

    def test_saved_predecessor_loads_before_advancing(self):
        app = self.make_app()
        app.img_fnames = ['new.jpg', 'saved.jpg']
        app.sam_gen = 2
        app.check_annotations_done = Mock(return_value=(True, False))
        app.annotator.create_new_annotation.return_value = (True, False)
        app.load_previous_annotations = Mock()
        app.propagate_good_masks = Mock()
        app.embed_img_pair = Mock()
        app.select_next_img()
        app.load_previous_annotations.assert_called_once_with(Path('saved.jpg'), Path('new.jpg'))
        app.annotator.create_new_annotation.assert_called_once_with(Path('new.jpg'), None)
        app.embed_img_pair.assert_called_once()

    def test_model_reuse_and_switch_invalidates_embeddings(self):
        app = self.make_app()
        app.threadpool = QThreadPool()
        app.annotator = Annotator(app.configs)
        app.sam_gen = 1
        app.annotator.set_sam_version = Mock(side_effect=lambda *a: setattr(app.annotator, 'sam', Mock()))
        app.set_sam()
        app.set_sam()
        self.assertEqual(app.annotator.set_sam_version.call_count, 1)
        app.ui.create_basic_loading_window.assert_called_once()
        app.ui.close_basic_loading_window.assert_called_once()
        app.annotator.annotation = Mock()
        app.embed_img = Mock()
        app.changed_sam_model('another_vit_b.pth')
        self.assertEqual(app.annotator.set_sam_version.call_count, 2)
        app.annotator.annotation.set_sam_parameters.assert_called_once_with(None, None, None)
        app.ui.construct_ui.assert_not_called()

    def test_old_model_result_does_not_reach_new_image(self):
        app = self.make_app()
        app.threadpool = QThreadPool()
        app.mutex = QMutex()
        app.annotator.annotation = Mock()
        app.annotator.next_annotation = None
        app.annotator.yolo.infer_image.return_value = []
        app.receive_yolo_results = Mock()
        app.start_yolo_worker()
        app.threadpool.waitForDone(-1)
        app.annotator.annotation = Mock()
        app.wait_for_workers()
        app.receive_yolo_results.assert_not_called()

    def test_propagation_purge_keeps_decisions_aligned(self):
        app = self.make_app()
        app.annotator.annotation = Mock(good_masks=[MaskData(mid=2, origin='Sam2_tracking')])
        masks = [MaskData(mid=i, origin='Sam2_tracking') for i in (1, 2, 3)]
        app.annotator.next_annotation = Mock(masks=masks, good_masks=masks[:], mask_decisions=[True, False, True])
        app._purge_falsely_propagated_masks()
        self.assertEqual([m.mid for m in app.annotator.next_annotation.masks], [2])
        self.assertEqual(app.annotator.next_annotation.mask_decisions, [False])


class QtWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = AramsamConfigs(img_tiles=ImgTiles(do_tiling=False),
                                    save_data=SaveData(mask_style='default'))

    def make_app(self, mode=None):
        with patch('aramsam_annotator.gui.get_monitors', return_value=[]):
            app = App(configs=self.config, experiment_mode=mode)
        def cleanup():
            app.shutdown()
            app.ui.hide()
            app.ui.deleteLater()
        self.addCleanup(cleanup)
        return app

    def test_loading_dialog_wraps_long_messages_without_clipping_at_large_fonts(self):
        parent = QWidget()
        self.addCleanup(parent.deleteLater)
        message = ('Loading SAM3 model from the selected checkpoint. Initial startup may take a while. ' * 4
                   + '\nPlease keep this window open until processing finishes. <No percentage available>')
        for point_size in (10, 24):
            with self.subTest(point_size=point_size):
                font = parent.font()
                font.setPointSize(point_size)
                parent.setFont(font)
                dialog = BasicLoadingWindow(parent, message)
                self.addCleanup(dialog.deleteLater)
                dialog.show()
                self.application.processEvents()
                label = dialog.message_label
                self.assertEqual(label.text(), message)
                self.assertEqual(label.textFormat(), Qt.TextFormat.PlainText)
                self.assertTrue(label.wordWrap())
                self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()))
                self.assertTrue(dialog.rect().contains(label.geometry()))
                self.assertLess(label.geometry().bottom(), dialog.progress_bar.y())
                self.assertGreater(dialog.height(), 100)
                self.assertLessEqual(dialog.width(), dialog.screen().availableGeometry().width())
                self.assertEqual((dialog.progress_bar.minimum(), dialog.progress_bar.maximum()), (0, 0))
                dialog.close()

    def test_loading_dialog_height_grows_with_message(self):
        short = BasicLoadingWindow(text='Loading SAM3 model...')
        long = BasicLoadingWindow(text='Loading the selected image and preparing its embedding. ' * 12)
        for dialog in (short, long):
            self.addCleanup(dialog.deleteLater)
            dialog.show()
            self.application.processEvents()
        self.assertGreater(long.height(), short.height())
        short.close()
        long.close()

    def test_loading_dialog_stays_responsive_while_model_loads_in_another_thread(self):
        image = self.root / 'input.png'
        cv2.imwrite(str(image), np.zeros((32, 48, 3), np.uint8))
        for generation in (1, 3):
            for method in ('load_img', 'load_img_folder'):
                with self.subTest(method=method, generation=generation):
                    self._check_threaded_loading(image, method, generation)

    def _check_threaded_loading(self, image, method, generation):
        app = self.make_app()
        app.configs.sam_configs.gen = generation
        app.ui.open_img_load_file_dialog = Mock(return_value=str(image))
        app.ui.open_load_folder_dialog = Mock(return_value=str(self.root))
        app.select_next_img = Mock()
        painted = []
        started, release = Event(), Event()
        worker_threads, ticks = [], []

        class PaintedLoadingWindow(BasicLoadingWindow):
            def paintEvent(self, event):
                painted.append(True)
                super().paintEvent(event)

        def load_model(*args):
            worker_threads.append(QThread.currentThread())
            started.set()
            if not release.wait(timeout=3):
                raise RuntimeError('GUI did not process events while loading')
            app.annotator.sam = Mock()

        timer = QTimer()

        def gui_tick():
            if not started.is_set():
                return
            window = app.ui.basic_loading_window
            ticks.append((QThread.currentThread(), window.isVisible(), window.text, bool(painted)))
            if len(ticks) >= 2:
                release.set()
                timer.stop()

        timer.timeout.connect(gui_tick)
        timer.start(10)
        app.annotator.set_sam_version = Mock(side_effect=load_model)
        try:
            with patch('aramsam_annotator.gui.BasicLoadingWindow', PaintedLoadingWindow):
                getattr(app, method)(None)
        finally:
            release.set()
            timer.stop()
        self.assertGreaterEqual(len(ticks), 2)
        self.assertNotEqual(worker_threads[0], self.application.thread())
        for thread, visible, text, was_painted in ticks:
            self.assertEqual(thread, self.application.thread())
            self.assertTrue(visible)
            self.assertIn(f'Loading SAM{generation} model', text)
            self.assertTrue(was_painted)
        app.annotator.set_sam_version.assert_called_once()
        app.select_next_img.assert_called_once()
        self.assertIsNone(app.ui.basic_loading_window)
        self.assertFalse(app.tasks.pending)

    def test_failed_model_load_closes_loading_dialog_for_image_and_folder(self):
        image = self.root / 'input.png'
        cv2.imwrite(str(image), np.zeros((32, 48, 3), np.uint8))
        for method in ('load_img', 'load_img_folder'):
            with self.subTest(method=method):
                app = self.make_app()
                app.ui.open_img_load_file_dialog = Mock(return_value=str(image))
                app.ui.open_load_folder_dialog = Mock(return_value=str(self.root))
                app.annotator.set_sam_version = Mock(side_effect=RuntimeError('model failed'))
                app.select_next_img = Mock()
                with self.assertRaisesRegex(RuntimeError, 'model failed'):
                    getattr(app, method)(None)
                self.assertIsNone(app.ui.basic_loading_window)
                app.select_next_img.assert_not_called()
                self.assertFalse(app.tasks.pending)

    def test_sam2_restores_gui_thread_precision_after_background_initialization(self):
        app = self.make_app()
        app.configs.sam_configs.gen = 2
        app.annotator.device = 'cuda'
        model = Mock()
        precision_threads = []
        model._init_mixed_precision.side_effect = lambda: precision_threads.append(QThread.currentThread())
        app.annotator.set_sam_version = Mock(side_effect=lambda *args: setattr(app.annotator, 'sam', model))
        app.set_sam()
        self.assertEqual(precision_threads, [self.application.thread()])

    def test_actual_ui_polygon_save_reload_delete_and_resave(self):
        app = self.make_app()
        self.assertEqual(app.ui.next_img_button.text(), 'next image')
        self.assertEqual(len(app.ui.annotation_visualizers), 4)
        image = self.root / 'sample.png'
        cv2.imwrite(str(image), np.zeros((32, 48, 3), np.uint8))
        app.output_dir = str(self.root)
        app.sam_gen = 1
        app.annotator.sam = Mock()
        app.ui.auto_embed_box.setChecked(False)
        app.img_fnames = [str(image)]
        app.select_next_img()
        app.ui.draw_button.click()
        app.annotator.manual_mask_points = [(3, 3), (20, 3), (20, 20)]
        app.annotator.mask_from_drawing()
        app.ui.good_mask_button.click()
        self.assertEqual(len(app.annotator.annotation.good_masks), 1)
        app.ui.save()
        self.assertTrue((self.root / 'sample_annots' / 'log.json').exists())
        app.load_previous_annotations(image, None)
        self.assertEqual(len(app.annotator.annotation.good_masks), 1)
        app.annotator.delete_mask(app.annotator.annotation.good_masks[0].mid)
        app.annotator.update_collections(app.annotator.annotation)
        app.ui.save()
        self.assertEqual(list((self.root / 'sample_annots' / 'masks').glob('*.png')), [])

    def test_previous_button_restores_disk_annotations_and_forward_queue(self):
        app = self.make_app()
        app.output_dir = str(self.root)
        app.sam_gen = 1
        app.annotator.sam = Mock()
        app.ui.auto_embed_box.setChecked(False)
        first, second = self.root / 'first.png', self.root / 'second.png'
        for image in (first, second):
            cv2.imwrite(str(image), np.zeros((32, 48, 3), np.uint8))
        app.img_fnames = [str(second), str(first)]
        app.select_next_img()
        self.assertFalse(app.ui.previous_img_button.isEnabled())
        self.assertLessEqual(app.ui.previous_img_button.x() + app.ui.previous_img_button.width(),
                             app.ui.next_img_button.x())
        previous, following = app.ui.previous_img_button, app.ui.next_img_button
        self.assertEqual(previous.width(), previous.height())
        self.assertLessEqual(following.x() - (previous.x() + previous.width()), 1)
        self.assertLessEqual(abs(previous.geometry().center().y() - following.geometry().center().y()), 1)
        mask = np.zeros((32, 48), np.uint8)
        mask[4:12, 4:12] = 255
        app.annotator.annotation.add_masks([MaskData(mid=1, origin='Polygon_drawing', mask=mask)], decision=True)
        app.annotator.annotation.good_masks = list(app.annotator.annotation.masks)
        app.select_next_img()
        self.assertTrue(app.ui.previous_img_button.isEnabled())
        app.ui.previous_img_button.click()
        self.assertEqual(app.annotator.annotation.filepath, str(first))
        self.assertEqual(len(app.annotator.annotation.good_masks), 1)
        np.testing.assert_array_equal(app.annotator.annotation.good_masks[0].mask, mask)
        self.assertEqual(app.img_fnames, [str(second)])
        app.select_next_img()
        self.assertEqual(app.annotator.annotation.filepath, str(second))
        self.assertEqual(app.img_fnames, [])

    def test_previous_without_saved_annotations_and_after_queue_end(self):
        app = self.make_app()
        app.configs.save_data.do_save = False
        app.output_dir = str(self.root)
        app.sam_gen = 1
        app.annotator.sam = Mock()
        app.ui.auto_embed_box.setChecked(False)
        images = [self.root / f'{i}.png' for i in range(2)]
        for image in images:
            cv2.imwrite(str(image), np.zeros((16, 16, 3), np.uint8))
        app.img_fnames = [str(image) for image in reversed(images)]
        app.select_next_img()
        app.select_next_img()
        with patch.object(app.ui, 'create_message_box', return_value=False):
            app.select_next_img()
        self.assertTrue(app.ui.previous_img_button.isEnabled())
        app.ui.previous_img_button.click()
        self.assertEqual(app.annotator.annotation.filepath, str(images[0]))
        self.assertEqual(app.annotator.annotation.good_masks, [])

    def test_ui_callback_exception_is_reported_without_escaping_qt(self):
        app = self.make_app()
        app.print_thread_error = Mock()
        button = QPushButton()
        def fail():
            raise OSError('Could not save annotation')
        app._connect_ui(button.clicked, fail)
        button.click()
        app.print_thread_error.assert_called_once()
        self.assertIs(app.print_thread_error.call_args.args[0][0], OSError)

    def test_experiment_modes_construct_and_keep_existing_next_control(self):
        for mode in ('structured', 'polygon', 'tutorial'):
            with self.subTest(mode=mode):
                app = self.make_app(mode)
                self.assertEqual(app.ui.next_method_button.text(), 'Next')
                self.assertTrue(app.ui.auto_save_box.isHidden())
                self.assertEqual(len(app.ui.annotation_visualizers), 4)


if __name__ == '__main__':
    unittest.main()
