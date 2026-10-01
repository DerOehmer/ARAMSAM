import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QThreadPool
from PyQt6.QtWidgets import QApplication, QPushButton

from aramsam_annotator.app import App
from aramsam_annotator.annotator import Annotator
from aramsam_annotator.backend.annotations import AnnotationObject, MaskData
from aramsam_annotator.configs import AramsamConfigs, ImgTiles, SaveData


class NavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def make_app(self):
        app = App.__new__(App)
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
        app.annotator = Annotator(app.configs)
        app.sam_gen = 1
        app.annotator.set_sam_version = Mock(side_effect=lambda *a: setattr(app.annotator, 'sam', Mock()))
        app.set_sam()
        app.set_sam()
        self.assertEqual(app.annotator.set_sam_version.call_count, 1)
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
