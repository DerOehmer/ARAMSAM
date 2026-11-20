import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QCheckBox

from aramsam_annotator.app import App
from aramsam_annotator.annotator import Annotator
from aramsam_annotator.backend.annotations import MaskData
from aramsam_annotator.configs import AramsamConfigs, SaveData, load_configs_from_yaml


class PropagationConfigTests(unittest.TestCase):
    def test_yaml_propagation_defaults_and_overrides(self):
        for source, expected in (('', True), ('mask_propagation: true', True),
                                 ('mask_propagation: false', False)):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'configs.yaml'
                path.write_text(source)
                self.assertEqual(load_configs_from_yaml(path).mask_propagation, expected)
        self.assertTrue(AramsamConfigs().mask_propagation)

    def test_disabled_propagation_blocks_all_generation_entry_points(self):
        for generation in (1, 2, 3):
            with self.subTest(generation=generation):
                app = App.__new__(App)
                app.configs = AramsamConfigs(mask_propagation=False)
                app.sam_gen = generation
                app.annotator = Mock()
                app.bbox_tracker = Mock()
                app.threadpool = Mock()
                app.propagate_good_masks()
                app.start_mask_batch_thread(track_remaining=True)
                app.bbox_tracker.add_annotation.assert_not_called()
                app.annotator.sam.set_masks.assert_not_called()
                app.annotator.sam.set_tracking_pair.assert_not_called()
                app.threadpool.start.assert_not_called()
                self.assertNotIn('tasks', app.__dict__)

    def test_disabled_propagation_blocks_panorama_tracking_during_amg(self):
        annotator = Annotator(AramsamConfigs(mask_propagation=False))
        annotator.sam = Mock()
        annotator.annotation = SimpleNamespace(img=np.zeros((4, 4, 3), np.uint8),
                                              masks=[], good_masks=[], mask_decisions=[])
        annotator._propagate_bboxes = Mock()
        annotator.prepare_amg(Mock())
        annotator._propagate_bboxes.assert_not_called()
        annotator.sam.amg.set_visualization_img.assert_called_once()


class SettingsUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def make_app(self, propagation=True):
        config = AramsamConfigs(mask_propagation=propagation, save_data=SaveData(auto_save=False))
        with patch('aramsam_annotator.gui.get_monitors', return_value=[]):
            app = App(configs=config)

        def cleanup():
            app.shutdown()
            app.ui.hide()
            app.ui.deleteLater()

        self.addCleanup(cleanup)
        return app

    def test_checkboxes_are_in_settings_and_auto_save_and_embed_still_work(self):
        app = self.make_app(propagation=False)
        ui = app.ui
        checkboxes = [ui.auto_save_box, ui.auto_embed_box, ui.mask_propagation_box]
        self.assertEqual([action.defaultWidget() for action in ui.settings_checkbox_actions], checkboxes)
        for action in ui.settings_checkbox_actions:
            self.assertIn(action, ui.menu_settings.actions())
        toolbar_checkboxes = ui.findChildren(QCheckBox, options=Qt.FindChildOption.FindDirectChildrenOnly)
        for checkbox in checkboxes:
            self.assertNotIn(checkbox, toolbar_checkboxes)
        self.assertFalse(ui.auto_save_box.isChecked())
        self.assertFalse(ui.mask_propagation_box.isChecked())
        ui.auto_save_box.click()
        self.assertTrue(app.configs.save_data.auto_save)
        app.annotator.annotation = SimpleNamespace(img_name='current.png')
        app.embed_img = Mock()
        app.sam_gen = 1
        ui.auto_embed_box.click()
        self.assertFalse(ui.manual_annotation_button.isEnabled())
        ui.auto_embed_box.click()
        self.assertTrue(ui.manual_annotation_button.isEnabled())
        app.embed_img.assert_called_once_with('current.png')
        ui.mask_propagation_box.click()
        self.assertTrue(app.configs.mask_propagation)

    def test_disabling_propagation_cleans_up_inflight_results_and_preserves_other_masks(self):
        app = self.make_app()
        source = MaskData(mid=7, origin='Sam3_text', mask=np.ones((4, 4), np.uint8))
        propagated = MaskData(mid=7, origin='Sam3_tracking', mask=source.mask)
        inflight = MaskData(mid=8, origin='Sam2_tracking', mask=source.mask)
        from_disk = MaskData(mid=99, origin='Sam3_tracking', mask=source.mask)
        polygon = MaskData(mid=10, origin='Polygon_drawing', mask=source.mask)
        current = SimpleNamespace(good_masks=[source])
        following = SimpleNamespace(masks=[propagated, from_disk, polygon],
                                    mask_decisions=[True, True, False],
                                    good_masks=[propagated, from_disk])
        app.annotator.annotation = current
        app.annotator.next_annotation = following
        app.propagated_mids = {7, 8}
        app.bbox_tracker = Mock()

        def finish_inflight_batch():
            self.assertFalse(app.configs.mask_propagation)
            following.masks.append(inflight)
            following.mask_decisions.append(True)
            following.good_masks.append(inflight)

        app.wait_for_workers = Mock(side_effect=finish_inflight_batch)
        app.ui.mask_propagation_box.click()
        app.wait_for_workers.assert_called_once()
        self.assertEqual(following.masks, [from_disk, polygon])
        self.assertEqual(following.mask_decisions, [True, False])
        self.assertEqual(following.good_masks, [from_disk])
        self.assertEqual(current.good_masks, [source])
        self.assertEqual(app.propagated_mids, set())
        self.assertIsNone(app.bbox_tracker)
        # A fresh enable allows the same accepted source masks to propagate again.
        app.ui.mask_propagation_box.click()
        app.sam_gen = 3
        app.start_mask_batch_thread = Mock()
        app.propagate_good_masks()
        app.start_mask_batch_thread.assert_called_once_with(track_remaining=True)

    def test_enabled_propagation_keeps_existing_sam1_and_sam2_workflows(self):
        for generation in (1, 2):
            with self.subTest(generation=generation):
                app = self.make_app()
                app.sam_gen = generation
                app.annotator.annotation = SimpleNamespace(good_masks=[Mock()])
                app.annotator.next_annotation = SimpleNamespace(masks=[])
                app.bbox_tracker = Mock()
                app.start_mask_batch_thread = Mock()
                app.propagate_good_masks()
                if generation == 1:
                    app.bbox_tracker.add_annotation.assert_called_once_with(app.annotator.annotation)
                    app.start_mask_batch_thread.assert_not_called()
                else:
                    app.start_mask_batch_thread.assert_called_once_with(track_remaining=True)


if __name__ == '__main__':
    unittest.main()
