from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import yaml
from PyQt6.QtWidgets import QApplication

from aramsam_annotator.app import App
from aramsam_annotator.configs import (
    AramsamConfigs, ImgTiles, SaveData, find_configs_path,
    load_configs_from_yaml, save_configs_to_yaml,
)
from aramsam_annotator.welcome import WelcomeWindow


class ConfigPersistenceTests(unittest.TestCase):
    def test_resolver_prefers_cwd_and_uses_same_file_for_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'configs.yaml'
            path.write_text('mask_propagation: false\n')
            with patch('aramsam_annotator.configs.Path.cwd', return_value=root):
                self.assertEqual(find_configs_path(), path)
                self.assertFalse(load_configs_from_yaml(find_configs_path()).mask_propagation)
                self.assertEqual(find_configs_path('new-settings.yaml'), root / 'new-settings.yaml')

    def test_settings_round_trip_and_unknown_top_level_keys_are_preserved(self):
        config = AramsamConfigs(use_gpu=False, mask_propagation=False,
                               img_tiles=ImgTiles(do_tiling=False, tile_size=256),
                               class_dict={'7': 'Seed', '9': 'Cup'})
        config.sam_configs.model_type = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'configs.yaml'
            path.write_text('custom_metadata: keep me\n')
            save_configs_to_yaml(config, path)
            self.assertEqual(asdict(load_configs_from_yaml(path)), asdict(config))
            self.assertEqual(yaml.safe_load(path.read_text())['custom_metadata'], 'keep me')

    def test_failed_replace_preserves_existing_configuration_and_cleans_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'configs.yaml'
            original = b'mask_propagation: false\n'
            path.write_bytes(original)
            with patch('aramsam_annotator.configs.os.replace', side_effect=OSError('cannot replace')):
                with self.assertRaisesRegex(OSError, 'cannot replace'):
                    save_configs_to_yaml(AramsamConfigs(), path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob('.configs.yaml.*')), [])


class WelcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / 'image.png'
        cv2.imwrite(str(self.image), np.zeros((12, 16, 3), np.uint8))
        self.counter = 0

    def make_app(self):
        self.counter += 1
        config = AramsamConfigs(use_gpu=False, img_tiles=ImgTiles(do_tiling=False),
                               save_data=SaveData(save_dir=str(self.root / 'output')))
        path = self.root / f'settings-{self.counter}.yaml'
        save_configs_to_yaml(config, path)
        with patch('aramsam_annotator.gui.get_monitors', return_value=[]):
            app = App(configs=config, config_path=path)

        def cleanup():
            if app.welcome_window is not None:
                app.welcome_window.hide()
                app.welcome_window.deleteLater()
            app.shutdown()
            app.ui.hide()
            app.ui.deleteLater()

        self.addCleanup(cleanup)
        return app

    def show_welcome(self, app):
        app.show_welcome()
        self.application.processEvents()
        return app.welcome_window

    def test_run_schedules_welcome_before_any_model_loading(self):
        app = self.make_app()
        app.annotator.set_sam_version = Mock()
        with patch.object(app.application, 'exec', return_value=0), \
                patch('aramsam_annotator.app.QTimer.singleShot') as schedule:
            with self.assertRaises(SystemExit):
                app.run()
        schedule.assert_called_once_with(0, app.show_welcome)
        app.annotator.set_sam_version.assert_not_called()

    def test_all_config_settings_have_controls_and_untouched_values_round_trip(self):
        app = self.make_app()
        app.configs.sam_configs.confidence_threshold = 0.123456789012345
        dialog = self.show_welcome(app)
        expected = set()
        for key, value in asdict(app.configs).items():
            if isinstance(value, dict) and key != 'class_dict':
                expected.update(f'{key}.{name}' for name in value)
            else:
                expected.add(key)
        self.assertEqual(set(dialog.controls), expected)
        self.assertEqual(asdict(dialog.read_configs()), asdict(app.configs))
        self.assertLess(dialog.quit_button.x(), dialog.image_button.x())
        self.assertLess(dialog.image_button.x(), dialog.folder_button.x())
        self.assertEqual(dialog.quit_button.y(), dialog.folder_button.y())

    def test_generation_selection_updates_default_checkpoint_but_preserves_custom_paths(self):
        dialog = self.show_welcome(self.make_app())
        dialog.controls['sam_configs.gen'].setCurrentIndex(2)
        self.assertEqual(dialog.controls['sam_configs.model_ckpt_p'].text(), 'sam3.pt')
        self.assertEqual(dialog.controls['sam_configs.model_type'].text(), '')
        dialog.controls['sam_configs.model_ckpt_p'].setText('/custom/my-model.pt')
        dialog.controls['sam_configs.gen'].setCurrentIndex(1)
        self.assertEqual(dialog.controls['sam_configs.model_ckpt_p'].text(), '/custom/my-model.pt')

    def test_both_load_buttons_save_edits_before_choosing_input_and_apply_settings(self):
        for mode in ('image', 'folder'):
            with self.subTest(mode=mode):
                app = self.make_app()
                dialog = self.show_welcome(app)
                dialog.controls['use_gpu'].setChecked(True)
                dialog.controls['sam_configs.gen'].setCurrentIndex(2)
                dialog.controls['mask_propagation'].setChecked(False)
                dialog.controls['img_tiles.do_tiling'].setChecked(True)
                dialog.controls['img_tiles.tile_size'].setValue(256)
                dialog.controls['img_tiles.tile_overlap'].setValue(0.35)
                dialog.controls['save_data.save_bboxes'].setChecked(True)
                dialog.controls['save_data.save_dir'].setText(str(self.root / 'changed-output'))
                dialog.classes_editor.setPlainText('"7": Seed\n"9": Cup\n')
                app.navigation.load_image_path = Mock()
                app.navigation.load_folder_path = Mock()

                def choose_image(*args):
                    self.assertEqual(load_configs_from_yaml(app.config_path).sam_configs.gen, 3)
                    return str(self.image), ''

                def choose_folder(*args):
                    self.assertFalse(load_configs_from_yaml(app.config_path).mask_propagation)
                    return str(self.root)

                with patch('aramsam_annotator.app.QFileDialog.getOpenFileName', side_effect=choose_image), \
                        patch('aramsam_annotator.app.QFileDialog.getExistingDirectory', side_effect=choose_folder):
                    (dialog.image_button if mode == 'image' else dialog.folder_button).click()
                saved = load_configs_from_yaml(app.config_path)
                self.assertEqual(asdict(app.configs), asdict(saved))
                self.assertEqual(saved.sam_configs.gen, 3)
                self.assertEqual(saved.sam_configs.model_ckpt_p, 'sam3.pt')
                self.assertIsNone(saved.sam_configs.model_type)
                self.assertFalse(saved.mask_propagation)
                self.assertTrue(saved.img_tiles.do_tiling)
                self.assertEqual(saved.img_tiles.tile_size, 256)
                self.assertEqual(saved.img_tiles.tile_overlap, 0.35)
                self.assertTrue(saved.save_data.save_bboxes)
                self.assertFalse(saved.save_data.save_masks)
                self.assertFalse(app.ui.mask_propagation_box.isChecked())
                self.assertTrue(app.ui.text_prompt_toggle.isEnabled())
                self.assertEqual(app.ui.get_selected_class(), 7)
                app.ui.class_actions['9'].trigger()
                self.assertEqual(app.ui.get_selected_class(), 9)
                self.assertEqual(app.output_dir, str(self.root / 'changed-output'))
                self.assertTrue(Path(app.output_dir).is_dir())
                self.assertTrue(dialog.isHidden())
                if mode == 'image':
                    app.navigation.load_image_path.assert_called_once_with(str(self.image))
                    app.navigation.load_folder_path.assert_not_called()
                else:
                    app.navigation.load_folder_path.assert_called_once_with(str(self.root))
                    app.navigation.load_image_path.assert_not_called()

    def test_unchanged_settings_do_not_rewrite_configuration(self):
        app = self.make_app()
        original = app.config_path.read_bytes()
        dialog = self.show_welcome(app)
        app.navigation.load_image_path = Mock()
        with patch('aramsam_annotator.app.QFileDialog.getOpenFileName', return_value=(str(self.image), '')), \
                patch('aramsam_annotator.app.save_configs_to_yaml') as save:
            dialog.image_button.click()
        save.assert_not_called()
        self.assertEqual(app.config_path.read_bytes(), original)
        app.navigation.load_image_path.assert_called_once()

    def test_cancelled_input_keeps_welcome_open_and_edits_are_already_saved(self):
        app = self.make_app()
        dialog = self.show_welcome(app)
        dialog.controls['mask_propagation'].setChecked(False)
        app.set_sam = Mock()
        with patch('aramsam_annotator.app.QFileDialog.getOpenFileName', return_value=('', '')):
            dialog.image_button.click()
        self.assertTrue(dialog.isVisible())
        self.assertFalse(load_configs_from_yaml(app.config_path).mask_propagation)
        app.set_sam.assert_not_called()
        self.assertEqual(app.img_fnames, [])

    def test_invalid_settings_or_failed_save_keep_window_open_without_starting_load(self):
        for failure in ('invalid', 'write_error'):
            with self.subTest(failure=failure):
                app = self.make_app()
                dialog = self.show_welcome(app)
                before = app.config_path.read_bytes()
                if failure == 'invalid':
                    dialog.classes_editor.setPlainText('not a mapping')
                else:
                    dialog.controls['mask_propagation'].setChecked(False)
                app.set_sam = Mock()
                with patch('aramsam_annotator.app.QFileDialog.getOpenFileName') as chooser, \
                        patch('aramsam_annotator.app.save_configs_to_yaml', side_effect=OSError('read only')), \
                        patch('aramsam_annotator.workers.traceback.print_exc'):
                    dialog.image_button.click()
                self.assertTrue(dialog.isVisible())
                self.assertTrue(dialog.error_label.isVisible())
                self.assertTrue(dialog.error_label.text())
                self.assertEqual(app.config_path.read_bytes(), before)
                chooser.assert_not_called()
                app.set_sam.assert_not_called()

    def test_quit_discards_edits_and_shuts_down_without_confirmation(self):
        app = self.make_app()
        dialog = self.show_welcome(app)
        before = app.config_path.read_bytes()
        dialog.controls['mask_propagation'].setChecked(False)
        with patch.object(app.application, 'quit') as quit_app, patch.object(app, 'shutdown') as shutdown:
            dialog.quit_button.click()
        quit_app.assert_called_once()
        shutdown.assert_called_once()
        self.assertTrue(app.ui.isHidden())
        self.assertEqual(app.config_path.read_bytes(), before)

    def test_sam_disabled_from_welcome_loads_image_without_model_initialization(self):
        app = self.make_app()
        dialog = self.show_welcome(app)
        dialog.controls['sam_configs.do_sam'].setChecked(False)
        with patch('aramsam_annotator.app.QFileDialog.getOpenFileName', return_value=(str(self.image), '')), \
                patch('aramsam_annotator.annotator.Annotator.set_sam_version') as load_model:
            dialog.image_button.click()
        app.wait_for_workers()
        load_model.assert_not_called()
        self.assertEqual(app.annotator.annotation.img_name, 'image.png')
        self.assertFalse(app.ui.manual_annotation_button.isEnabled())
        self.assertTrue(app.ui.draw_button.isEnabled())


if __name__ == '__main__':
    unittest.main()
