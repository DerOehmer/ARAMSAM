from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

from aramsam_annotator.app import App
from aramsam_annotator.annotator import Annotator
from aramsam_annotator.backend.annotations import AnnotationObject, MaskData
from aramsam_annotator.backend.storage import AnnotationRepository
from aramsam_annotator.configs import (
    AramsamConfigs, SaveData, load_configs_from_yaml, save_configs_to_yaml,
)
from aramsam_annotator.gui import OptionsWindow
from aramsam_annotator.main import create_vis_options
from aramsam_annotator.mask_visualizations import MaskVisualization, draw_mask_instance
from aramsam_annotator.visualization_options import CONTROL_IMAGE_OPTIONS, default_layout
from aramsam_annotator.welcome import WelcomeWindow


class InstanceDrawingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((128, 128, 3), 100, np.uint8)
        self.mask = np.zeros((128, 128), np.uint8)
        self.mask[32:96, 32:96] = 255
        self.color = (0, 100, 200)

    def test_contour_circle_alpha_and_id_match_the_supplied_style(self):
        with patch('aramsam_annotator.mask_visualizations.cv2.putText', wraps=cv2.putText) as text:
            result = draw_mask_instance(self.image, self.mask, 123, self.color)
        self.assertIs(result, self.image)
        np.testing.assert_array_equal(result[32, 32], self.color)
        np.testing.assert_array_equal(result[53, 63], (50, 100, 150))
        np.testing.assert_array_equal(result[40, 40], (100, 100, 100))
        np.testing.assert_array_equal(result[0, 0], (100, 100, 100))
        self.assertEqual(text.call_args.args[1], '123')
        self.assertEqual(text.call_args.args[4], 0.45)
        np.testing.assert_array_equal(text.call_args.args[5], (255, 255, 255))

    def test_centroid_falls_back_inside_a_mask_with_a_hole(self):
        mask = np.zeros((128, 128), np.uint8)
        cv2.circle(mask, (64, 64), 40, 255, -1)
        cv2.circle(mask, (64, 64), 25, 0, -1)
        self.assertEqual(mask[64, 64], 0)
        with patch('aramsam_annotator.mask_visualizations.cv2.putText', wraps=cv2.putText) as text:
            draw_mask_instance(self.image, mask, 7, self.color)
        args = text.call_args.args
        (width, height), _ = cv2.getTextSize(args[1], args[3], args[4], args[6])
        cx, cy = args[2][0] + width // 2, args[2][1] - height // 2
        self.assertNotEqual(mask[cy, cx], 0)

    def test_empty_mask_does_nothing_and_corner_masks_are_clipped_safely(self):
        original = self.image.copy()
        draw_mask_instance(self.image, np.zeros_like(self.mask), 0, self.color)
        np.testing.assert_array_equal(self.image, original)
        for y, x in ((0, 0), (127, 127)):
            with self.subTest(corner=(y, x)):
                mask = np.zeros_like(self.mask)
                mask[y, x] = 1
                result = draw_mask_instance(original.copy(), mask, 999, self.color)
                self.assertEqual(result.shape, original.shape)
                self.assertEqual(result.dtype, np.uint8)
                self.assertFalse(np.array_equal(result, original))

    def test_cached_instance_view_matches_fresh_drawing_after_add_remove_and_replace(self):
        with patch.object(AnnotationObject, '_load_img', return_value=self.image):
            annotation = AnnotationObject('image.png')
        visualizer = annotation.mask_visualizer
        first = MaskData(mid=1, origin='Polygon_drawing', mask=self.mask.copy(), color_idx=1)
        second = MaskData(mid=2, origin='Polygon_drawing', mask=np.roll(self.mask, 20, axis=1), color_idx=2)
        for objects in ([first], [first, second], [second], [self.mask_object(2, 0)]):
            annotation.good_masks = list(objects)
            visualizer.set_annotation(annotation=annotation)
            actual = visualizer.get_instance_contours().copy()
            fresh = MaskVisualization()
            fresh.set_annotation(img=self.image, mask_objs=objects)
            np.testing.assert_array_equal(actual, fresh.get_instance_contours())
            visualizer.set_annotation(annotation=annotation)
            np.testing.assert_array_equal(actual, visualizer.get_instance_contours())
        np.testing.assert_array_equal(self.image, np.full_like(self.image, 100))

    def mask_object(self, mid, offset):
        mask = np.zeros_like(self.mask)
        mask[2 + offset:10 + offset, 2:10] = 255
        return MaskData(mid=mid, origin='Polygon_drawing', mask=mask, color_idx=mid)

    def test_new_view_is_populated_by_the_annotation_renderer(self):
        with patch.object(AnnotationObject, '_load_img', return_value=self.image):
            annotator = Annotator(AramsamConfigs())
            annotator.create_new_annotation('image.png')
        annotation = annotator.annotation
        annotation.good_masks = [self.mask_object(1, 0)]
        annotator.update_collections(annotation)
        self.assertEqual(annotation.mask_visualizations.instance_contours.shape, self.image.shape)
        self.assertFalse(np.array_equal(annotation.mask_visualizations.instance_contours, self.image))


class VisualizationConfigTests(unittest.TestCase):
    def test_default_and_custom_layouts_and_control_selection_round_trip(self):
        for bbox_mode in (False, True):
            settings = SaveData(save_masks=not bbox_mode, save_bboxes=bbox_mode)
            configs = AramsamConfigs(save_data=settings)
            options, defaults, current = create_vis_options(configs)
            self.assertEqual(defaults, default_layout(bbox_mode))
            self.assertIn('instance_contours', options)
            self.assertNotIn('bbox', options)  # Coordinates are not an image view.
            self.assertEqual(current, defaults)
            self.assertIsNot(current, defaults)
        custom = ['instance_contours', 'img', 'mask_collection', 'bbox_img']
        configs = AramsamConfigs(layout=custom,
                                 save_data=SaveData(control_image_visualization='instance_contours'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'configs.yaml'
            save_configs_to_yaml(configs, path)
            loaded = load_configs_from_yaml(path)
        self.assertEqual(asdict(loaded), asdict(configs))
        self.assertEqual(create_vis_options(loaded)[1], custom)

    def test_unknown_visualizations_and_malformed_layouts_are_rejected(self):
        layouts = (['img'] * 3, 'img', ['img', 'img', 'img', 'unknown'],
                   ['img', 'img', 'img', None], {'top_left': 'img'})
        for layout in layouts:
            with self.subTest(layout=layout), self.assertRaisesRegex(ValueError, 'Layout'):
                AramsamConfigs(layout=layout)
        for selection in ('unknown', None, []):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, 'control image'):
                SaveData(control_image_visualization=selection)


class VisualizationExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / 'image.png'
        cv2.imwrite(str(self.image), np.full((128, 128, 3), 100, np.uint8))
        self.annotation = AnnotationObject(self.image)
        pixels = np.zeros((128, 128), np.uint8)
        pixels[32:96, 32:96] = 255
        self.source = MaskData(mid=12, origin='Polygon_drawing', mask=pixels, color_idx=2, class_id=4)
        self.annotation.add_masks([self.source], decision=True)
        self.annotation.good_masks = [self.source]

    def test_control_images_use_the_selected_view_without_changing_labels(self):
        labels = None
        for selection in CONTROL_IMAGE_OPTIONS:
            with self.subTest(selection=selection):
                repository = AnnotationRepository(SaveData(control_image_visualization=selection))
                folder = self.root / selection
                expected = repository.control_image(self.annotation)
                repository.save(self.annotation, folder)
                actual = (folder / 'control_images' / 'image_control.jpg').read_bytes()
                encoded, expected_jpeg = cv2.imencode('.jpg', expected)
                self.assertTrue(encoded)
                self.assertEqual(actual, expected_jpeg.tobytes())
                contents = (folder / 'labels' / 'image.txt').read_text()
                if labels is None:
                    labels = contents
                self.assertEqual(contents, labels)

    def test_export_ignores_cached_images_rejected_masks_and_interactive_previews(self):
        repository = AnnotationRepository(SaveData(control_image_visualization='instance_contours'))
        self.annotation.mask_visualizations.instance_contours = np.zeros_like(self.annotation.img)
        self.annotation.preview_mask = np.full_like(self.source.mask, 255)
        rejected = MaskData(mid=99, origin='Polygon_drawing', mask=self.annotation.preview_mask)
        self.annotation.add_masks([rejected])
        original = self.annotation.img.copy()
        color = tuple(int(channel) for channel in MaskVisualization().get_color(2)[::-1])
        expected = draw_mask_instance(original.copy(), self.source.mask, self.source.mid, color)
        np.testing.assert_array_equal(repository.control_image(self.annotation), expected)
        np.testing.assert_array_equal(self.annotation.img, original)

    def test_automatic_export_uses_masks_or_boxes_and_empty_annotations_are_valid(self):
        repository = AnnotationRepository(SaveData())
        view = MaskVisualization()
        view.set_annotation(img=self.annotation.img, mask_objs=[self.source])
        np.testing.assert_array_equal(repository.control_image(self.annotation), view.get_masked_img())
        view = MaskVisualization()
        view.set_annotation(img=self.annotation.img, mask_objs=[self.source])
        np.testing.assert_array_equal(repository.control_image(self.annotation, segmentation=False),
                                      view.get_bbox_img())
        self.annotation.good_masks = []
        repository = AnnotationRepository(SaveData(control_image_visualization='instance_contours'))
        np.testing.assert_array_equal(repository.control_image(self.annotation), self.annotation.img)

    def test_native_export_also_uses_the_selected_view(self):
        repository = AnnotationRepository(SaveData(mask_style='default',
                                                   control_image_visualization='instance_contours'))
        expected = repository.control_image(self.annotation)
        repository.save(self.annotation, self.root / 'native')
        _, encoded = cv2.imencode('.jpg', expected)
        self.assertEqual((self.root / 'native' / 'image_annots' / 'annotations.jpg').read_bytes(),
                         encoded.tobytes())


class VisualizationSettingsUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'configs.yaml'

    def make_app(self, configs):
        save_configs_to_yaml(configs, self.path)
        with patch('aramsam_annotator.gui.get_monitors', return_value=[]):
            app = App(configs=load_configs_from_yaml(self.path), config_path=self.path)
        self.addCleanup(app.ui.deleteLater)
        self.addCleanup(app.ui.hide)
        self.addCleanup(app.shutdown)
        return app

    def test_welcome_saves_custom_layout_and_export_style_and_applies_them(self):
        app = self.make_app(AramsamConfigs(save_data=SaveData(save_dir=None)))
        app.show_welcome()
        dialog = app.welcome_window
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(dialog.hide)
        selector = dialog.controls['layout']
        self.assertTrue(selector.use_default.isChecked())
        selector.use_default.setChecked(False)
        custom = ['instance_contours', 'img', 'mask_collection', 'bbox_img']
        for control, selected in zip(selector.comboboxes, custom):
            control.setCurrentIndex(control.findData(selected))
        export = dialog.controls['save_data.control_image_visualization']
        export.setCurrentIndex(export.findData('instance_contours'))
        app.navigation.load_image_path = Mock()
        with patch('aramsam_annotator.app.QFileDialog.getOpenFileName', return_value=('image.png', '')):
            dialog.image_button.click()
        self.assertEqual(load_configs_from_yaml(self.path).layout, custom)
        self.assertEqual(load_configs_from_yaml(self.path).save_data.control_image_visualization,
                         'instance_contours')
        self.assertEqual(app.fields, custom)
        self.assertEqual(app.ui.ui_options['layout_settings_options']['current'], custom)
        self.assertEqual(app.configs.save_data.control_image_visualization, 'instance_contours')

    def test_layout_settings_include_the_new_view_and_update_the_session(self):
        custom = ['instance_contours', 'img', 'mask_collection', 'bbox_img']
        app = self.make_app(AramsamConfigs(layout=custom))
        self.assertEqual(app.fields, custom)
        dialog = OptionsWindow(app.ui.ui_options['layout_settings_options'])
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(dialog.hide)
        for control, selected in zip(dialog.comboboxes, custom):
            self.assertEqual(control.currentData(), selected)
            self.assertGreaterEqual(control.findData('instance_contours'), 0)
        dialog.layout_settings_options_signal.connect(app.ui._recv_layout_options_changes)
        dialog.comboboxes[1].setCurrentIndex(dialog.comboboxes[1].findData('instance_contours'))
        dialog.ok_button.click()
        self.assertEqual(app.fields, ['instance_contours', 'instance_contours', 'mask_collection', 'bbox_img'])
        self.assertEqual(app.configs.layout, app.fields)

    def test_default_layout_adapts_to_mask_or_bbox_mode_without_persisting_a_custom_layout(self):
        dialog = WelcomeWindow(AramsamConfigs(), self.path)
        self.addCleanup(dialog.deleteLater)
        dialog.controls['save_data.save_bboxes'].setChecked(True)
        self.assertEqual(dialog.controls['layout'].comboboxes[2].currentData(), 'bbox_img_cnt')
        self.assertIsNone(dialog.read_configs().layout)
        dialog.controls['save_data.save_masks'].setChecked(True)
        self.assertEqual(dialog.controls['layout'].comboboxes[2].currentData(), 'masked_img_cnt')
        self.assertIsNone(dialog.read_configs().layout)


if __name__ == '__main__':
    unittest.main()
