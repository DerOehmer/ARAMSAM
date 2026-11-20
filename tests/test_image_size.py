import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from aramsam_annotator.backend.annotations import AnnotationObject, MaskData, MaskIdHandler
from aramsam_annotator.backend.storage import AnnotationRepository
from aramsam_annotator.configs import ImgTiles, SaveData
from aramsam_annotator.img_tiling import split_image_into_tiles


class ImageSizeTests(unittest.TestCase):
    def test_image_size_limit_and_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'image.png'
            for height, width, expected in [(1200, 2400, (512, 1024)),
                                             (2400, 1200, (1024, 512)),
                                             (1024, 1024, (1024, 1024)),
                                             (40, 80, (40, 80))]:
                with self.subTest(size=(height, width)):
                    cv2.imwrite(str(image), np.zeros((height, width, 3), np.uint8))
                    self.assertEqual(AnnotationObject(image).img.shape[:2], expected)
                    self.assertEqual(cv2.imread(str(image)).shape[:2], (height, width))

    def test_configured_large_tiles_keep_their_size_and_cover_smaller_edges(self):
        settings = ImgTiles(tile_size=2048, tile_overlap=0)
        self.assertEqual(settings.tile_size, 2048)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'image.png'
            cv2.imwrite(str(image), np.zeros((2300, 3000, 3), np.uint8))
            tiles = split_image_into_tiles(str(image), directory, settings)
            self.assertEqual(len(tiles), 4)
            self.assertCountEqual([cv2.imread(tile).shape[:2] for tile in tiles],
                                  [(2048, 2048), (2048, 952), (252, 2048), (252, 952)])
            settings.tile_size = 4096
            tiles = split_image_into_tiles(str(image), directory, settings)
            self.assertEqual(len(tiles), 1)
            self.assertEqual(cv2.imread(tiles[0]).shape[:2], (2300, 3000))

    def test_1024_pixel_tiles_allow_small_edges_or_shift_full_size_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'image.png'
            cv2.imwrite(str(image), np.zeros((1200, 1800, 3), np.uint8))
            for smaller_edges in (True, False):
                with self.subTest(smaller_edges=smaller_edges):
                    settings = ImgTiles(tile_size=1024, tile_overlap=0,
                                        allow_smaller_border_tiles=smaller_edges)
                    tiles = split_image_into_tiles(str(image), directory, settings)
                    expected = ([(1024, 1024), (1024, 776), (176, 1024), (176, 776)]
                                if smaller_edges else [(1024, 1024)] * 4)
                    self.assertCountEqual([cv2.imread(tile).shape[:2] for tile in tiles], expected)

    def test_yolo_exports_original_image_bytes_and_keeps_controls_downsampled(self):
        for height, width in ((1200, 1800), (1800, 1200)):
            for segmentation in (True, False):
                with self.subTest(shape=(height, width), segmentation=segmentation), \
                        tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = root / 'source.png'
                    original = np.full((height, width, 3), (20, 70, 120), np.uint8)
                    cv2.imwrite(str(source), original)
                    annotation = AnnotationObject(source)
                    h, w = annotation.img.shape[:2]
                    mask = np.zeros((h, w), np.uint8)
                    mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = 255
                    obj = MaskData(mid=1, origin='Polygon_drawing', mask=mask if segmentation else None,
                                   bbox=(w // 4, h // 4, 3 * w // 4, 3 * h // 4), class_id=2)
                    annotation.good_masks = [obj]
                    output = root / 'output'
                    repository = AnnotationRepository(SaveData(save_masks=segmentation,
                                                               save_bboxes=not segmentation))
                    repository.save(annotation, output)
                    exported = output / 'images' / source.name
                    self.assertEqual(exported.read_bytes(), source.read_bytes())
                    self.assertEqual(cv2.imread(str(exported)).shape[:2], (height, width))
                    control = cv2.imread(str(output / 'control_images' / 'source_control.jpg'))
                    self.assertEqual(control.shape[:2], annotation.img.shape[:2])
                    self.assertEqual(max(control.shape[:2]), 1024)
                    values = np.asarray((output / 'labels' / 'source.txt').read_text().split()[1:], float)
                    if segmentation:
                        normalized = values.reshape(-1, 2)
                        np.testing.assert_allclose(normalized.min(axis=0), [w // 4 / w, h // 4 / h], atol=1e-6)
                    else:
                        np.testing.assert_allclose(values, [.5, .5, .5, .5], atol=2 / min(h, w))
                    restored = AnnotationObject(exported)
                    repository.load(restored, output, MaskIdHandler())
                    if segmentation:
                        np.testing.assert_array_equal(restored.good_masks[0].mask, mask)
                    else:
                        self.assertEqual(restored.good_masks[0].bbox, obj.bbox)

    def test_tile_exports_preserve_configured_and_border_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.png'
            cv2.imwrite(str(source), np.zeros((2300, 3000, 3), np.uint8))
            tiles_dir = root / 'tiles'
            tiles_dir.mkdir()
            tiles = split_image_into_tiles(str(source), str(tiles_dir), ImgTiles(tile_size=2048, tile_overlap=0))
            repository = AnnotationRepository(SaveData())
            output = root / 'output'
            for tile in tiles:
                annotation = AnnotationObject(tile)
                repository.save(annotation, output)
                exported = output / 'images' / Path(tile).name
                self.assertEqual(exported.read_bytes(), Path(tile).read_bytes())
                self.assertEqual(cv2.imread(str(exported)).shape[:2], cv2.imread(tile).shape[:2])
                self.assertLessEqual(max(annotation.img.shape[:2]), 1024)

    def test_jpeg_export_preserves_original_encoding_and_can_be_saved_again_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.jpg'
            cv2.imwrite(str(source), np.full((1100, 1600, 3), (30, 90, 160), np.uint8))
            repository = AnnotationRepository(SaveData())
            output = root / 'output'
            repository.save(AnnotationObject(source), output)
            exported = output / 'images' / source.name
            self.assertEqual(exported.read_bytes(), source.read_bytes())
            repository.save(AnnotationObject(exported), output)
            self.assertEqual(exported.read_bytes(), source.read_bytes())
            self.assertEqual(cv2.imread(str(exported)).shape[:2], (1100, 1600))

    def test_missing_original_fails_without_overwriting_existing_output_with_a_thumbnail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.png'
            cv2.imwrite(str(source), np.zeros((1100, 1600, 3), np.uint8))
            annotation = AnnotationObject(source)
            repository = AnnotationRepository(SaveData())
            output = root / 'output'
            repository.save(annotation, output)
            original_image = (output / 'images' / source.name).read_bytes()
            labels = (output / 'labels' / 'source.txt').read_bytes()
            metadata = (output / 'metadata' / 'source.json').read_bytes()
            source.unlink()
            with self.assertRaises(FileNotFoundError):
                repository.save(annotation, output)
            self.assertEqual((output / 'images' / source.name).read_bytes(), original_image)
            self.assertEqual((output / 'labels' / 'source.txt').read_bytes(), labels)
            self.assertEqual((output / 'metadata' / 'source.json').read_bytes(), metadata)
