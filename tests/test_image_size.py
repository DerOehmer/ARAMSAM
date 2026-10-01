import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from aramsam_annotator.backend.annotations import AnnotationObject
from aramsam_annotator.configs import ImgTiles
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

    def test_oversized_tiles_are_clamped_in_config_and_at_use(self):
        settings = ImgTiles(tile_size=2048, tile_overlap=0)
        self.assertEqual(settings.tile_size, 1024)
        settings.tile_size = 4096
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'image.png'
            cv2.imwrite(str(image), np.zeros((1200, 1800, 3), np.uint8))
            tiles = split_image_into_tiles(str(image), directory, settings)
            self.assertEqual(len(tiles), 4)
            for tile in tiles:
                self.assertEqual(cv2.imread(tile).shape[:2], (1024, 1024))
