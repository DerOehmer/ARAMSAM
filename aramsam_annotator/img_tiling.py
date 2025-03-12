import cv2
from aramsam_annotator.configs import ImgTiles
import tempfile
import os


def split_image_into_tiles(img_path: str, config: ImgTiles) -> list[str]:
    if not config.do_tiling:
        return [img_path]

    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    stride = int(config.tile_size * (1 - config.tile_overlap))
    temp_dir = tempfile.mkdtemp(dir=os.getcwd())
    tile_paths = []

    for top in range(0, height, stride):
        for left in range(0, width, stride):
            bottom = min(top + config.tile_size, height)
            right = min(left + config.tile_size, width)

            # Adjust tiles to have consistent tile_size dimensions
            if bottom - top < config.tile_size:
                top = bottom - config.tile_size
            if right - left < config.tile_size:
                left = right - config.tile_size

            top, left = max(0, top), max(0, left)
            tile = img[top:bottom, left:right]

            tile_filename = os.path.join(temp_dir, f"tile_{top}_{left}.jpg")
            cv2.imwrite(tile_filename, tile)
            tile_paths.append(tile_filename)

    return tile_paths
