import cv2
from aramsam_annotator.configs import ImgTiles

import os


def split_image_into_tiles(img_path: str, temp_dir: str, config: ImgTiles) -> list[str]:
    if not config.do_tiling:
        return [img_path]

    if config.tile_size <= 0 or not 0 <= config.tile_overlap < 1:
        raise ValueError("Tile size must be positive and overlap must be in [0, 1)")

    img = cv2.imread(str(img_path))
    if img is None:
        raise OSError(f"Could not read image: {img_path}")

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    height, width = img.shape[:2]

    tile_size = min(config.tile_size, 1024)
    stride = max(1, int(tile_size * (1 - config.tile_overlap)))

    def get_positions(length: int) -> list[int]:
        if length <= tile_size:
            return [0]

        if config.allow_smaller_border_tiles:
            return list(range(0, length - tile_size + stride, stride))

        # Original behavior: shift last tile to image boundary
        positions = list(range(0, length - tile_size, stride))
        positions.append(length - tile_size)
        return positions

    top_positions = get_positions(height)
    left_positions = get_positions(width)

    tile_paths = []
    for top in top_positions:
        for left in left_positions:
            # Border tiles may be smaller than tile_size
            tile = img[top : top + tile_size, left : left + tile_size]

            tile_filename = os.path.join(temp_dir, f"{img_name}_{top}_{left}.jpg")
            if not cv2.imwrite(tile_filename, tile):
                raise OSError(f"Could not write tile: {tile_filename}")

            tile_paths.append(tile_filename)

    return tile_paths