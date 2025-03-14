import cv2
from aramsam_annotator.configs import ImgTiles

import os


def split_image_into_tiles(img_path: str, temp_dir: str, config: ImgTiles) -> list[str]:
    if not config.do_tiling:
        return [img_path]

    img = cv2.imread(img_path)
    img_file_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_file_name)[0]
    height, width = img.shape[:2]

    tile_size = config.tile_size
    overlap = config.tile_overlap
    stride = int(tile_size * (1 - overlap))

    # Compute unique starting positions for vertical tiles.
    if height <= tile_size:
        top_positions = [0]
    else:
        top_positions = list(range(0, height - tile_size, stride))
        # Ensure the bottom tile covers the image edge.
        top_positions.append(height - tile_size)

    # Compute unique starting positions for horizontal tiles.
    if width <= tile_size:
        left_positions = [0]
    else:
        left_positions = list(range(0, width - tile_size, stride))
        left_positions.append(width - tile_size)

    tile_paths = []
    for top in top_positions:
        for left in left_positions:
            # Extract a tile of fixed size.
            tile = img[top : top + tile_size, left : left + tile_size]
            tile_filename = os.path.join(temp_dir, f"{img_name}_{top}_{left}.jpg")
            cv2.imwrite(tile_filename, tile)
            tile_paths.append(tile_filename)

    return tile_paths


"""def split_image_into_tiles(img_path: str, temp_dir: str, config: ImgTiles) -> list[str]:
    if not config.do_tiling:
        return [img_path]

    img = cv2.imread(img_path)
    img_file_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_file_name)[0]
    height, width = img.shape[:2]

    stride = int(config.tile_size * (1 - config.tile_overlap))
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

            tile_filename = os.path.join(temp_dir, f"{img_name}_{top}_{left}.jpg")
            cv2.imwrite(tile_filename, tile)
            tile_paths.append(tile_filename)

    return tile_paths"""
