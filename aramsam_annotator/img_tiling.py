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
    img_file_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_file_name)[0]
    height, width = img.shape[:2]

    tile_size = min(config.tile_size, 1024)
    overlap = config.tile_overlap
    stride = max(1, int(tile_size * (1 - overlap)))

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
            if not cv2.imwrite(tile_filename, tile):
                raise OSError(f"Could not write tile: {tile_filename}")
            tile_paths.append(tile_filename)

    return tile_paths
