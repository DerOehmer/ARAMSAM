"""Image identity, look-ahead selection, and atomic annotation transitions."""
from pathlib import Path

from aramsam_annotator.backend.annotations import AnnotationObject
from aramsam_annotator.img_tiling import split_image_into_tiles


def image_key(path):
    return str(Path(path).resolve())


class ImageQueue:
    """Adapt the public list queue, retaining its original last-in-first-out order."""
    def __init__(self, paths, temp_dir, tile_settings):
        self.paths = paths
        self.temp_dir = Path(temp_dir)
        self.tile_settings = tile_settings

    def _expand_last(self, paths):
        if not paths:
            return
        image = Path(paths[-1])
        if (self.tile_settings.do_tiling
                and not image.resolve().is_relative_to(self.temp_dir.resolve())):
            tiles = split_image_into_tiles(str(image), str(self.temp_dir), self.tile_settings)
            if not tiles:
                raise ValueError(f'No tiles produced for {image}')
            paths[-1:] = tiles

    def pop_pair(self):
        if not self.paths:
            raise IndexError('No images left in the queue')
        # Keep the live queue unchanged if reading/tiling either image fails.
        remaining = list(self.paths)
        self._expand_last(remaining)
        current = Path(remaining.pop())
        self._expand_last(remaining)
        following = Path(remaining[-1]) if remaining else None
        self.paths[:] = remaining
        return current, following


class AnnotationSession:
    """Select records by full path, never by basename or queue position alone."""
    def __init__(self, annotation_factory=AnnotationObject):
        self.annotation_factory = annotation_factory

    def prepare_pair(self, current, prefetched, filepath, next_filepath):
        available = [a for a in (current, prefetched) if a is not None]

        def select(path):
            if path is None:
                return None, False
            key = image_key(path)
            for annotation in available:
                if image_key(annotation.filepath) == key:
                    return annotation, annotation.features is None
            return self.annotation_factory(path), True

        annotation, embed_current = select(filepath)
        next_annotation, embed_next = select(next_filepath)
        return annotation, next_annotation, embed_current, embed_next
