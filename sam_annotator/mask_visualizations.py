import numpy as np
import cv2
from scipy.spatial import KDTree
from dataclasses import dataclass
from os.path import basename
from pathlib import Path
from typing import Tuple


class MaskIdHandler:
    def __init__(self):
        self._id = 0
        self.ids: list[int] = []

    def set_id(self):
        current_id = self._id
        self.ids.append(current_id)
        self._id += 1
        return current_id


@dataclass
class MaskData:
    mid: int
    mask: np.ndarray
    origin: str
    time_stamp: int = None  # in deciseconds (1/10th of a second)
    center: tuple = None
    color_idx: int = None
    contour: np.ndarray = None


@dataclass
class MaskVisualizationData:
    img: np.ndarray = None
    img_sam_preview: np.ndarray = None
    mask: np.ndarray = None
    maskinrgb: np.ndarray = None
    masked_img: np.ndarray = None
    mask_collection: np.ndarray = None
    masked_img_cnt: np.ndarray = None
    mask_collection_cnt: np.ndarray = None


class AnnotationObject:
    def __init__(self, filepath: Path) -> None:
        self.filepath: str = str(filepath)
        self.img: np.ndarray = cv2.imread(self.filepath)
        self.img_name = basename(filepath)
        if self.img.shape[2] == 4:
            print("Loaded image with 4 channels - ignoring last")
            self.img = self.img[:, :, :3]
        self.masks: list[MaskData] = []
        self.good_masks: list[MaskData] = []
        self.mask_decisions: list[bool] = []

        self.features = None
        self.original_size = None
        self.input_size = None

        self.mask_visualizer = MaskVisualization()
        self.mask_visualizations: MaskVisualizationData = MaskVisualizationData(
            img=self.img
        )
        self.preview_mask = None

    """def set_masks(self, mask_objects: list[MaskData]):
        self.masks = mask_objects
        self.mask_decisions = [False for _ in range(len(self.masks))]"""

    def set_current_mask(self, mask_idx: int):
        self.mask_visualizations.mask = cv2.cvtColor(
            self.masks[mask_idx].mask, cv2.COLOR_GRAY2BGR
        )

    def add_masks(self, masks, decision=False):
        self.masks.extend(masks)
        self.mask_decisions.extend([decision for _ in range(len(masks))])

    def set_sam_parameters(self, features, original_size, input_size):
        self.features = features
        self.original_size = original_size
        self.input_size = input_size

    def get_sam_parameters(self):
        return self.features, self.original_size, self.input_size

    def load_masks_from_dir(self, masks_dir: Path, mid_handler: MaskIdHandler):
        mask_files = sorted(masks_dir.glob("*.png"))
        for mask_file in mask_files:
            if not mask_file.is_file():
                raise FileNotFoundError(f"File {mask_file} not found")
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            mask_data = MaskData(
                mid=mid_handler.set_id(),
                mask=mask,
                origin="",
            )
            self.good_masks.append(mask_data)


class MaskVisualization:
    def __init__(
        self,
        cnt_color: tuple[int] = (0, 255, 0),
    ):
        self.img = None
        self.mask_objs = None
        self.preview_mask = None

        self.mask_ids = []
        self.mask_ids_to_remove = []
        self.mask_ids_to_add = []

        self.cnt_color = cnt_color
        self.masked_img: np.ndarray = None
        self.maskinrgb: np.ndarray = None
        self.mask_collection: np.ndarray = None
        self._mask_coll_bin: np.ndarray = None
        self.masked_img_cnt: np.ndarray = None
        self.mask_collection_cnt: np.ndarray = None
        self.img_man_preview: np.ndarray = None

        self.no_collection_to_update = False

    @property
    def colormap(self) -> np.ndarray:
        # https://vega.github.io/vega/docs/schemes/ # category10
        return np.array(
            [
                [31, 119, 180],
                [255, 127, 14],
                [44, 160, 44],
                [214, 39, 40],
                [148, 103, 189],
                [140, 86, 75],
                [227, 119, 194],
                [127, 127, 127],
                [188, 189, 34],
                [23, 190, 207],
            ],
            dtype=np.uint8,
        )

    def set_annotation(
        self,
        annotation: AnnotationObject = None,
        img: np.ndarray = None,
        mask_objs: list[MaskData] = None,
    ):
        if annotation is not None:
            self.img = annotation.img
            self.mask_objs = annotation.good_masks
            self.preview_mask = annotation.preview_mask

            new_ids = None
            if self.mask_ids:
                new_ids = self._compare_mask_ids()
            self._set_mask_ids(new_ids)

        elif img is not None and mask_objs is not None:
            self.img = img
            self.mask_objs = mask_objs
            self.preview_mask = None
        else:
            raise ValueError("Either annotation or img and mask_objs must be provided")

    def _set_mask_ids(self, mask_ids: list[int] | None):
        if mask_ids is not None:
            self.mask_ids = mask_ids
        else:
            self.mask_ids = [m.mid for m in self.mask_objs]

    def _compare_mask_ids(self):
        new_mask_ids = [m.mid for m in self.mask_objs]

        old_set = set(self.mask_ids)
        new_set = set(new_mask_ids)

        mids_to_remove = old_set - new_set
        mids_to_add = new_set - old_set
        self.mask_ids_to_remove = list(mids_to_remove)
        self.mask_ids_to_add = list(mids_to_add)
        if new_mask_ids == self.mask_ids:
            self.no_collection_to_update = True
        else:
            self.no_collection_to_update = False

        return new_mask_ids

    def set_contour(self, mask_obj: MaskData) -> np.ndarray:
        cnts, _ = cv2.findContours(
            mask_obj.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        mask_obj.contour = cnts
        return cnts

    def get_masked_img(self) -> np.ndarray:
        if self.no_collection_to_update:
            return self.masked_img
        if self.masked_img is None or self.mask_ids_to_remove:
            self.masked_img = self.img.copy()

        self._set_mask_centers()
        self._assign_mask_colors()

        for m in self.mask_objs:
            if (
                self.mask_ids_to_add
                and not self.mask_ids_to_remove
                and m.mid not in self.mask_ids_to_add
            ):
                continue
            cnts = self.set_contour(m)
            r, g, b = self.colormap[m.color_idx]
            self.masked_img = cv2.drawContours(
                self.masked_img,
                cnts,
                -1,
                (int(b), int(g), int(r)),
                -1,
                lineType=cv2.LINE_8,
            )
            cv2.putText(
                self.masked_img,
                str(m.mid),
                m.center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return self.masked_img

    def get_masked_img_cnt(self, cnt: np.ndarray) -> np.ndarray:
        if self.masked_img is None:
            self.get_masked_img()
        self.masked_img_cnt = self.masked_img.copy()
        if cnt is not None:
            self.masked_img_cnt = cv2.drawContours(
                self.masked_img_cnt,
                cnt,
                -1,
                self.cnt_color,
                thickness=2,
                lineType=cv2.LINE_8,
            )
        if self.preview_mask is not None:
            self.masked_img_cnt[self.preview_mask == 255] = 255, 255, 255
        return self.masked_img_cnt

    def get_mask_collection(self) -> np.ndarray:
        if self.no_collection_to_update:
            return self.mask_collection

        y, x, _ = self.img.shape
        if self.mask_collection is None or self.mask_ids_to_remove:
            self.mask_collection = np.zeros((y, x, 3), dtype=np.uint8)
        if self._mask_coll_bin is None or self.mask_ids_to_remove:
            self._mask_coll_bin = np.zeros((y, x), dtype=np.uint8)

        for m in self.mask_objs:
            if (
                self.mask_ids_to_add
                and not self.mask_ids_to_remove
                and m.mid not in self.mask_ids_to_add
            ):
                continue
            m = m.mask
            overlap = cv2.bitwise_and(m, self._mask_coll_bin)
            self._mask_coll_bin = np.where(m == 255, 255, self._mask_coll_bin)
            self.mask_collection[np.where(m == 255)] = [255, 255, 255]
            self.mask_collection[np.where(overlap == 255)] = [0, 0, 255]

        return self.mask_collection

    def get_mask_collection_cnt(self, cnt: np.ndarray) -> np.ndarray:
        if self.mask_collection is None:
            self.get_mask_collection()
        self.mask_collection_cnt = self.mask_collection.copy()
        if cnt is not None:
            self.mask_collection_cnt = cv2.drawContours(
                self.mask_collection_cnt,
                cnt,
                -1,
                self.cnt_color,
                thickness=2,
                lineType=cv2.LINE_8,
            )
        if self.preview_mask is not None:
            self.mask_collection_cnt[self.preview_mask == 255] = (
                224,
                167,
                61,
            )  # 191, 196, 45

        return self.mask_collection_cnt

    def get_maskinrgb(self, mask_obj: MaskData) -> np.ndarray:
        img = self.img.copy()
        mask = mask_obj.mask
        self.maskinrgb = cv2.bitwise_and(img, img, mask=mask)

        return self.maskinrgb

    def _assign_mask_colors(self, k: int = 8):
        masks = self.mask_objs
        coli = 0
        for i, m in enumerate(masks):
            if m.color_idx is None:
                k = len(masks) - 1 if len(masks) - 1 < k else k
                if k:
                    nnbs = self._find_nearest_neighbors(
                        np.array([mask.center for mask in masks]), k
                    )
                    neighbours = nnbs[i]
                    nb_cols = [
                        masks[nb].color_idx
                        for nb in neighbours
                        if masks[nb].color_idx is not None
                    ]
                    n_unique_nb_cols = (
                        len(np.unique(nb_cols)) if len(nb_cols) > 1 else len(nb_cols)
                    )
                    while coli in nb_cols:
                        coli += 1
                        if n_unique_nb_cols >= len(self.colormap):
                            raise ValueError("Not enough colors in the colormap")
                        if coli >= len(self.colormap):
                            coli = 0

                m.color_idx = coli

    def _find_nearest_neighbors(self, points: np.ndarray, k: int) -> np.ndarray:
        """
        Find the k nearest neighbors for each point in a 2D space.

        Parameters:
        - points: np.ndarray, an array of shape (n_points, 2) representing the 2D coordinates of the points
        - k: int, the number of nearest neighbors to find for each point (default is 5)

        Returns:
        - neighbors_indices: np.ndarray, an array of shape (n_points, k) containing the indices of the k nearest neighbors for each point
        """
        # Create a KDTree from the points
        tree = KDTree(points)

        # Query the KDTree to find the k nearest neighbors for each point
        _, indices = tree.query(
            points, k=k + 1
        )  # k+1 because the first neighbor is the point itself

        # Exclude the first neighbor (the point itself)
        return indices[:, 1:]

    def _set_mask_centers(self):
        for m in self.mask_objs:
            if m.center is None:
                try:
                    yindcs, xindcs = np.where(m.mask == 255)
                except ValueError:
                    print("error")
                xmin, xmax = np.amin(xindcs), np.amax(xindcs)
                ymin, ymax = np.amin(yindcs), np.amax(yindcs)
                cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
                m.center = (cx, cy)

    def _weighted_mask(self, alpha=1):
        if self.preview_mask is None:
            return self.img
        else:
            red_img = np.zeros(self.img.shape, dtype="uint8")
            red_img[:, :] = (0, 255, 0)
            red_mask = cv2.bitwise_and(red_img, red_img, mask=self.preview_mask)
            return cv2.addWeighted(red_mask, alpha, self.img, 1, 0)

    def get_sam_preview(
        self, manual_mask_points: list[Tuple[int]], manual_mask_point_labels: list[int]
    ) -> np.ndarray:

        self.img_man_preview = self._weighted_mask(alpha=1)
        for p, l in zip(manual_mask_points, manual_mask_point_labels):
            self.img_man_preview = cv2.circle(
                self.img_man_preview,
                center=p,
                radius=2,
                color=(255 * l, 255 * l, 255),
                thickness=-1,
            )
        return self.img_man_preview

    def get_polygon_preview(self, manual_mask_points: list[Tuple[int]]) -> np.ndarray:
        self.img_man_preview = self._weighted_mask(alpha=0.1)
        for p in manual_mask_points:
            self.img_man_preview = cv2.circle(
                self.img_man_preview,
                center=p,
                radius=1,
                color=(255, 255, 255),
                thickness=-1,
            )
        return self.img_man_preview

    def get_mask_deletion_preview(self) -> np.ndarray:
        if self.preview_mask is None:
            return None
        img = self.img.copy()
        extended_mask = cv2.dilate(
            self.preview_mask, np.ones((5, 5), np.uint8), iterations=5
        )
        result_img = cv2.bitwise_and(img, img, mask=extended_mask)
        cnts = cv2.findContours(
            self.preview_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        for cnt in cnts:
            cv2.drawContours(result_img, [cnt], -1, (0, 0, 255), 1)
        self.img_man_preview = result_img
        return self.img_man_preview
