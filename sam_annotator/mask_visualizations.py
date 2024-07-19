import numpy as np
import cv2
from scipy.spatial import KDTree
from dataclasses import dataclass


@dataclass
class MaskData:
    mask: np.ndarray
    origin: str
    center: tuple = None
    color_idx: int = None
    contour: np.ndarray = None


@dataclass
class MaskVisualizationData:  # TODO add img and mask, mask collection white machen, an namen kommen
    img: np.ndarray = None
    mask: np.ndarray = None
    maskinrgb: np.ndarray = None
    masked_img: np.ndarray = None
    mask_collection: np.ndarray = None
    masked_img_cnt: np.ndarray = None
    mask_collection_cnt: np.ndarray = None


class MaskVisualization:
    def __init__(
        self,
        img: np.ndarray,
        mask_objs: list[MaskData],
        cnt_color: tuple[int] = (0, 255, 0),
    ):
        self.img = img
        self.mask_objs = mask_objs
        self.cnt_color = cnt_color

        self.masked_img: np.ndarray = None
        self.maskinrgb: np.ndarray = None
        self.mask_collection: np.ndarray = None
        self.masked_img_cnt: np.ndarray = None
        self.mask_collection_cnt: np.ndarray = None

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

    def get_masked_img(self) -> np.ndarray:
        self._set_mask_centers()
        self._assign_mask_colors()
        self.masked_img = self.img.copy()

        for m in self.mask_objs:
            cnts, _ = cv2.findContours(m.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            m.contour = cnts
            r, g, b = self.colormap[m.color_idx]
            self.masked_img = cv2.drawContours(
                self.masked_img,
                cnts,
                -1,
                (int(b), int(g), int(r)),
                -1,
                lineType=cv2.LINE_8,
            )

        return self.masked_img

    def get_masked_img_cnt(self, cnt: np.ndarray) -> np.ndarray:
        if self.masked_img is None:
            self.get_masked_img()
        self.masked_img_cnt = self.masked_img.copy()
        self.masked_img_cnt = cv2.drawContours(
            self.masked_img_cnt,
            cnt,
            -1,
            self.cnt_color,
            thickness=2,
            lineType=cv2.LINE_8,
        )
        return self.masked_img_cnt

    def get_mask_collection(self) -> np.ndarray:

        y, x, _ = self.img.shape
        self.mask_collection = np.zeros((y, x, 3), dtype=np.uint8)
        mask_coll_bin = np.zeros((y, x), dtype=np.uint8)

        for m in self.mask_objs:
            m = m.mask
            overlap = cv2.bitwise_and(m, mask_coll_bin)
            mask_coll_bin = np.where(m == 255, 255, mask_coll_bin)
            self.mask_collection[np.where(m == 255)] = [255, 255, 255]
            self.mask_collection[np.where(overlap == 255)] = [0, 0, 255]

        return self.mask_collection

    def get_mask_collection_cnt(self, cnt: np.ndarray) -> np.ndarray:
        if self.mask_collection is None:
            self.get_mask_collection()
        self.mask_collection_cnt = self.mask_collection.copy()
        self.mask_collection_cnt = cv2.drawContours(
            self.mask_collection_cnt,
            cnt,
            -1,
            self.cnt_color,
            thickness=2,
            lineType=cv2.LINE_8,
        )

        return self.mask_collection_cnt

    def get_maskinrgb(self, mask_obj: MaskData) -> np.ndarray:
        img = self.img.copy()
        mask = mask_obj.mask
        self.maskinrgb = cv2.bitwise_and(img, img, mask=mask)

        return self.maskinrgb

    def _assign_mask_colors(self):
        masks = self.mask_objs
        coli = 0
        for i, m in enumerate(masks):
            if m.color_idx is None:
                k = len(masks) - 1 if len(masks) - 1 < 7 else 7
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

    def _find_nearest_neighbors(self, points: np.ndarray, k: int = 5) -> np.ndarray:
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
                yindcs, xindcs = np.where(m.mask == 255)
                xmin, xmax = np.amin(xindcs), np.amax(xindcs)
                ymin, ymax = np.amin(yindcs), np.amax(yindcs)
                cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
                m.center = (cx, cy)
