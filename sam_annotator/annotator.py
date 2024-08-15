import numpy as np
import torch
import cv2
from pathlib import Path
from scipy.spatial import KDTree


from src.run_sam import SamInference
from sam_annotator.mask_visualizations import (
    MaskData,
    MaskVisualization,
    MaskVisualizationData,
    AnnotationObject,
)


class Annotator:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = SamInference(
            sam_checkpoint="sam_vit_b_01ec64.pth", model_type="vit_b", device=device
        )
        self.prev_annotation: AnnotationObject = None
        self.annotation: AnnotationObject = None
        self.next_annotation: AnnotationObject = None
        self.mask_idx = 0

        self.manual_annotation_enabled = False
        self.polygon_drawing_enabled = False
        self.manual_mask_points = []
        self.manual_mask_point_labels = []

    def toggle_manual_annotation(self):
        self.reset_manual_annotation()
        if not self.manual_annotation_enabled and not self.sam.predictor.is_image_set:
            print("Embed image before manually annotation")
            return
        self.manual_annotation_enabled = not self.manual_annotation_enabled
        if self.manual_annotation_enabled:
            self.polygon_drawing_enabled = False

    def toggle_polygon_drawing(self):
        self.reset_manual_annotation()
        self.polygon_drawing_enabled = not self.polygon_drawing_enabled
        if self.polygon_drawing_enabled:
            self.manual_annotation_enabled = False

    def reset_manual_annotation(self):
        self.annotation.preview_mask = None
        self.manual_mask_points = []
        self.manual_mask_point_labels = []

    def predict_sam_manually(self, position: tuple[int]):
        if self.manual_annotation_enabled:
            # create live mask preview
            self.annotation.preview_mask = (
                self.sam.predict(
                    pts=np.array(
                        [[position[0], position[1]], *self.manual_mask_points]
                    ),
                    pts_labels=np.array([1, *self.manual_mask_point_labels]),
                )
                * 255
            )
            self.update_collections(self.annotation)

    def mask_from_polygon(self):
        if self.polygon_drawing_enabled:
            self.annotation.preview_mask = np.zeros(
                self.annotation.img.shape[:2], dtype=np.uint8
            )
            if len(self.manual_mask_points) > 2:
                polypts = np.array(self.manual_mask_points, np.int32).reshape(
                    (-1, 1, 2)
                )
                cv2.fillPoly(self.annotation.preview_mask, [polypts], 255)
            self.update_collections(self.annotation)

    def update_mask_idx(self, new_idx: int = 0):
        if new_idx < 0:
            new_idx = 0
            print("Mask index cannot be negative. Setting to 0.")
        self.mask_idx = new_idx
        self.annotation.set_current_mask(self.mask_idx)

    def create_new_annotation(
        self, filepath: Path, next_filepath: Path = None
    ) -> tuple[bool]:
        embed_current = False
        embed_next = False
        if self.next_annotation is None:
            self.annotation = AnnotationObject(filepath=filepath)
            embed_current = True
        else:
            # TODO: check for bugs of shallow copies
            self.annotation = self.next_annotation

        if next_filepath is None:
            self.next_annotation = None
        else:
            self.next_annotation = AnnotationObject(filepath=next_filepath)
            embed_next = True

        return embed_current, embed_next

    def get_annotation_img_name(self):
        if self.annotation:
            return self.annotation.img_name
        else:
            return None

    def get_next_annotation_img_name(self):
        if self.next_annotation:
            return self.next_annotation.img_name
        else:
            return None

    def update_sam_features_to_current_annotation(self):
        self.sam.predictor.set_features(
            features=self.annotation.features,
            original_size=self.annotation.original_size,
            input_size=self.annotation.input_size,
        )

    def predict_with_sam(self):
        if self.annotation is None:
            raise ValueError("No annotation object found.")

        print(self.annotation.img.shape)
        self.sam.image_embedding(self.annotation.img)
        mask_objs, annotated_image = self.sam.custom_amg(roi_pts=False, n_points=100)
        self.annotation.mask_visualizations.masked_img = annotated_image
        self.annotation.set_masks(mask_objs)

        self.update_mask_idx()
        self.update_collections(self.annotation)
        self.preselect_mask()

    def good_mask(self):
        annot = self.annotation

        if self.manual_annotation_enabled:
            mask_to_store = MaskData(mask=annot.preview_mask, origin="sam_interactive")
            annot.masks.insert(self.mask_idx, mask_to_store)
            annot.mask_decisions.insert(self.mask_idx, True)
            self.reset_manual_annotation()
        elif self.polygon_drawing_enabled:
            mask_to_store = MaskData(mask=annot.preview_mask, origin="polygon")
            annot.masks.insert(self.mask_idx, mask_to_store)
            annot.mask_decisions.insert(self.mask_idx, True)
            self.reset_manual_annotation()
        else:
            mask_to_store = annot.masks[self.mask_idx]
            annot.mask_decisions[self.mask_idx] = True

        annot.good_masks.append(mask_to_store)
        self.mask_idx += 1

        self.update_collections(annot)
        if self.mask_idx >= len(annot.masks) - 1:
            next_mask_center = None  # all masks have been labeled
        else:
            self.annotation.set_current_mask(self.mask_idx)
            self.preselect_mask()
            next_mask_center = self.annotation.masks[self.mask_idx].center
        # TODO: instead of done return coordinates of center of next mask
        return next_mask_center

    def bad_mask(self):
        annot = self.annotation

        annot.mask_decisions[self.mask_idx] = False
        self.mask_idx += 1

        self.update_collections(annot)
        if self.mask_idx >= len(annot.masks) - 1:
            next_mask_center = None  # all masks have been labeled
        else:
            self.annotation.set_current_mask(self.mask_idx)
            self.preselect_mask()
            next_mask_center = self.annotation.masks[self.mask_idx].center
        # TODO: instead of done return coordinates of center of next mask
        return next_mask_center

    def preselect_mask(self, max_overlap_ratio: float = 0.4):
        annot = self.annotation
        mask_obj: MaskData = annot.masks[self.mask_idx]
        mask = mask_obj.mask
        maskorigin = mask_obj.origin

        mask_coll_bin = (
            np.any(
                annot.mask_visualizations.mask_collection != [0, 0, 0], axis=-1
            ).astype(np.uint8)
            * 255
        )
        mask_overlap = cv2.bitwise_and(mask_coll_bin, mask)
        mask_size = np.count_nonzero(mask)
        mask_overlap_size = np.count_nonzero(mask_overlap)
        overlap_ratio = mask_overlap_size / mask_size

        if overlap_ratio > max_overlap_ratio:
            self.bad_mask()

        if maskorigin == "sam_tracking" and len(annot.mask_decisions) == len(
            annot.good_masks
        ):
            self.good_mask()

    def step_back(self):
        annot = self.annotation

        if self.mask_idx > 0:

            if annot.mask_decisions[-1]:
                annot.good_masks.pop()

            annot.mask_decisions[self.mask_idx] = False
            self.mask_idx -= 1
            self.update_collections(annot)

    def update_collections(self, annot: AnnotationObject):

        mask_vis = MaskVisualization(annotation=annot)
        mvis_data: MaskVisualizationData = self.annotation.mask_visualizations
        masked_img = mask_vis.get_masked_img()
        mask_collection = mask_vis.get_mask_collection()
        if len(annot.masks) > self.mask_idx:
            mask_obj = annot.masks[self.mask_idx]
            cnt = mask_obj.contour
            maskinrgb = mask_vis.get_maskinrgb(mask_obj)
            masked_img_cnt = mask_vis.get_masked_img_cnt(cnt)
            mask_collection_cnt = mask_vis.get_mask_collection_cnt(cnt)
        else:
            maskinrgb = np.zeros_like(masked_img)
            masked_img_cnt = np.zeros_like(masked_img)
            mask_collection_cnt = np.zeros_like(masked_img)

        if self.manual_annotation_enabled:
            img_sam_preview = mask_vis.get_sam_preview(
                self.manual_mask_points, self.manual_mask_point_labels
            )
            mvis_data.img_sam_preview = img_sam_preview

        elif self.polygon_drawing_enabled:
            img_sam_preview = mask_vis.get_polygon_preview(self.manual_mask_points)
            mvis_data.img_sam_preview = img_sam_preview

        mvis_data.maskinrgb = maskinrgb
        mvis_data.masked_img = masked_img
        mvis_data.mask_collection = mask_collection
        mvis_data.masked_img_cnt = masked_img_cnt
        mvis_data.mask_collection_cnt = mask_collection_cnt

        self.annotation.good_masks = mask_vis.mask_objs


class PanoImageAligner:
    def __init__(self):
        self.images: list[np.ndarray] = []
        self.matching_masks: list[list[MaskData]] = []
        self.h_matrix: np.ndarray = None

    def add_image(self, img: np.ndarray, masks: list[MaskData]):
        imgbgr = img
        img_gray = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2GRAY)

        # Add the image to the list
        self.images.append(img_gray)
        self.matching_masks.append(masks)
        if len(self.images) > 2:
            # Keep only the two most recent images
            self.images.pop(0)
            self.matching_masks.pop(0)

    def match_and_align(self):
        img_gray1, img_gray2 = self.images

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img_gray1, None)
        kp2, des2 = orb.detectAndCompute(img_gray2, None)

        # Match descriptors.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort matches by distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw top matches.
        # img_matches = cv2.drawMatches(img_gray1, kp1, img_gray2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Estimate homography.
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        self.h_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp image using homography.
        height, width = img_gray2.shape

        tracked_masks = self.warp_masks(self.matching_masks[0], width, height)
        selected_masks, centerpts = self.select_masks_with_highest_iou(
            tracked_masks, self.matching_masks[1]
        )

        """annotimg = cv2.cvtColor(img_gray2.copy(), cv2.COLOR_GRAY2BGR)

        
        for m in self.masks[1]:
            mask = m.mask
            annotimg = self.get_mask_cnts(annotimg, mask, col=(10,10,10))
        for m in tracked_masks:
            annotimg = self.get_mask_cnts(annotimg, m, col=(100,100,100))
        for m in selected_masks:
            #mask = cv2.warpPerspective(mask, self.h_matrix, (width, height))
            annotimg = self.get_mask_cnts(annotimg, m, col=(255,0,0))
        for pt in centerpts:
            y,x = pt
            annotimg = cv2.circle(annotimg, (x,y), 2, (0, 255, 0), -1)

        # Save and display the results.
        ShowMe([annotimg], factor=1)"""
        return [MaskData(mask, "sam_tracking") for mask in selected_masks]

    def warp_masks(self, masks, w, h):
        return [
            cv2.warpPerspective(
                warp_m.mask, self.h_matrix, (w, h), flags=cv2.INTER_NEAREST
            )
            for warp_m in masks
        ]

    def get_mask_cnts(self, annnotimg, mask, thickness=2, col=None):
        cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if col is None:
            b, g, r = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )
        else:
            b, g, r = col
        annnotimg = cv2.drawContours(
            annnotimg, cnts, -1, (b, g, r), thickness, lineType=cv2.LINE_8
        )
        return annnotimg

    def compute_iou(self, mask1, mask2):
        # Compute intersection
        intersection = np.logical_and(mask1, mask2).sum()
        # Compute union
        union = np.logical_or(mask1, mask2).sum()
        # Compute IoU
        iou = intersection / union if union != 0 else 0
        return iou

    def compute_centroid(self, mask):
        indices = np.argwhere(mask)
        if len(indices) == 0:
            return None
        centroid = np.mean(indices, axis=0, dtype=np.int32)
        return centroid

    def select_masks_with_highest_iou(
        self, tracked_masks, new_masks, threshold=0.7, k=3
    ):
        # Compute centroids for all masks in tracked masks
        centroids_tr = []
        tracked_masks_roi = []
        for maskt in tracked_masks:
            centerpt = self.compute_centroid(maskt)
            if centerpt is not None:
                centroids_tr.append(centerpt)
                tracked_masks_roi.append(maskt)

        # Build a KDTree with the centroids of tracked masks
        tracked_tree = KDTree(centroids_tr)

        selected_masks = []
        for mn in new_masks:
            mask_n = mn.mask
            centroid_new = self.compute_centroid(mask_n)

            # Find the k nearest masks in tracked masks to the mask in new masks
            distances, indices = tracked_tree.query(centroid_new, k=k)
            best_mask = None
            max_iou = 0
            for idx, dist in zip(indices, distances):

                mask_t = tracked_masks_roi[idx]

                iou = self.compute_iou(mask_t, mask_n)
                # print("IoU: ", iou)
                # print("Distance: ", dist)
                # ShowMe([mask_t, mask_n])
                if iou > max_iou:
                    max_iou = iou
                    best_mask = mask_n

            if max_iou > threshold:
                selected_masks.append(best_mask)
                # ShowMe([best_mask], .7)

        highest_iou_masks = np.array([mask for mask in selected_masks], dtype=np.uint8)

        return highest_iou_masks, centroids_tr
