import cv2
import numpy as np
from scipy.spatial import KDTree
from sam_annotator.mask_visualizations import MaskData
from filterpy.kalman import KalmanFilter
from sam_annotator.mask_visualizations import AnnotationObject


class PanoImageAligner:
    def __init__(self):
        self.images: list[np.ndarray] = []
        self.matching_masks: list[list[MaskData]] = []
        self.h_matrix: np.ndarray = None

    def add_annotation(self, annot: AnnotationObject):
        img = annot.img
        masks = annot.good_masks
        self.add_image(img, masks)

    def track(self, annot: AnnotationObject):
        img = annot.img
        self.add_image(img)
        return self.match_and_align()

    def add_image(self, img: np.ndarray, masks: list[MaskData] = None):
        imgbgr = img
        img_gray = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2GRAY)

        # Add the image to the list
        self.images.append(img_gray)
        if masks is not None:
            self.matching_masks.append(masks)
        if len(self.images) > 2:
            # Keep only the two most recent images and masks
            self.images.pop(0)
            if len(self.matching_masks) > 1:
                self.matching_masks.pop(0)

    def match_and_align(self):
        for masks in self.matching_masks:
            if len(masks) == 0:
                return []

        assert (
            len(self.images) == 2 and len(self.matching_masks) >= 1,
            "Two images are required for panorama tracking",
        )

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
        """img_matches = cv2.drawMatches(
            img_gray1,
            kp1,
            img_gray2,
            kp2,
            matches[:10],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )"""

        # Estimate homography.
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        self.h_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp image using homography.
        height, width = img_gray2.shape

        tracked_masks = self._warp_masks(self.matching_masks[0], width, height)

        """annotimg = cv2.cvtColor(img_gray2.copy(), cv2.COLOR_GRAY2BGR)

        for m in self.masks[1]:
            mask = m.mask
            annotimg = self._get_mask_cnts(annotimg, mask, col=(10,10,10))
        for m in tracked_masks:
            annotimg = self._get_mask_cnts(annotimg, m, col=(100, 100, 100))
        for m in selected_masks:
            #mask = cv2.warpPerspective(mask, self.h_matrix, (width, height))
            annotimg = self._get_mask_cnts(annotimg, m, col=(255,0,0))
        for pt in centerpts:
            y, x = pt
            annotimg = cv2.circle(annotimg, (x, y), 2, (0, 255, 0), -1)

        # Save and display the results.

        cv2.imshow("Annotated Image", annotimg)
        cv2.waitKey(0)"""

        if len(self.matching_masks) == 2:
            selected_masks, centerpts = self._select_masks_with_highest_iou(
                tracked_masks, self.matching_masks[1]
            )
            print(len(selected_masks), "masks selected in panorama tracking")
            return [MaskData(mask, "Panorama_tracking") for mask in selected_masks]

        elif len(self.matching_masks) == 1:
            bboxes_np = self._masks_to_bboxes_np(tracked_masks)
            return bboxes_np

    def _warp_masks(self, masks, w, h):
        return [
            cv2.warpPerspective(
                warp_m.mask, self.h_matrix, (w, h), flags=cv2.INTER_NEAREST
            )
            for warp_m in masks
        ]

    def _get_mask_cnts(self, annnotimg, mask, thickness=2, col=None):
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

    def _compute_iou(self, mask1, mask2):
        # Compute intersection
        intersection = np.logical_and(mask1, mask2).sum()
        # Compute union
        union = np.logical_or(mask1, mask2).sum()
        # Compute IoU
        iou = intersection / union if union != 0 else 0
        return iou

    def _compute_centroid(self, mask):
        indices = np.argwhere(mask)
        if len(indices) == 0:
            return None
        centroid = np.mean(indices, axis=0, dtype=np.int32)
        return centroid

    def _select_masks_with_highest_iou(
        self, tracked_masks, new_masks, threshold=0.7, k=3
    ):
        # Compute centroids for all masks in tracked masks
        centroids_tr = []
        tracked_masks_roi = []
        for maskt in tracked_masks:
            centerpt = self._compute_centroid(maskt)
            if centerpt is not None:
                centroids_tr.append(centerpt)
                tracked_masks_roi.append(maskt)

        if len(tracked_masks_roi) < k:
            k = len(tracked_masks_roi)
        # Build a KDTree with the centroids of tracked masks
        tracked_tree = KDTree(centroids_tr)

        selected_masks = []
        for mn in new_masks:
            mask_n = mn.mask
            centroid_new = self._compute_centroid(mask_n)

            # Find the k nearest masks in tracked masks to the mask in new masks
            distances, indices = tracked_tree.query(centroid_new, k=k)
            best_mask = None
            max_iou = 0
            if len(tracked_masks_roi) == 1:
                max_iou = self._compute_iou(tracked_masks_roi[0], mask_n)
                best_mask = tracked_masks_roi[0]
            else:
                for idx, dist in zip(indices, distances):

                    mask_t = tracked_masks_roi[idx]

                    iou = self._compute_iou(mask_t, mask_n)
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

    def _masks_to_bboxes_np(self, masks: list[np.ndarray]):
        bboxes = []
        for mask in masks:
            if np.any(mask):
                mask = mask.astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                x, y, w, h = cv2.boundingRect(contours[0])
                bboxes.append([x, y, x + w, y + h])
        return bboxes


class MultiObjectTracker:
    def __init__(self, dt=1.0, process_var=1.0, measurement_var=1.0):
        # Dictionary to store multiple KalmanObjectTracker instances
        self.trackers = {}
        self.dt = dt
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.tracked_masks: list[MaskData] = []

    def add_annotation(self, annot: AnnotationObject):
        self._maintain_tracker_objs(annot.good_masks)

    def track(self, annot: AnnotationObject):
        # return {mobj.mid: mobj.kalman.predict() for mobj in self.tracked_masks}
        return [mobj.kalman.predict() for mobj in self.tracked_masks]

    def _maintain_tracker_objs(self, masks: list[MaskData]):
        self.tracked_masks = []
        for mobj in masks:
            bbox = self._bbox_from_mask(mobj.mask)
            if mobj.kalman is None:
                mobj.kalman = KalmanObjectTracker(
                    dt=self.dt,
                    process_var=self.process_var,
                    measurement_var=self.measurement_var,
                )
                mobj.kalman.init_state(*bbox)
            else:
                mobj.kalman.update(*bbox)
            self.tracked_masks.append(mobj)

    def _bbox_from_mask(self, mask: np.ndarray):
        """
        Compute the bounding box of a binary mask using numpy.

        :param mask: A binary mask (2D numpy array) where the region of interest has non-zero values.
        :return: A tuple (x, y, w, h) representing the bounding box.
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return x_min, y_min, x_max, y_max


class KalmanObjectTracker:
    def __init__(self, dt=1.0, process_var=1.0, measurement_var=1.0):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.dt = dt  # Time step

        # State Transition Matrix (F)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, dt, 0, 0, 0],
                [0, 1, 0, 0, 0, dt, 0, 0],
                [0, 0, 1, 0, 0, 0, dt, 0],
                [0, 0, 0, 1, 0, 0, 0, dt],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # Measurement Matrix (H)
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        # Initial State (x1, y1, x2, y2, dx1, dy1, dx2, dy2)
        self.kf.x = np.zeros(
            8
        )  # Initialize all to 0, can be set to the initial bounding box and velocities

        # Process Noise Covariance Matrix (Q)
        q = process_var
        self.kf.Q = q * np.array(
            [
                [dt**4 / 4, 0, 0, 0, dt**3 / 2, 0, 0, 0],
                [0, dt**4 / 4, 0, 0, 0, dt**3 / 2, 0, 0],
                [0, 0, dt**4 / 4, 0, 0, 0, dt**3 / 2, 0],
                [0, 0, 0, dt**4 / 4, 0, 0, 0, dt**3 / 2],
                [dt**3 / 2, 0, 0, 0, dt**2, 0, 0, 0],
                [0, dt**3 / 2, 0, 0, 0, dt**2, 0, 0],
                [0, 0, dt**3 / 2, 0, 0, 0, dt**2, 0],
                [0, 0, 0, dt**3 / 2, 0, 0, 0, dt**2],
            ]
        )

        # Measurement Noise Covariance Matrix (R)
        r = measurement_var
        self.kf.R = r * np.eye(4)

        # Initial State Covariance Matrix (P)
        self.kf.P = np.eye(8)

    def init_state(self, x1, y1, x2, y2, dx1=0, dy1=0, dx2=0, dy2=0):
        self.kf.x = np.array([x1, y1, x2, y2, dx1, dy1, dx2, dy2])

    def predict(self):
        self.kf.predict()
        return self.kf.x[:4]  # Return the predicted bounding box (x1, y1, x2, y2)

    def update(self, measurement: tuple[float, float, float, float]):
        self.kf.update(measurement)
        return self.kf.x[:4]  # Return the updated bounding box (x1, y1, x2, y2)


"""# Simulated Measurements
true_positions = np.array([
    [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]
])
noise_std_dev = 0.5
measurements = true_positions + np.random.normal(0, noise_std_dev, true_positions.shape)

# Initialize the Kalman Tracker
tracker = KalmanObjectTracker(dt=1.0, process_var=1.0, measurement_var=1.0)

predicted_positions = []
for measurement in measurements:
    # Prediction step
    pred_pos = tracker.predict()
    
    # Update step
    updated_pos = tracker.update(measurement)
    
    # Store the predicted position
    predicted_positions.append(updated_pos)

predicted_positions = np.array(predicted_positions)"""
