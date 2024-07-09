from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch
import cv2
import numpy as np
import pandas as pd
import threading


class SamInference:
    def __init__(
        self, sam_checkpoint="sam_vit_b_01ec64.pth", model_type="vit_b", device="cpu"
    ):
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        
        self.device = device
        self.lock = threading.Lock()
        self.img_embed_thread = None
        self.is_img_embedded = False

        checkpoint = torch.load(sam_checkpoint, map_location=torch.device(device))

        # Initialize the model from the registry without loading the checkpoint
        self.sam = sam_model_registry[model_type]()

        # Load the state_dict from the checkpoint
        self.sam.load_state_dict(checkpoint)

        # Move the model to the appropriate device
        self.sam.to(device=self.device)

        self.predictor = SamPredictor(self.sam)
        self.img = None

    def amg(self, img, msize_thresh=10000, pt_grid=None):

        sam_amg = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=40,
            points_per_batch=64,
            pred_iou_thresh=0.88,
            box_nms_thresh=0.4,
            point_grids=pt_grid,
        )
        result = sam_amg.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        masks = []
        xmins = []
        ymins = []
        annotated_image = img.copy()
        for mask in result:
            if mask["area"] < msize_thresh:
                b, g, r = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                )
                kernelmask = np.array(mask["segmentation"], dtype=np.uint8) * 255
                masks.append(kernelmask)
                ycords, xcords = np.where(kernelmask == 255)
                annotated_image[ycords, xcords] = b, g, r
                ymin, xmin = np.amin(ycords), np.amin(xcords)

                ymins.append(ymin)
                xmins.append(xmin)

        cord_dict = {"ymins": ymins, "xmins": xmins}

        cord_df = pd.DataFrame(cord_dict)
        cord_df = cord_df.sort_values(by=["ymins", "xmins"])
        order = cord_df.index.to_list()
        masks = np.array(masks, dtype=np.uint8)[order]
        return masks, annotated_image

    def image_embedding(self, img):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        self.imgbgr = img
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self.lock:
            self.is_img_embedded = True
            self.img_embed_thread = threading.Thread(
                target=self.predictor.set_image, args=(self.img,)
            )
            self.img_embed_thread.start()

    def predict_batch(
        self, img, pts=None, pts_labels=None, bboxes=None, mask_input=None
    ):
        pass

    def predict(self, pts=None, pts_labels=None, bboxes=None, mask_input=None):
        if not self.is_img_embedded:
            raise ValueError("Image embedding has not been initialized yet")
        with self.lock:
            if self.img_embed_thread.is_alive():
                print("Waiting for image embedding to finish...")
            self.img_embed_thread.join()
            masks, scores, logits = self.predictor.predict(
                point_coords=pts,
                point_labels=pts_labels,
                box=bboxes,
                mask_input=mask_input,
                multimask_output=True,
            )
        return self.select_masks(masks)

    def select_masks(self, masks, alpha=0.1):
        while True:
            for mask in masks:
                mask = mask.astype(np.uint8)
                green_mask = cv2.merge(
                    [
                        np.zeros_like(mask),
                        255 * (mask > 0).astype(np.uint8),
                        np.zeros_like(mask),
                    ]
                )
                result0 = cv2.addWeighted(self.imgbgr, 1 - alpha, green_mask, alpha, 0)
                result = self.zoom_in(result0, mask)
                cv2.imshow("mask", result)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("n"):
                    cv2.destroyWindow("mask")
                    return mask * 255
                if key == ord("m"):
                    cv2.destroyWindow("mask")
                    return []
                if key == ord("x"):
                    cv2.destroyWindow("mask")
                    raise KeyboardInterrupt()

    def zoom_in(self, annotation, binmask, zoom_factor=3):
        yindcs, xindcs = np.where(binmask > 0)
        ymin, ymax = np.amin(yindcs), np.amax(yindcs)
        xmin, xmax = np.amin(xindcs), np.amax(xindcs)
        w, h = xmax - xmin, ymax - ymin
        zoom_sizex, zoom_sizey = int(zoom_factor * w), int(
            zoom_factor * h
        )  # Adjust zoom size as needed
        zoomxy = max(0, xmin - zoom_sizex), max(0, ymin - zoom_sizey)
        x1, y1 = zoomxy
        x2, y2 = min(annotation.shape[1], xmax + zoom_sizex), min(
            annotation.shape[0], ymax + zoom_sizey
        )

        # Extract zoomed region and resize
        zoomed_img0 = annotation[y1:y2, x1:x2]
        zoomed_img = cv2.resize(zoomed_img0, (0, 0), fx=zoom_factor, fy=zoom_factor)
        return zoomed_img

    def reset_img(self):
        self.predictor.reset_image()
