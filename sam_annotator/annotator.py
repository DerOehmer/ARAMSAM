import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path

from src.run_sam import SamInference
from dataclasses import dataclass

# from src.sam_mask_inspection import SelectMask, ImgCheckup
# from src.sam_mask_creation import ImageData


@dataclass
class MaskData:
    mask: np.ndarray
    origin: str


class AnnotationObject:
    def __init__(self, filepath: Path) -> None:
        self.filepath: str = filepath
        self.pil_image: Image = Image.open(filepath)
        self.img: np.array = np.array(self.pil_image, dtype=np.uint8)

        if self.img.shape[2] == 4:
            print("Loaded image with 4 channels - ignoring last")
            self.img = self.img[:, :, :3]
        self.masks: MaskData = []
        self.good_masks = []
        self.mask_decisions = np.zeros((len(self.masks)), dtype=bool)
        self.masked_img = np.array(self.pil_image).copy()
        self.mask_collection = np.zeros_like(self.img)
        self.current_mask = None

    def check_img(self):
        pass

    def set_masks(self, maskobject: MaskData):
        self.masks = maskobject
        self.mask_decisions = np.zeros((len(self.masks)), dtype=bool)


class Annotator:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = SamInference(
            sam_checkpoint="sam_vit_b_01ec64.pth", model_type="vit_b", device=device
        )
        self.annotation: AnnotationObject = None
        self.mask_idx = 0

    def update_mask_idx(self, new_idx: int = 0):
        self.mask_idx = new_idx
        self.annotation.current_mask = None

    def create_new_annotation(self, filepath: Path):
        self.annotation = AnnotationObject(filepath=filepath)

    def predict_with_sam(self):
        if self.annotation is None:
            raise ValueError("No annotation object found.")

        print(self.annotation.img.shape)
        self.sam.image_embedding(self.annotation.img)
        masks, annotated_image = self.sam.custom_amg(roi_pts=False, n_points=100)
        self.annotation.masked_img = annotated_image
        self.annotation.set_masks(
            [MaskData(mask=mask, origin="sam_proposed") for mask in masks]
        )
        self.update_mask_idx()
        self.annotation.current_mask = self.annotation.masks[self.mask_idx]
        self.preselect_mask()

    def good_mask(self):
        done = False
        annot = self.annotation
        annot.good_masks.append(annot.masks[self.mask_idx])
        annot.mask_decisions[self.mask_idx] = True
        # add track if not last idx
        self.update_collections(annot)
        self.mask_idx += 1
        annot.current_mask = annot.masks[self.mask_idx]
        if self.mask_idx == len(annot.masks):
            done = True  # all masks have been labeled
        else:
            self.preselect_mask()
        return done

    def bad_mask(self):
        done = False
        annot = self.annotation
        annot.mask_decisions[self.mask_idx] = False
        self.mask_idx += 1
        annot.current_mask = annot.masks[self.mask_idx]
        if self.mask_idx == len(annot.masks):
            done = True  # all masks have been labeled
        else:
            self.preselect_mask()
        return done

    def preselect_mask(self, max_overlap_ratio: float = 0.4):
        annot = self.annotation
        mask_obj: MaskData = annot.masks[self.mask_idx]
        mask = mask_obj.mask
        maskorigin = mask_obj.origin

        # imgresult = cv2.bitwise_and(annot.img, annot.img ,mask=mask)

        # show_lst = [self.annotation.masked_img, imgresult]

        mask_coll_bin = (
            np.any(annot.mask_collection != [0, 0, 0], axis=-1).astype(np.uint8) * 255
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
                self.update_collections(annot)
            annot.mask_decisions[self.mask_idx] = False
            self.mask_idx -= 1

    def update_collections(self, annot: AnnotationObject):

        y, x, _ = annot.img.shape
        mask_coll = np.zeros((y, x, 3), dtype=np.uint8)
        mask_coll_bin = np.zeros((y, x), dtype=np.uint8)
        cnt_coll = annot.img.copy()

        thickness = -1

        for m in annot.good_masks:
            m = m.mask
            overlap = cv2.bitwise_and(m, mask_coll_bin)
            mask_coll_bin = np.where(m == 255, 255, mask_coll_bin)
            mask_coll[np.where(m == 255)] = [255, 255, 255]
            mask_coll[np.where(overlap == 255)] = [0, 0, 255]

            cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            b, g, r = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )
            cnt_coll = cv2.drawContours(
                cnt_coll, cnts, -1, (b, g, r), thickness, lineType=cv2.LINE_8
            )

        self.annotation.mask_collection = mask_coll
        self.annotation.masked_img = cnt_coll
