import numpy as np
import torch
from PIL import Image
from pathlib import Path

from src.run_sam import SamInference


class AnnotationObject:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.pil_image = Image.open(filepath)
        self.img = np.array(self.pil_image)
        if self.img.shape[2] == 4:
            print("Loaded image with 4 channels - ignoring lase")
            self.img = self.img[:, :, :3]
        self.mask = np.zeros(self.img.shape, dtype="uint8")
        self.masked_img = np.array(self.pil_image).copy()

    def check_img(self):
        pass


class Annotator:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"	
        self.sam = SamInference(
            sam_checkpoint="sam_vit_b_01ec64.pth", model_type="vit_b", device=device
        )
        self.annotation = None

    def create_new_annotation(self, filepath: Path):
        self.annotation = AnnotationObject(filepath=filepath)

    def predict_with_sam(self, annotation_object: AnnotationObject) -> AnnotationObject:
        print(annotation_object.img.shape)
        self.sam.image_embedding(annotation_object.img)
        masks, annotated_image = self.sam.custom_amg(roi_pts=False, n_points=100)
        annotation_object.masked_img = annotated_image
        return annotation_object
