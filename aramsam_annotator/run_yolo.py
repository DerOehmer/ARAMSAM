from ultralytics import YOLO
from aramsam_annotator.mask_visualizations import MaskData, MaskIdHandler
import numpy as np


class YoloInference:
    def __init__(self, object_id_handler: MaskIdHandler):
        self.model = None
        self.image = None
        self.obj_ident = object_id_handler

    def set_img(self, img):
        self.image = img

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load the YOLO model checkpoint from the specified file path.

        Args:
            checkpoint_path (str): The file path to the model checkpoint (.pt file).
        """
        self.model = YOLO(checkpoint_path)
        print(f"Checkpoint loaded from {checkpoint_path}")

    def _get_mask_data_instance(self, yolo_res, idx):
        bbox = yolo_res[0].boxes.xyxy[idx].cpu().numpy().astype(np.int32)

        return MaskData(
            mid=self.obj_ident.get_id(), origin="Yolo_prediction", bbox=bbox
        )

    def infer_image(self):
        """
        Run inference on a single image and return the results.

        Args:
            image_path (str): The file path to the image.

        Returns:
            The inference results as bboxes returned by the YOLO model.
        """
        if self.model is None:
            raise ValueError(
                "Model not loaded. Please load a checkpoint first using load_checkpoint()."
            )
        if self.image is None:
            raise ValueError("No image set. Please set an image using set_img().")

        results = self.model(self.image)
        n_boxes = len(results[0].boxes)

        bboxes = [self._get_mask_data_instance(results, i) for i in range(n_boxes)]
        return bboxes


if __name__ == "__main__":
    yolo = YoloInference()
    yolo.load_checkpoint("KernelYOLO8x.pt")
    yolo.set_img(
        "ExperimentData/AmgEvaluationData/39320223138024_low_3136/39320223138024_low_3136_img.png"
    )
    bboxes = yolo.infer_image()
    print(bboxes)
