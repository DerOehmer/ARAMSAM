from ultralytics import YOLO


class YOLOInference:
    def __init__(self):
        self.model = None

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load the YOLO model checkpoint from the specified file path.

        Args:
            checkpoint_path (str): The file path to the model checkpoint (.pt file).
        """
        self.model = YOLO(checkpoint_path)
        print(f"Checkpoint loaded from {checkpoint_path}")

    def infer_image(self, image_path: str):
        """
        Run inference on a single image and return the results.

        Args:
            image_path (str): The file path to the image.

        Returns:
            The inference results as returned by the YOLO model.
        """
        if self.model is None:
            raise ValueError(
                "Model not loaded. Please load a checkpoint first using load_checkpoint()."
            )

        results = self.model(image_path)

        bboxes = results.xyxy[0].cpu().numpy()
        return bboxes
