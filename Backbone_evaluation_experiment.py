from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import cv2
import os
import glob
import numpy as np
from scipy.spatial.distance import cdist
from monai.metrics import (
    MeanIoU,
    DiceMetric,
    GeneralizedDiceScore,
    ConfusionMatrixMetric,
)
from torchmetrics import segmentation
import torch.nn.functional as F


class TestImages:
    def __init__(self, img_folder_dir: str):
        print(glob.glob(img_folder_dir + "/*img*")[0])
        self.bgr = cv2.imread(glob.glob(img_folder_dir + "/*img*")[0])
        self.rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        mask_paths = glob.glob(os.path.join(img_folder_dir, "masks") + "/*.png")
        # if len(mask_paths) != 10:
        # raise ValueError(f"Expected 10 masks, got {len(mask_paths)}")
        self.masks = [
            cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in mask_paths
        ]

    def _get_geometric_median(self, i: int, eps: float = 1e-9):
        # from "The multivariate L1-median and associated data depth" by Yehuda Vardi and Cun-Hui Zhang
        # Implementation https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points

        mask_coords = np.column_stack(np.where(self.masks[i] > 0))
        y = np.mean(mask_coords, 0)

        while True:
            D = cdist(mask_coords, [y])
            nonzeros = (D != 0)[:, 0]

            Dinv = 1 / D[nonzeros]
            Dinvs = np.sum(Dinv)
            W = Dinv / Dinvs
            T = np.sum(W * mask_coords[nonzeros], 0)

            num_zeros = len(mask_coords) - np.sum(nonzeros)
            if num_zeros == 0:
                y1 = T
            elif num_zeros == len(mask_coords):
                return y
            else:
                R = (T - y) * Dinvs
                r = np.linalg.norm(R)
                rinv = 0 if r == 0 else num_zeros / r
                y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

            if np.linalg.norm(y - y1) < eps:
                distances = np.linalg.norm(mask_coords - y1, axis=1)
                closest_index = np.argmin(distances)
                closest_point = mask_coords[closest_index]
                cy, cx = closest_point

                return cx, cy

            y = y1

    def __iter__(self):
        for i in range(len(self.masks)):
            cx, cy = self._get_geometric_median(i)
            if self.masks[i][cy, cx] != 255:
                raise ValueError("Mass center lies outside the mask")
            yield self.masks[i], np.array([[cx, cy]])


def init_sam_model(sam_gen: int, weights_path: str, config: str, device: str = "cuda"):
    if sam_gen == 1:
        checkpoint = torch.load(
            weights_path, map_location=torch.device(device), weights_only=True
        )
        sam_model = sam_model_registry[config]()
        sam_model.load_state_dict(checkpoint)
        sam_predictor = SamPredictor(sam_model)
    elif sam_gen == 2:
        sam_model = build_sam2(config, ckpt_path=weights_path, device=device)
        sam_predictor = SAM2ImagePredictor(sam_model)

    else:
        raise ValueError("Invalid SAM generation")
    return sam_predictor


def one_hot_tensor(mask: np.ndarray, num_classes: int = 2, dtype=torch.long):
    cold_tensor = torch.tensor(mask // 255).to(dtype)

    one_hot_mask = F.one_hot(cold_tensor, num_classes=num_classes).permute(2, 0, 1)
    return one_hot_mask


def preprocess_masks(masks):
    return [one_hot_tensor(mask) for mask in masks]


def compute_torch_metrics(pred_masks, gt_masks):

    pred_masks = preprocess_masks(pred_masks)
    pred_masks = torch.stack(pred_masks)  # .unsqueeze(1)
    gt_masks = preprocess_masks(gt_masks)
    gt_masks = torch.stack(gt_masks)  # .unsqueeze(1)

    iou_metric = segmentation.MeanIoU(
        num_classes=2, input_format="one-hot", include_background=True
    )
    print(iou_metric(pred_masks, gt_masks))


def compute_metrics(pred_masks, gt_masks):

    pred_masks = preprocess_masks(pred_masks)
    gt_masks = preprocess_masks(gt_masks)

    # Dice Metric
    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )
    dice_metric(y_pred=pred_masks, y=gt_masks)
    dice_value = dice_metric.aggregate().item()

    gdice_metric = GeneralizedDiceScore(include_background=True, reduction="mean")
    gdice_metric(pred_masks, gt_masks)
    gdice_value = gdice_metric.aggregate().item()

    # IoU Metric (Jaccard Index)
    iou_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    iou_metric(pred_masks, gt_masks)
    iou_value = iou_metric.aggregate().item()

    cm_metric = ConfusionMatrixMetric(include_background=True, metric_name="precision")
    cm_metric(pred_masks, gt_masks)
    cm_value = cm_metric.aggregate()
    print("CM", cm_value)

    iou_metric.reset()
    gdice_metric.reset()
    dice_metric.reset()

    return {
        "Mean Dice": dice_value,
        "Generalized Dice": gdice_value,
        "Mean IoU": iou_value,
    }


def get_sam_inference(
    sam_predictor: SamPredictor | SAM2ImagePredictor, test_img_obj: TestImages
):
    sam_predictor.set_image(test_img_obj.rgb)
    gt_masks = []
    pred_masks = []

    for gt_mask, center in test_img_obj:
        masks, scores, logits = sam_predictor.predict(
            point_coords=center, point_labels=np.array([1]), multimask_output=False
        )
        pred_mask = masks[0].astype(np.uint8) * 255
        pred_masks.append(pred_mask)
        gt_masks.append(gt_mask)
        # draw_mask(test_img_obj.bgr, gt_mask, pred_mask, center)

    metrics = compute_metrics(pred_masks, gt_masks)
    compute_torch_metrics(pred_masks, test_img_obj.masks)
    print(metrics)


def draw_mask(img, gtmask, predmask, center):
    show_img = img.copy()
    contours, _ = cv2.findContours(gtmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(show_img, contours, -1, (0, 255, 0), 1)
    contours, _ = cv2.findContours(predmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(show_img, contours, -1, (255, 0, 0), 1)
    cv2.circle(show_img, tuple(center[0]), 3, (0, 0, 255), -1)
    show_img = cv2.resize(show_img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("img", show_img)
    cv2.waitKey(0)


def main():
    # test_img_dirs = glob.glob("BackboneExperimentData/MaizeUAV/*")
    test_img_dirs = glob.glob(
        "/media/geink81/hddE/3DMais/Ears/LabelData/EarsBreeders/20MP/done/39320223017022_done/*"
    )
    for test_img_dir in test_img_dirs:
        test_img = TestImages(test_img_dir)
        sam = init_sam_model(
            sam_gen=1,
            # weights_path="/home/geink81/pythonstuff/CobScanws/sam_vit_h_4b8939.pth",
            weights_path="sam_vit_b_01ec64.pth",
            config="vit_b",
        )
        get_sam_inference(sam, test_img)

        """img_show = test_img.bgr.copy()
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img_show, contours, -1, (0, 255, 0), 1)
        cv2.circle(img_show, center, 5, (0, 0, 255), -1)
        img_show = cv2.resize(img_show, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("img", img_show)
        cv2.waitKey(0)"""


if __name__ == "__main__":
    main()
