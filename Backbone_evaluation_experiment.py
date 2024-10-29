from segment_anything import sam_model_registry, SamPredictor
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

import torch.nn.functional as F
import pandas as pd


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


class SamTestInference:
    def __init__(
        self, sam_gen: int, weights_path: str, config: str, device: str = "cuda"
    ):
        self.device = device
        if sam_gen == 1:
            checkpoint = torch.load(
                weights_path, map_location=torch.device(device), weights_only=True
            )
            self.sam_model = sam_model_registry[config]()
            self.sam_model.load_state_dict(checkpoint)
            self.sam_model.to(device)
            self.sam_predictor = SamPredictor(self.sam_model)
        elif sam_gen == 2:
            self._init_mixed_precision()
            self.sam_model = build_sam2(config, ckpt_path=weights_path, device=device)
            self.sam_predictor = SAM2ImagePredictor(self.sam_model)

        else:
            raise ValueError("Invalid SAM generation")

    def _init_mixed_precision(self):
        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _one_hot_tensor(self, mask: np.ndarray, num_classes: int = 2, dtype=torch.long):
        cold_tensor = torch.tensor(mask // 255).to(dtype)

        one_hot_mask = F.one_hot(cold_tensor, num_classes=num_classes).permute(2, 0, 1)
        return one_hot_mask

    def _preprocess_masks(self, masks):
        return [self._one_hot_tensor(mask) for mask in masks]

    def compute_metrics(self, pred_masks, gt_masks):

        pred_masks = self._preprocess_masks(pred_masks)
        gt_masks = self._preprocess_masks(gt_masks)

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
        iou_metric = MeanIoU(
            include_background=True, reduction="mean", get_not_nans=False
        )
        iou_metric(pred_masks, gt_masks)
        iou_value = iou_metric.aggregate().item()

        cm_metric = ConfusionMatrixMetric(
            include_background=False, metric_name="precision"
        )
        cm_metric(pred_masks, gt_masks)
        cm_value = cm_metric.aggregate()
        print("Precision", cm_value)

        iou_metric.reset()
        gdice_metric.reset()
        dice_metric.reset()

        return {
            "Mean Dice": dice_value,
            "Generalized Dice": gdice_value,
            "Mean IoU": iou_value,
        }

    def get_sam_inference(self, test_img_objs: list[TestImages]):

        gt_masks = []
        pred_masks = []

        for test_img_obj in test_img_objs:
            self.sam_predictor.set_image(test_img_obj.rgb)
            for gt_mask, center in test_img_obj:
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=center,
                    point_labels=np.array([1]),
                    multimask_output=False,
                )
                pred_mask = masks[0].astype(np.uint8) * 255
                pred_masks.append(pred_mask)
                gt_masks.append(gt_mask)
                # draw_mask(test_img_obj.bgr, gt_mask, pred_mask, center)
        metrics = self.compute_metrics(pred_masks, gt_masks)
        print("N masks", len(pred_masks), len(gt_masks))
        # self.compute_torch_metrics(pred_masks, gt_masks)
        return metrics

    def draw_mask(img, gtmask, predmask, center):
        show_img = img.copy()
        contours, _ = cv2.findContours(
            gtmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(show_img, contours, -1, (0, 255, 0), 1)
        contours, _ = cv2.findContours(
            predmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(show_img, contours, -1, (255, 0, 0), 1)
        cv2.circle(show_img, tuple(center[0]), 3, (0, 0, 255), -1)
        show_img = cv2.resize(show_img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("img", show_img)
        cv2.waitKey(0)


class SamFlavors:
    def __init__(self):
        pass

    @property
    def _all_sams(self):
        return {
            1: {
                "weights": [
                    "sam_vit_b_01ec64.pth",
                    "sam_vit_l_0b3195.pth",
                    "sam_vit_h_4b8939.pth",
                ],
                "config": ["vit_b", "vit_l", "vit_h"],
            },
            2: {
                "weights": [
                    "sam2_hiera_tiny.pt",
                    "sam2_hiera_small.pt",
                    "sam2_hiera_base_plus.pt",
                    "sam2_hiera_large.pt",
                    "sam2.1_hiera_tiny.pt",
                    "sam2.1_hiera_small.pt",
                    "sam2.1_hiera_base_plus.pt",
                    "sam2.1_hiera_large.pt",
                ],
                "config": [
                    "sam2_hiera_t.yaml",
                    "sam2_hiera_s.yaml",
                    "sam2_hiera_b+.yaml",
                    "sam2_hiera_l.yaml",
                    "configs/sam2.1/sam2.1_hiera_t.yaml",
                    "configs/sam2.1/sam2.1_hiera_s.yaml",
                    "configs/sam2.1/sam2.1_hiera_b+.yaml",
                    "configs/sam2.1/sam2.1_hiera_l.yaml",
                ],
            },
        }

    def __iter__(self):
        for sam_gen, sam_params in self._all_sams.items():
            for weights, config in zip(sam_params["weights"], sam_params["config"]):
                yield sam_gen, weights, config


def main():
    datasets = ["BackboneExperimentData/MaizeEar/"]
    sam_flavors = SamFlavors()
    metric_results = []
    for sam_gen, weights_path, config in sam_flavors:
        sam = SamTestInference(
            sam_gen=sam_gen,
            weights_path=weights_path,
            config=config,
        )
        for dataset in datasets:
            test_imgs = []
            for test_img_dir in glob.glob(dataset + "/*"):
                test_img = TestImages(test_img_dir)
                test_imgs.append(test_img)

            metrics = sam.get_sam_inference(test_imgs)
            metrics["Sam generation"] = sam_gen
            metrics["Weights path"] = weights_path
            metrics["Dataset"] = dataset
            print(metrics)
            metric_results.append(metrics)
        del sam

    df = pd.DataFrame(metric_results)
    df.to_csv("BackboneExperimentData/MaizeEar_results.csv", index=False)


if __name__ == "__main__":
    main()
