import pandas as pd
import glob
import itertools
import torch
import numpy as np
import torchvision.ops.boxes as bops

from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from tqdm import tqdm

from Backbone_evaluation_experiment import TestImages, SamTestInference, SamFlavors


class AmgSamTestInference(SamTestInference):

    def __init__(
        self, sam_gen, weights_path, config, amg_config, device="cuda", iou_thresh=0.5
    ):
        super().__init__(sam_gen, weights_path, config, device)

        self.iou_thresh = iou_thresh
        amg_config["model"] = self.sam_model
        if sam_gen == 1:
            self.amg_predictor = SamAutomaticMaskGenerator(**amg_config)
        elif sam_gen == 2:
            self.amg_predictor = SAM2AutomaticMaskGenerator(**amg_config)
        else:
            raise ValueError("Invalid SAM generation")

    def get_sam_inference(self, test_img_obj: TestImages):
        gt_masks = []
        pred_masks = []
        self.sam_predictor.set_image(test_img_obj.rgb)
        self.sam_predictor
        masks = self.amg_predictor.generate(test_img_obj.rgb)
        pred_masks = [m["segmentation"].astype(np.uint8) * 255 for m in masks]
        gt_masks = test_img_obj.masks
        metrics = self.match_masks_and_evaluate(
            gt_masks=gt_masks, pred_masks=pred_masks
        )
        return metrics

    def match_masks_and_evaluate(
        self, gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]
    ):
        # TODO: I think there is a faster torchvision implementation for this
        gt_boxes, gt_masks = self.get_torch_bbox_of_masks(masks=gt_masks)
        pred_boxes, pred_masks = self.get_torch_bbox_of_masks(masks=pred_masks)

        bbox_iou = bops.box_iou(pred_boxes, gt_boxes)

        total_positives = gt_boxes.shape[0]
        total_predictions = pred_boxes.shape[0]

        best_pred_matches_for_gt_boxes = torch.max(bbox_iou, dim=0)

        gt_ious_with_pred = self.calculate_ious(
            pred_masks=pred_masks,
            gt_masks=gt_masks,
            gt_idcs=best_pred_matches_for_gt_boxes.indices,
        )
        tp = gt_ious_with_pred[gt_ious_with_pred > self.iou_thresh].shape[0]
        fn = total_positives - tp
        fp = total_predictions - tp

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        metrics = {}
        metrics["recall"] = recall
        metrics["precision"] = precision
        metrics["predictions"] = len(pred_masks)
        metrics["ground_truth"] = len(gt_masks)
        metrics["TP"] = tp
        metrics["FP"] = fp
        metrics["FN"] = fn

        return metrics

    def calculate_ious(
        self,
        pred_masks: list[torch.Tensor],
        gt_masks: list[torch.Tensor],
        gt_idcs: torch.Tensor,
    ):
        intersections = gt_masks.logical_and(pred_masks[gt_idcs])
        intersections_sum = intersections.sum((1, 2))
        unions = gt_masks.logical_or(pred_masks[gt_idcs])
        unions_sum = unions.sum((1, 2))
        ious = intersections_sum / unions_sum
        return ious

    def get_torch_bbox_of_masks(self, masks: list[np.ndarray]):
        boxes = []
        mask_color = 255
        for mask in masks:
            y_idcs, x_idcs = np.where(
                np.all(mask.reshape(*mask.shape, 1) == mask_color, axis=-1)
            )
            bbox_x_min = int(x_idcs.min())
            bbox_x_max = int(x_idcs.max())
            bbox_y_min = int(y_idcs.min())
            bbox_y_max = int(y_idcs.max())
            boxes.append([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max])

        return (torch.tensor(boxes), torch.tensor(np.array(masks)))


def create_sam_amg_configs() -> list[dict]:

    param_dict = {
        "points_per_side": [32, 64, 128],
        "points_per_batch": [128],
        "pred_iou_thresh": [0.72, 0.8, 0.88],
        "stability_score_thresh": [0.92, 0.95, 0.98],
        "stability_score_offset": [0.7, 1.0, 1.3],
        "box_nms_thresh": [0.7],
        "crop_n_layers": [0, 1, 2],
        "crop_nms_thresh": [0.7],
        "crop_n_points_downscale_factor": [1, 2, 4],
    }
    keys, values = zip(*param_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def main():
    datasets = ["Exp32"]
    iou_thresh = 0.5
    metric_results = []
    sam_flavors = SamFlavors()
    sam_gen, weights_path, config = next(iter(sam_flavors))
    sam_amg_configs = create_sam_amg_configs()

    for dataset in datasets:

        for test_img_dir in glob.glob(dataset + "/*"):
            test_img = TestImages(test_img_dir)

            for amg_config in tqdm(sam_amg_configs):
                sam = AmgSamTestInference(
                    sam_gen, weights_path, config, amg_config, iou_thresh=iou_thresh
                )

                metrics = sam.get_sam_inference(test_img)
                metrics["Dataset"] = dataset
                metrics["img_dir"] = test_img_dir
                amg_config["model"] = weights_path
                metrics.update(amg_config)
                print(metrics)
                metric_results.append(metrics)
                del sam

                df = pd.DataFrame(metric_results)
                df.to_csv("test.csv", index=False)


if __name__ == "__main__":
    main()
