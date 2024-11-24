import pandas as pd
import glob
import itertools
import torch
import numpy as np
import torchvision.ops.boxes as bops
from torchvision.ops import masks_to_boxes
from segment_anything import sam_model_registry
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from tqdm import tqdm
from natsort import natsorted
import multiprocessing as mp

from Backbone_evaluation_experiment import TestImages


class AmgSamTestInference:

    def __init__(
        self,
        sam_gen: int,
        weights_path: str,
        config: str,
        amg_config,
        device: str = "cuda",
        iou_thresh: float = 0.5,
    ):
        self.device = device
        self.iou_thresh = iou_thresh
        self.amg_config = amg_config
        if sam_gen == 1:
            checkpoint = torch.load(
                weights_path, map_location=torch.device(device), weights_only=True
            )
            self.sam_model = sam_model_registry[config]()
            self.sam_model.load_state_dict(checkpoint)
            self.sam_model.to(device)
            self.amg_predictor = SamAutomaticMaskGenerator(self.sam_model, **amg_config)
        elif sam_gen == 2:
            self._init_mixed_precision()
            self.sam_model = build_sam2(config, ckpt_path=weights_path, device=device)
            self.amg_predictor = SAM2AutomaticMaskGenerator(
                self.sam_model, **amg_config
            )

        else:
            raise ValueError("Invalid SAM generation")

    def _init_mixed_precision(self):
        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def get_sam_inference(self, test_img_obj: TestImages):
        gt_masks = []
        pred_masks = []
        try:
            masks = self.amg_predictor.generate(test_img_obj.rgb)
            pred_masks = [m["segmentation"].astype(np.uint8) * 255 for m in masks]
            gt_masks = test_img_obj.masks
            error = None
        except Exception as e:
            print(f"Current amg config failed: {e}")
            pred_masks = []
            error = type(e).__name__

        metrics = self.match_masks_and_evaluate(
            gt_masks=gt_masks, pred_masks=pred_masks, error=error
        )
        return metrics

    def match_masks_and_evaluate(
        self, gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], error=None
    ):
        if len(pred_masks) == 0:
            # SAM did not find any masks or crashed
            recall = 0
            precision = 0
            tp = 0
            fp = 0
            fn = len(gt_masks)
            mean_iou_of_tp = 0
            std_iou_of_tp = 0
            n_pred_masks = 0

        elif len(pred_masks) > 0:
            # SAM found at least one mask
            gt_boxes = masks_to_boxes(torch.tensor(np.array(gt_masks)))
            gt_masks = torch.tensor(np.array(gt_masks))
            pred_boxes = masks_to_boxes(torch.tensor(np.array(pred_masks)))
            pred_masks = torch.tensor(np.array(pred_masks))

            bbox_iou = bops.box_iou(pred_boxes, gt_boxes)

            total_positives = gt_boxes.shape[0]
            total_predictions = pred_boxes.shape[0]

            best_pred_matches_for_gt_boxes = torch.max(bbox_iou, dim=0)

            gt_ious_with_pred = self.calculate_ious(
                pred_masks=pred_masks,
                gt_masks=gt_masks,
                gt_idcs=best_pred_matches_for_gt_boxes.indices,
            )

            mean_iou_of_tp = (
                gt_ious_with_pred[gt_ious_with_pred > self.iou_thresh].mean().item()
            )
            std_iou_of_tp = (
                gt_ious_with_pred[gt_ious_with_pred > self.iou_thresh].std().item()
            )
            tp = gt_ious_with_pred[gt_ious_with_pred > self.iou_thresh].shape[0]
            fn = total_positives - tp
            fp = total_predictions - tp

            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            n_pred_masks = len(pred_masks)

        metrics = {}
        metrics["recall"] = recall
        metrics["precision"] = precision
        metrics["predictions"] = n_pred_masks
        metrics["ground_truth"] = len(gt_masks)
        metrics["TP"] = tp
        metrics["FP"] = fp
        metrics["FN"] = fn
        metrics["mean_iou_of_tp"] = mean_iou_of_tp
        metrics["std_iou_of_tp"] = std_iou_of_tp
        metrics["error"] = error

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


def sam_process(
    sam_gen,
    weights_path,
    config,
    amg_config,
    test_img_dir,
    dataset,
    iou_thresh,
    test_img,
    queue,
):
    # https://github.com/facebookresearch/segment-anything/issues/743
    sam = AmgSamTestInference(
        sam_gen,
        weights_path,
        config,
        amg_config,
        iou_thresh=iou_thresh,
        device="cuda:1",
    )

    metrics = sam.get_sam_inference(test_img)
    metrics["Dataset"] = dataset
    metrics["img_dir"] = test_img_dir
    metrics["model"] = weights_path
    metrics.update(amg_config)
    queue.put(metrics)
    print(metrics)
    del sam
    torch.cuda.empty_cache()


def main():
    datasets = ["ExperimentData/AmgEvaluationData"]
    start_img = 8
    end_img_exclusive = 9

    iou_thresh = 0.8
    metric_results = []
    sam_gen = 2
    weights_path = "sam2.1_hiera_small.pt"
    config = "configs/sam2.1/sam2.1_hiera_s.yaml"
    # sam_gen = 2
    # weights_path = "sam2_hiera_small.pt"
    # config = "sam2_hiera_s.yaml"
    mp.set_start_method("spawn")
    queue = mp.Queue()
    sam_amg_configs = create_sam_amg_configs()

    for dataset in datasets:

        test_img_dirs = natsorted(glob.glob(dataset + "/*"))
        test_img_dirs = test_img_dirs[start_img:end_img_exclusive]
        for test_img_dir in test_img_dirs:
            test_img = TestImages(test_img_dir)

            for amg_config in tqdm(sam_amg_configs):
                sam_proc = mp.Process(
                    target=sam_process,
                    args=(
                        sam_gen,
                        weights_path,
                        config,
                        amg_config,
                        test_img_dir,
                        dataset,
                        iou_thresh,
                        test_img,
                        queue,
                    ),
                )
                sam_proc.start()
                sam_proc.join()

                metrics = queue.get()
                metric_results.append(metrics)
                df = pd.DataFrame(metric_results)
                df.to_csv("Exp32/Sam2_hieraS2.1_8.csv", index=False)


if __name__ == "__main__":
    main()
