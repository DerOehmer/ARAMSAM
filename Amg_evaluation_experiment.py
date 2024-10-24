import pandas as pd
import glob
import itertools
import numpy as np

from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from Backbone_evaluation_experiment import TestImages, SamTestInference, SamFlavors


class AmgSamTestInference(SamTestInference):

    def __init__(self, sam_gen, weights_path, config, amg_config, device="cuda"):
        super().__init__(sam_gen, weights_path, config, device)

        amg_config["model"] = self.sam_model
        if sam_gen == 1:
            self.amg_predictor = SamAutomaticMaskGenerator(**amg_config)
        elif sam_gen == 2:
            self.amg_predictor = SAM2AutomaticMaskGenerator(**amg_config)
        else:
            raise ValueError("Invalid SAM generation")

    def get_sam_inference(self, test_img_objs: list[TestImages]):
        gt_masks = []
        pred_masks = []
        # TODO: adapt to incoming data and postprocess to reuse metrics of parent
        for test_img_obj in test_img_objs:
            self.sam_predictor.set_image(test_img_obj.rgb)
            self.sam_predictor
            masks = self.amg_predictor.generate(test_img_obj.rgb)
            pred_mask = masks[0].astype(np.uint8) * 255
            pred_masks.append(pred_mask)
            gt_masks.append(test_img_obj.gt_mask)
        # check if metric computation needs an update
        metrics = self.compute_metrics(pred_masks, gt_masks)
        print("N masks", len(pred_masks), len(gt_masks))
        return metrics


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
    datasets = ["BackboneExperimentData/MaizeUAV/"]
    metric_results = []
    sam_flavors = SamFlavors()
    sam_gen, weights_path, config = next(iter(sam_flavors))
    sam_amg_configs = create_sam_amg_configs()
    for amg_config in sam_amg_configs:
        sam = AmgSamTestInference(sam_gen, weights_path, config, amg_config)
        for dataset in datasets:
            test_imgs = []

            for test_img_dir in glob.glob(dataset + "/*"):
                test_img = TestImages(test_img_dir)
                test_imgs.append(test_img)

            metrics = sam.get_sam_inference(test_imgs)
            metrics["Dataset"] = dataset
            print(metrics)
            metric_results.append(metrics)
        del sam

    df = pd.DataFrame(metric_results)
    df.to_csv("MaizeUAV_results.csv", index=False)


if __name__ == "__main__":
    main()
