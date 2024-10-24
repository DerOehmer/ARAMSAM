import pandas as pd
import glob
import numpy as np

from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from Backbone_evaluation_experiment import TestImages, SamTestInference


class AmgSamTestInference(SamTestInference):

    def __init__(self, sam_gen, weights_path, config, device="cuda"):
        super().__init__(sam_gen, weights_path, config, device)

        if sam_gen == 1:
            self.amg_predictor = SamAutomaticMaskGenerator(self.sam_model)
        elif sam_gen == 2:
            self.amg_predictor = SAM2AutomaticMaskGenerator(self.sam_model)
        else:
            raise ValueError("Invalid SAM generation")

    def get_sam_inference(self, test_img_objs: list[TestImages]):
        gt_masks = []
        pred_masks = []
        # get correct output
        for test_img_obj in test_img_objs:
            self.sam_predictor.set_image(test_img_obj.rgb)
            self.sam_predictor
            for gt_mask, center in test_img_obj:
                masks, scores, logits = self.amg_predictor.generate(test_img_obj)
                pred_mask = masks[0].astype(np.uint8) * 255
                pred_masks.append(pred_mask)
                gt_masks.append(gt_mask)
        # check if metric computation needs an update
        metrics = self.compute_metrics(pred_masks, gt_masks)
        print("N masks", len(pred_masks), len(gt_masks))
        return metrics


def main():
    datasets = ["BackboneExperimentData/MaizeUAV/"]
    metric_results = []
    sam = SamTestInference()  # TODO use correct version
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


if __name__ == "__main__":
    main()
