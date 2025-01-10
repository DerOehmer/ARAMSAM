import pandas as pd
import os
import json
import numpy as np


def f_beta(row, beta, epsilon=1e-7):
    precision = row["precision"]
    recall = row["recall"]
    return (
        (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + epsilon)
    )


if __name__ == "__main__":
    BETA = 2
    PATH = "ExperimentData/AmgEvaluationData/Sam2_hieraS2.1.csv"
    # PATH = "ExperimentData/AmgEvaluationData/Sam1_VitH.csv"

    amg_setting_cols = {
        "points_per_side": int,
        "points_per_batch": int,
        "pred_iou_thresh": float,
        "stability_score_thresh": float,
        "stability_score_offset": float,
        "box_nms_thresh": float,
        "crop_n_layers": int,
        "crop_nms_thresh": float,
        "crop_n_points_downscale_factor": int,
    }

    sam_type = os.path.basename(PATH).split(".csv")[0]
    config_json_path = f"ExperimentData/AmgEvaluationData/{sam_type}_best_config.json"
    amg_results_path = f"ExperimentData/AmgEvaluationData/{sam_type}_amg_results.csv"
    best_config_results_path = (
        f"ExperimentData/AmgEvaluationData/{sam_type}_best_config_results.csv"
    )

    df = pd.read_csv(PATH)

    df["f_beta"] = df.apply(f_beta, axis=1, beta=BETA)

    mean_fbeta_per_config = (
        df.groupby(list(amg_setting_cols.keys()))
        .agg(
            {
                "recall": "mean",
                "precision": "mean",
                "predictions": "mean",
                "ground_truth": "mean",
                "TP": "mean",
                "FP": "mean",
                "FN": "mean",
                "mean_iou_of_tp": "mean",
                "std_iou_of_tp": "mean",
                "f_beta": "mean",
            }
        )
        .reset_index()
    )

    best_config_ser = mean_fbeta_per_config.sort_values("f_beta", ascending=False).iloc[
        [0]
    ]

    best_config_dict = best_config_ser.loc[:, list(amg_setting_cols.keys())].to_dict(
        orient="records"
    )[0]
    conditions = [(df[col] == val) for col, val in best_config_dict.items()]
    best_config_results = df[np.logical_and.reduce(conditions)]

    with open(config_json_path, "w") as f:
        json.dump(best_config_dict, f, indent=4)

    mean_fbeta_per_config.to_csv(amg_results_path, index=False)
    best_config_results.to_csv(best_config_results_path, index=False)
