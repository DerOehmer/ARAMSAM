import Polygon_user_experiment
import SAM_user_experiment
import numpy as np
import glob
import sys
import os
import time
from multiprocessing import Process
import pandas as pd


def experiment_process(
    exp_steps: dict,
    config_order: np.ndarray,
    i: int,
    polygon_paths: list[str],
    structured_ear_dirs: list[str],
    user_idx: int,
):
    confg = config_order[i]
    exp_mode = exp_steps["exp_mode"][confg]
    sam_gen = exp_steps["sam_gen"][confg]
    img_idx = exp_steps["img_idx"][confg]
    print(exp_mode, sam_gen, img_idx)
    progress = (i + 1, len(config_order))
    if exp_mode == "polygon":
        Polygon_user_experiment.mock_main(
            [polygon_paths[img_idx]],
            tutorial=False,
            user_id=user_idx,
            progress=progress,
        )
    elif exp_mode == "structured":
        SAM_user_experiment.mock_main(
            structured_ear_dirs[img_idx],
            tutorial=False,
            sam_gen=sam_gen,
            user_id=user_idx,
            progress=progress,
        )


def main():
    exp_start = time.time()

    exp_steps = {
        "exp_mode": [
            "polygon",
            "polygon",
            "polygon",
            "structured",
            "structured",
            "structured",
            "structured",
            "structured",
            "structured",
        ],
        "sam_gen": [None, None, None, 1, 1, 1, 2, 2, 2],
        "img_idx": [0, 1, 2, 0, 1, 2, 0, 1, 2],
    }
    config_order = np.random.choice(9, 9, replace=False)
    polygon_paths = glob.glob(
        "ExperimentData/IndicatedPolygonPositionImages/Experiment/*"
    )
    structured_ear_dirs = glob.glob("ExperimentData/EarImgPairs/Experiment/*")
    user_idx = 0
    while os.path.exists(f"output/User_{user_idx}"):
        user_idx += 1
    output_dir = f"output/User_{user_idx}"
    os.makedirs(output_dir)

    for i in range(len(config_order)):
        p = Process(
            target=experiment_process,
            args=(
                exp_steps,
                config_order,
                i,
                polygon_paths,
                structured_ear_dirs,
                user_idx,
            ),
        )
        p.start()
        p.join()

    print(f"Experiment took {time.time() - exp_start:.2f} seconds")
    with open(f"{output_dir}/experiment_time.txt", "w") as f:
        f.write(f"{time.time() - exp_start:.2f} seconds")

    exp_config_df = pd.DataFrame(exp_steps)
    sorted_df = exp_config_df.loc[config_order].reset_index(drop=True)
    sorted_df.to_csv(f"{output_dir}/experiment_config.csv", index=True)


if __name__ == "__main__":
    sys.exit(main())
