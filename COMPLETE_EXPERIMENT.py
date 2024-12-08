import Polygon_user_experiment
import SAM_user_experiment
import numpy as np
import glob
import sys
import os
import time


def random_experiment_order():
    pass


def main():
    exp_start = time.time()

    exp_steps = {
        "exp_mode": [
            "polygon",
            "polygon",
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
        "sam_gen": [None, None, None, None, None, 1, 1, 1, 2, 2, 2],
        "img_idx": [0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2],
    }
    config_order = np.random.choice(11, 11, replace=False)
    polygon_paths = glob.glob(
        "ExperimentData/IndicatedPolygonPositionImages/Experiment/*"
    )
    structured_ear_dirs = glob.glob("ExperimentData/EarImgPairs/Experiment/*")
    user_idx = 0
    while os.path.exists(f"output/User_{user_idx}"):
        user_idx += 1
    output_dir = f"output/User_{user_idx}"
    os.makedirs(output_dir)

    for i, confg in enumerate(config_order):
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
        time.sleep(1)

    print(f"Experiment took {time.time() - exp_start:.2f} seconds")
    with open(f"{output_dir}/experiment_time.txt", "w") as f:
        f.write(f"{time.time() - exp_start:.2f} seconds")


if __name__ == "__main__":
    sys.exit(main())
