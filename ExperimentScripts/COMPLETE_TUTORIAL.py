import User_tutorial
import Polygon_user_experiment
import SAM_user_experiment
import sys
import glob


def main():
    User_tutorial.mock_main()
    Polygon_user_experiment.mock_main(
        glob.glob("ExperimentData/IndicatedPolygonPositionImages/Tutorial/*"),
        tutorial=True,
    )
    for img_pair_folder_p in glob.glob("ExperimentData/EarImgPairs/Tutorial/*"):
        SAM_user_experiment.mock_main(img_pair_folder_p, tutorial=True)


if __name__ == "__main__":
    sys.exit(main())
