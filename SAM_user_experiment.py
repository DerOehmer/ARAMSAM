from sam_annotator.main import create_vis_options
from natsort import natsorted
from sam_annotator.app import App
from PyQt6.QtTest import QTest
import sys
import glob
import os


def extract_img_pos(img_path: str) -> int:
    img_name = os.path.basename(img_path)
    img_name = img_name.split(".")[0]
    img_pos = img_name.split("_low_")[-1]
    return img_pos


def mock_open_load_folder_dialog():
    return ""


def mock_main(
    img_pair_folder_p: str,
    tutorial: bool = False,
    sam_gen: int = 2,
    user_id: int = 0,
    progress: tuple[int, int] = None,
):
    vis_options, default_vis_options, current_options = create_vis_options()
    ui_options = {
        "layout_settings_options": {
            "options": vis_options,
            "default": default_vis_options,
            "current": current_options,
        }
    }
    app = App(
        ui_options=ui_options,
        experiment_mode="structured",
        experiment_progress=progress,
    )
    if tutorial:
        app.tutorial_flag = True
    app.output_dir = f"output/User_{user_id}"
    app.sam_gen = sam_gen
    ui = app.ui
    ui.create_basic_loading_window()
    ui.open_load_folder_dialog = mock_open_load_folder_dialog
    ui.run()
    QTest.qWait(50)

    ui.menu.setDisabled(True)
    ui.menu_open.setDisabled(True)
    ui.menu_settings.setDisabled(True)
    ui.manual_annotation_button.setDisabled(True)
    ui.draw_poly_button.setDisabled(True)

    ui.performing_embedding_label.setText(f"Step 1/3: Select the good proposed masks")
    img_paths = glob.glob(f"{img_pair_folder_p}/*")
    sorted_img_paths = sorted(img_paths, key=extract_img_pos)
    app.img_fnames = sorted_img_paths
    load_folder_action = ui.menu_open.actions()[1]
    load_folder_action.trigger()
    return app.application.exec()


if __name__ == "__main__":
    root_p = "ExperimentData/EarImgPairs/Tutorial"
    for img_pair_folder_p in natsorted(glob.glob(f"{root_p}/*")):
        # update_mock_path(img_pair_folder_p)
        sys.exit(mock_main(img_pair_folder_p, tutorial=False))
