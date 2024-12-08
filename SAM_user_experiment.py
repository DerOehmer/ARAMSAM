from sam_annotator.main import create_vis_options
from natsort import natsorted
from sam_annotator.app import App
from PyQt6.QtTest import QTest
import sys
import glob


"""def mock_open_load_folder_dialog():
    return img_pair_folder_p"""


def mock_open_load_folder_dialog():
    # return "UserExperiment/TutorialImages"
    return ""


def update_mock_path(new_path):
    global img_pair_folder_p  # Use global to modify the img_pair_folder_p variable
    img_pair_folder_p = new_path


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
    app.img_fnames = glob.glob(f"{img_pair_folder_p}/*")
    load_folder_action = ui.menu_open.actions()[1]
    load_folder_action.trigger()
    return app.application.exec()


if __name__ == "__main__":
    root_p = "ExperimentData/EarImgPairs/Tutorial"
    for img_pair_folder_p in natsorted(glob.glob(f"{root_p}/*")):
        # update_mock_path(img_pair_folder_p)
        sys.exit(mock_main(img_pair_folder_p, tutorial=False))
