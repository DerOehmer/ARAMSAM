from sam_annotator.main import create_vis_options
from sam_annotator.app import App
from PyQt6.QtTest import QTest
import sys


def mock_open_load_folder_dialog():
    # return "UserExperiment/TutorialImages"
    return ""


def mock_main():
    vis_options, default_vis_options, current_options = create_vis_options()
    ui_options = {
        "layout_settings_options": {
            "options": vis_options,
            "default": default_vis_options,
            "current": current_options,
        }
    }
    app = App(ui_options=ui_options, experiment_mode="tutorial")
    app.output_dir = "./output"
    app.sam_gen = 1
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

    # set path
    app.img_fnames = [
        "ExperimentData/TutorialImages/39320223532020_low_64.jpg",
        "ExperimentData/TutorialImages/39320223511025_low_192.jpg",
    ]
    load_folder_action = ui.menu_open.actions()[1]  # 1 is for loading folder
    # sys.exit(app.application.exec())
    sys.exit(load_folder_action.trigger())

    # app.select_next_img()


if __name__ == "__main__":
    mock_main()
