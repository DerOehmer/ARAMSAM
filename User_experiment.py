from sam_annotator.main import create_vis_options
from sam_annotator.app import App
from PyQt6.QtTest import QTest
import sys


def mock_open_img_load_file_dialog():
    return "/home/geink81/pythonstuff/CobScanws/earframes/0.jpg"


def mock_open_load_folder_dialog():
    # return "/home/geink81/pythonstuff/SequenceSAM-Annotator/ear_centers"
    # return "/home/geink81/pythonstuff/CobScanws/earframes/"
    return "/home/geink81/pythonstuff/CobScanws/earframesjonas/"


def mock_main():
    vis_options, default_vis_options, current_options = create_vis_options()
    ui_options = {
        "layout_settings_options": {
            "options": vis_options,
            "default": default_vis_options,
            "current": current_options,
        }
    }
    app = App(ui_options=ui_options)
    app.output_dir = "/home/geink81/pythonstuff/SequenceSAM-Annotator/raw_output"
    ui = app.ui
    ui.open_img_load_file_dialog = mock_open_img_load_file_dialog
    ui.open_load_folder_dialog = mock_open_load_folder_dialog
    # ui.manual_annotation_button.setDisabled(True)
    ui.run()
    QTest.qWait(50)

    ui.sam2_checkbox.setChecked(True)

    # set path
    load_folder_action = ui.menu_open.actions()[1]
    load_folder_action.trigger()

    sys.exit(app.application.exec())
    # app.exit()


if __name__ == "__main__":
    mock_main()
