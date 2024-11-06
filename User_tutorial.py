from sam_annotator.main import create_vis_options
from sam_annotator.app import App
from PyQt6.QtTest import QTest


def mock_open_load_folder_dialog():
    return "UserExperiment/TutorialImages"


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
    app.output_dir = "/home/geink81/pythonstuff/SequenceSAM-Annotator/output"
    app.sam_gen = 1
    ui = app.ui
    """ui.create_info_box(
        False,
        "Loading, please wait...",
        wait_for_user=False,
    )"""
    ui.open_load_folder_dialog = mock_open_load_folder_dialog
    ui.run()
    QTest.qWait(50)
    # set path
    load_folder_action = ui.menu_open.actions()[1]  # 0 is for loading image
    load_folder_action.trigger()

    ui.menu.setDisabled(True)
    ui.menu_open.setDisabled(True)
    ui.menu_settings.setDisabled(True)
    ui.manual_annotation_button.setDisabled(True)
    ui.draw_poly_button.setDisabled(True)

    ui.performing_embedding_label.setText(f"Step 1/3: Select the good proposed masks")
    ui.create_info_box(
        False,
        "Welcome to the ARAMSAM tutorial!",
        wait_for_user=True,
    )
    app.application.exec()


if __name__ == "__main__":
    mock_main()
