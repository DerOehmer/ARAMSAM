import sys
import os

sys.path.append(os.path.abspath("."))
from aramsam_annotator.main import create_vis_options
from aramsam_annotator.app import App
from aramsam_annotator.annotator import Annotator
from aramsam_annotator.mask_visualizations import MaskData
from PyQt6.QtTest import QTest
import sys
import numpy as np
import glob


def mock_set_sam():
    # Skip img embedding with Sam and Sam inference
    pass


def mock_propagate_good_masks():
    # Skip any propagation or tracking
    pass


def mock_open_load_folder_dialog():
    # return "UserExperiment/TutorialImages"
    return ""


def set_vis(annotator: Annotator) -> tuple[bool]:
    self = annotator
    self.annotation.mask_visualizations.masked_img = self.annotation.img.copy()
    mock_mask = (
        np.ones(
            (self.annotation.img.shape[0], self.annotation.img.shape[1]), dtype=np.uint8
        )
        * 255
    )
    self.annotation.add_masks(
        [MaskData(mid=-1, mask=mock_mask, origin="Polygon_drawing")]  # mock mask
    )

    self.update_mask_idx()

    self.update_collections(self.annotation)


def mock_main(
    polygon_img_paths: str,
    tutorial: bool = False,
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
        ui_options=ui_options, experiment_mode="polygon", experiment_progress=progress
    )
    if tutorial:
        app.tutorial_flag = True
    app.set_sam = mock_set_sam
    app.propagate_good_masks = mock_propagate_good_masks
    app.change_output_dir(f"output/User_{user_id}")
    ui = app.ui
    ui.open_load_folder_dialog = mock_open_load_folder_dialog
    ui.manual_annotation_button.setDisabled(True)
    ui.menu.setDisabled(True)
    ui.menu_open.setDisabled(True)
    ui.menu_settings.setDisabled(True)

    ui.run()
    QTest.qWait(50)

    app.img_fnames = polygon_img_paths
    load_folder_action = ui.menu_open.actions()[1]
    load_folder_action.trigger()
    set_vis(app.annotator)
    app.update_ui_imgs()
    ui.draw_poly_button.click()
    ui.draw_poly_button.setDisabled(True)
    return app.application.exec()


if __name__ == "__main__":
    sys.exit(mock_main("ExperimentData/IndicatedPolygonPositionImages/Experiment"))
