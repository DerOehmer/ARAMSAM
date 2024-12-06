from sam_annotator.main import create_vis_options
from sam_annotator.app import App
from sam_annotator.annotator import Annotator
from sam_annotator.mask_visualizations import MaskData
from PyQt6.QtTest import QTest
import sys
import numpy as np


def mock_set_sam():
    # Skip img embedding with Sam and Sam inference
    pass


def mock_propagate_good_masks():
    # Skip any propagation or tracking
    pass


def mock_open_load_folder_dialog():
    return "ExperimentData/IndicatedPolygonPositionImages"


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


def mock_main(tutorial=True):
    vis_options, default_vis_options, current_options = create_vis_options()
    ui_options = {
        "layout_settings_options": {
            "options": vis_options,
            "default": default_vis_options,
            "current": current_options,
        }
    }
    app = App(ui_options=ui_options, experiment_mode="polygon")
    if tutorial:
        app.tutorial_flag = True
    app.set_sam = mock_set_sam
    app.propagate_good_masks = mock_propagate_good_masks
    app.change_output_dir("output")
    ui = app.ui
    ui.open_load_folder_dialog = mock_open_load_folder_dialog
    ui.manual_annotation_button.setDisabled(True)
    ui.menu.setDisabled(True)
    ui.menu_open.setDisabled(True)
    ui.menu_settings.setDisabled(True)

    ui.run()
    QTest.qWait(50)

    # set path
    load_folder_action = ui.menu_open.actions()[1]
    load_folder_action.trigger()
    QTest.qWait(50)
    set_vis(app.annotator)
    app.update_ui_imgs()
    ui.draw_poly_button.click()
    ui.draw_poly_button.setDisabled(True)

    sys.exit(app.application.exec())


if __name__ == "__main__":
    mock_main()
