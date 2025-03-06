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


def mock_set_sam():
    # Skip img embedding with Sam and Sam inference
    pass


def mock_propagate_good_masks():
    # Skip any propagation or tracking
    pass


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
        [MaskData(mid=-1, mask=mock_mask, origin="Sam1_proposed")]  # mock mask
    )

    self.update_mask_idx()

    self.update_collections(self.annotation)


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
    app.experiment_mode = "polygon"
    app.set_sam = mock_set_sam
    app.propagate_good_masks = mock_propagate_good_masks
    ui = app.ui
    ui.manual_annotation_button.setDisabled(True)

    ui.run()
    QTest.qWait(50)

    # set path
    load_folder_action = ui.menu_open.actions()[0]
    load_folder_action.trigger()
    QTest.qWait(50)
    set_vis(app.annotator)
    app.update_ui_imgs()
    app.annotator.toggle_polygon_drawing()

    sys.exit(app.application.exec())


if __name__ == "__main__":
    mock_main()
