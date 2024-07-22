import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication

from sam_annotator.gui import UserInterface
from sam_annotator.annotator import Annotator


class App:
    """
    Contains the UI with main thread and subprocesses for the annotation
    """

    def __init__(self) -> None:
        self.application = QApplication([])
        self.ui = UserInterface()
        self.annotator = Annotator()

        self.ui.test_button.clicked.connect(self.segment_anything)
        self.ui.good_mask_button.clicked.connect(self.add_good_mask)
        self.ui.bad_mask_button.clicked.connect(self.add_bad_mask)
        self.ui.back_button.clicked.connect(self.last_mask)

        self.ui.load_img_signal.connect(self.load_img)

    def run(self) -> None:
        self.ui.run()
        sys.exit(self.application.exec())

    def load_img(self, _) -> None:
        print("loading new image")
        img_fpath = self.ui.open_img_load_file_dialog()
        self.annotator.create_new_annotation(Path(img_fpath))
        self.update_ui_imgs()

    def segment_anything(self):
        self.annotator.predict_with_sam()
        self.update_ui_imgs()

    def update_ui_imgs(self):
        mviss = self.annotator.annotation.mask_visualizations
        fields = ["img", "mask", "masked_img_cnt", "mask_collection_cnt"]
        if len(fields) > 4:
            print(
                f"Too many fields selected for visualization ({len(fields)}) expected 4"
            )
        for idx, field in enumerate(fields):
            if getattr(mviss, field) is not None:
                self.ui.update_main_pix_map(idx=idx, img=getattr(mviss, field))

    def add_good_mask(self):
        done = self.annotator.good_mask()
        if done:
            self.ui.create_message_box(False, "All masks are done")
        # TODO: instead of done receive coordinates of center of next mask
        # TODO: use coordinates to center the view on mask
        self.update_ui_imgs()

    def add_bad_mask(self):
        done = self.annotator.bad_mask()
        if done:
            self.ui.create_message_box(False, "All masks are done")
        # TODO: instead of done receive coordinates of center of next mask
        # TODO: use coordinates to center the view on mask
        self.update_ui_imgs()

    def last_mask(self):
        done = self.annotator.update_mask_idx(self.annotator.mask_idx - 1)
        # TODO: error handling if mask idx is out of bounds
        self.update_ui_imgs()
