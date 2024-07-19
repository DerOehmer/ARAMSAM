import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication

from sam_annotator.gui import UserInterface
from sam_annotator.annotator import Annotator
from sam_annotator.mask_visualizations import MaskVisualizationData


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
        for i in range(4):
            if getattr(mviss, fields[i]) is not None:
                self.ui.update_main_pix_map(idx=i, img=getattr(mviss, fields[i]))

    def add_good_mask(self):
        done = self.annotator.good_mask()
        if done:
            self.ui.create_message_box(False, "All masks are done")
        self.update_ui_imgs()

    def add_bad_mask(self):
        done = self.annotator.bad_mask()
        if done:
            self.ui.create_message_box(False, "All masks are done")
        self.update_ui_imgs()
