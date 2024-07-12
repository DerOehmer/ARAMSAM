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

        self.ui.menu.addAction("Load Image", self.load_img)
        self.ui.test_button.clicked.connect(self.segment_anything)
        self.ui.good_mask_button.clicked.connect(self.add_good_mask)

    def run(self) -> None:
        self.ui.run()
        sys.exit(self.application.exec())

    def load_img(self) -> None:
        print("loading new image")
        img_fpath = self.ui.open_img_load_file_dialog()
        self.annotator.create_new_annotation(Path(img_fpath))
        self.ui.update_img(img=self.annotator.annotation.img)

    def segment_anything(self):
        self.annotator.annotation = self.annotator.predict_with_sam(
            self.annotator.annotation
        )
        self.ui.update_masked_img(masked_img=self.annotator.annotation.masked_img)

    def add_good_mask(self):
        # self.annotator.good_mask()
        self.ui.update_masked_img(masked_img=self.annotator.annotation.masked_img)
        self.ui.update_mask(mask=self.annotator.annotation.mask_collection)
