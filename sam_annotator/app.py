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
        self.ui.load_img_signal.connect(self.load_img)

    def run(self) -> None:
        self.ui.run()
        sys.exit(self.application.exec())

    def load_img(self, _) -> None:
        print("loading new image")
        img_fpath = self.ui.open_img_load_file_dialog()
        self.annotator.create_new_annotation(Path(img_fpath))
        self.update_ui_imgs(img=True)

    def segment_anything(self):
        self.annotator.predict_with_sam()
        self.update_ui_imgs(masked_img=True, mask=True)

    def update_ui_imgs(
        self, img: bool = False, mask: bool = False, masked_img: bool = False
    ):
        if masked_img:
            self.ui.update_main_pix_map(idx=2, img=self.annotator.annotation.masked_img)
        if mask:
            self.ui.update_main_pix_map(
                idx=1, img=self.annotator.annotation.current_mask
            )
        if img:
            self.ui.update_main_pix_map(idx=0, img=self.annotator.annotation.img)

    def add_good_mask(self):
        done = self.annotator.good_mask()
        if done:
            self.ui.create_message_box(False, "All masks are done")
        self.update_ui_imgs(masked_img=True, mask=True)

    def add_bad_mask(self):
        done = self.annotator.bad_mask()
        if done:
            self.ui.create_message_box(False, "All masks are done")
        self.update_ui_imgs(masked_img=True, mask=True)
