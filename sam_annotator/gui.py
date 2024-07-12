import numpy as np
import PyQt6 as Qt

from PyQt6.QtGui import QPixmap, QImage, QKeyEvent
from PyQt6.QtWidgets import (
    QWidget,
    QMainWindow,
    QFileDialog,
    QLabel,
    QPushButton,
    QTabWidget,
)


class UserInterface(QMainWindow):
    def __init__(self) -> None:
        super().__init__(parent=None)
        self.left = 10
        self.top = 10
        img_shape = (512, 512)
        offset = 50
        self.resize(img_shape[0] * 3 + offset, img_shape[1] + offset)
        self.menu = self.menuBar().addMenu("&Menu")
        self.menu.addAction("&Exit", self.close)

        self.construct_ui(img_shape=img_shape, offset=offset)

    def construct_ui(self, img_shape: tuple[int], offset: int):
        # img, mask and masked img display labels
        img = np.zeros((img_shape[0], img_shape[1], 3), dtype="uint8")
        self.img = QImage(img, img_shape[0], img_shape[1], QImage.Format.Format_RGB888)

        # self.tabs = QTabWidget(parent=self)
        # self.sequence_tab = QWidget()

        self.img_label = QLabel(self)
        self.img_label.move(offset, offset)
        self.img_label.resize(img_shape[0], img_shape[1])
        self.img_label.setPixmap(QPixmap.fromImage(self.img))
        self.img_label.show()

        self.mask_label = QLabel(self)
        self.mask_label.move(offset + img_shape[0], offset)
        self.mask_label.resize(img_shape[0], img_shape[1])
        self.mask_label.setPixmap(QPixmap.fromImage(self.img))
        self.mask_label.show()

        self.masked_img_label = QLabel(self)
        self.masked_img_label.move(offset + 2 * img_shape[0], offset)
        self.masked_img_label.resize(img_shape[0], img_shape[1])
        self.masked_img_label.setPixmap(QPixmap.fromImage(self.img))
        self.masked_img_label.show()

        self.test_button = QPushButton(text="segment anything!", parent=self)
        self.test_button.move(20, 20)

        self.good_mask_button = QPushButton(text="good mask", parent=self)
        self.good_mask_button.move(120, 20)

        self.bad_mask_button = QPushButton(text="bad mask", parent=self)
        self.bad_mask_button.move(220, 20)

    def keyPressEvent(self, event: QKeyEvent):
        if isinstance(event, QKeyEvent):
            key_text = event.text()
            print(f"Last Key Pressed: {key_text}")
        if event.key() == "n":
            self.good_mask_button.click()
            pass

        elif event.key() == "m":
            self.bad_mask_button.click()
            pass

    def open_img_load_file_dialog(self):
        filepath = QFileDialog.getOpenFileName(
            parent=self,
            caption="Open File",
            directory="${HOME}/data",
            filter="All Files (*);; PNG Files (*.png)",
        )
        return filepath[0]

    def run(self):
        self.show()

    def convert_ndarray_to_qimage(self, img: np.ndarray) -> QImage:
        h, w, ch = img.shape
        bytes_per_line = ch * w
        return QImage(
            img.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

    def update_img(self, img: np.ndarray):
        q_img = self.convert_ndarray_to_qimage(img=img)
        self.img_label.setPixmap(QPixmap.fromImage(q_img))

    def update_mask(self, mask: np.ndarray):
        q_mask = self.convert_ndarray_to_qimage(img=mask)
        self.mask_label.setPixmap(QPixmap.fromImage(q_mask))

    def update_masked_img(self, masked_img: np.ndarray):
        q_masked_img = self.convert_ndarray_to_qimage(img=masked_img)
        self.masked_img_label.setPixmap(QPixmap.fromImage(q_masked_img))
