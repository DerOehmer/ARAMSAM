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
    QMessageBox,
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
        self.viz_labels: list[QLabel] = []
        self.create_pixmaps(img_shape=img_shape, offset=offset)

        # self.tabs = QTabWidget(parent=self)
        # self.sequence_tab = QWidget()
        # self.img_label = QLabel(self)
        # self.img_label.move(offset, offset)
        # self.img_label.resize(img_shape[0], img_shape[1])
        # self.img_label.setPixmap(QPixmap.fromImage(self.img))
        # self.img_label.show()

        # self.mask_label = QLabel(self)
        # self.mask_label.move(offset + img_shape[0], offset)
        # self.mask_label.resize(img_shape[0], img_shape[1])
        # self.mask_label.setPixmap(QPixmap.fromImage(self.img))
        # self.mask_label.show()

        # self.masked_img_label = QLabel(self)
        # self.masked_img_label.move(offset + 2 * img_shape[0], offset)
        # self.masked_img_label.resize(img_shape[0], img_shape[1])
        # self.masked_img_label.setPixmap(QPixmap.fromImage(self.img))
        # self.masked_img_label.show()

        self.test_button = QPushButton(text="segment anything!", parent=self)
        self.test_button.move(20, 20)

        self.good_mask_button = QPushButton(text="good mask", parent=self)
        self.good_mask_button.move(120, 20)

        self.bad_mask_button = QPushButton(text="bad mask", parent=self)
        self.bad_mask_button.move(220, 20)

    def create_pixmaps(self, img_shape: tuple[int], offset: int):
        for i in range(3):
            label = QLabel(self)
            label.move(offset + i * img_shape[0], offset)
            label.resize(img_shape[0], img_shape[1])
            label.setPixmap(QPixmap.fromImage(self.img))
            label.show()
            self.viz_labels.append(label)

    def keyPressEvent(self, event: QKeyEvent):
        if isinstance(event, QKeyEvent):
            key_text = event.text()
            print(f"Last Key Pressed: {key_text}")
        if event.text() == "n":
            self.good_mask_button.click()
        elif event.text() == "m":
            self.bad_mask_button.click()

    def create_message_box(self, crticial: bool = False, text: str = ""):
        self.msg_box = QMessageBox(self)
        icon = QMessageBox.Icon.Critical if crticial else QMessageBox.Icon.Information
        self.msg_box.setIcon(icon)
        self.msg_box.setText(text)
        self.msg_box.show()

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

    def update_main_pix_map(self, idx: int, img: np.ndarray):
        q_img = self.convert_ndarray_to_qimage(img=img)
        self.viz_labels[idx].setPixmap(QPixmap.fromImage(q_img))
