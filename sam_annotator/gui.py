import numpy as np
import cv2

from PyQt6 import QtWidgets
from PyQt6 import QtGui
from screeninfo import get_monitors

from PyQt6.QtCore import Qt, QPoint, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QPixmap,
    QImage,
    QKeyEvent,
    QWheelEvent,
    QMouseEvent,
    QBrush,
    QColor,
    QCursor,
    QWindow,
)
from PyQt6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QLabel,
    QPushButton,
    QMessageBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QFrame,
    QGridLayout,
)


class UserInterface(QMainWindow):
    load_img_signal: pyqtSignal = pyqtSignal(int)
    load_img_folder_signal: pyqtSignal = pyqtSignal(int)
    mouse_position: pyqtSignal = pyqtSignal(tuple)
    preview_annotation_point_signal: pyqtSignal = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__(parent=None)

        monitors = get_monitors()
        width = 1920
        height = 1080
        for m in monitors:
            if m.width < width:
                width = m.width
            if m.height < height:
                height = m.height

        self.offset = 50
        self.resize(width, height)
        self.menu = self.menuBar().addMenu("Menu")
        self.menu_open = self.menuBar().addMenu("Open")
        self.menu_settings = self.menuBar().addMenu("Settings")
        self.menu.addAction("Exit", self.close)
        self.menu_open.addAction("Load Image", self.load_img)
        self.menu_open.addAction("Load Folder", self.load_img_folder)
        self.menu_settings.addAction("Change Layout", self.open_layout_settings_box)
        self.zoom_level = 0
        self.zoom_factor = 1.25

        self.construct_ui()
        self.showMaximized()

    def construct_ui(self):
        self.labelCoords = QLabel(self, text="Pixel Pos.")
        self.labelCoords.move(400, 20)
        self.labelCoords.show()

        vis_width, vis_height = self.calcluate_size_of_annotation_visualizers()

        self.annotation_visualizers: list[InteractiveGraphicsView] = []
        for i in range(2):
            for j in range(2):
                annotation_visualizer = InteractiveGraphicsView(
                    self, width=vis_width, height=vis_height
                )
                annotation_visualizer.setGeometry(
                    self.offset + i * vis_width,
                    self.offset + j * vis_height,
                    vis_width,
                    vis_height,
                )
                annotation_visualizer.show()
                annotation_visualizer.coordinatesChanged.connect(self.handleCoords)
                annotation_visualizer.mouseWheelScroll.connect(self.wheelEvent)
                annotation_visualizer.mouseMove.connect(self.childMouseMoveEvent)
                annotation_visualizer.mousePressSignal.connect(
                    self.childMousePressEvent
                )
                self.annotation_visualizers.append(annotation_visualizer)

        self.test_button = QPushButton(text="segment anything!", parent=self)
        self.test_button.move(20, 20)

        self.good_mask_button = QPushButton(text="good mask", parent=self)
        self.good_mask_button.move(120, 20)

        self.bad_mask_button = QPushButton(text="bad mask", parent=self)
        self.bad_mask_button.move(220, 20)

        self.back_button = QPushButton(text="undo OR back", parent=self)
        self.back_button.move(320, 20)

        self.manual_annotation_button = QPushButton(
            text="add mask manually", parent=self
        )
        self.manual_annotation_button.move(420, 20)

        self.draw_poly_button = QPushButton(text="draw polygon", parent=self)
        self.draw_poly_button.move(520, 20)

        self.next_img_button = QPushButton(text="next image", parent=self)
        self.next_img_button.move(620, 20)

    def calcluate_size_of_annotation_visualizers(self) -> tuple[int]:
        vis_width = int((self.width() - self.offset) / 2)
        vis_height = int((self.height() - self.offset) / 2)
        return vis_width, vis_height

    def load_img(self):
        self.load_img_signal.emit(1)

    def load_img_folder(self):
        self.load_img_folder_signal.emit(1)

    def open_layout_settings_box(self):
        options = ["image", "mask", "other mask"]
        self.options_window = OptionsWindow(options=options)
        self.options_window.show()

    def handleCoords(self, point: QPoint):
        if not point.isNull():
            self.labelCoords.setText(f"{point.x()}, {point.y()}")
            self.mouse_position.emit((point.x(), point.y()))
        else:
            self.labelCoords.clear()

    def keyPressEvent(self, event: QKeyEvent):
        if isinstance(event, QKeyEvent):
            key_text = event.text()
            print(f"Last Key Pressed: {key_text}")
        if event.text() == "n":
            self.good_mask_button.click()
        elif event.text() == "m":
            self.bad_mask_button.click()
        elif event.text() == "b":
            self.back_button.click()
        elif event.text() == "q":
            self.manual_annotation_button.click()
        elif event.text() == "a":
            self.preview_annotation_point_signal.emit(1)
        elif event.text() == "s":
            self.preview_annotation_point_signal.emit(0)
        elif event.text() == "d":
            self.preview_annotation_point_signal.emit(-1)

    def childMousePressEvent(self, event: QMouseEvent):
        if event.button().name == "RightButton":
            if event.modifiers().name == "NoModifier":
                # add foreground
                self.preview_annotation_point_signal.emit(1)
            if event.modifiers().name == "ControlModifier":
                # add background
                self.preview_annotation_point_signal.emit(0)

    def childMouseMoveEvent(self, event: QMouseEvent):
        scene_idx_under_mouse = self.get_annotation_visualizer_idx_under_mouse()
        if scene_idx_under_mouse is None:
            print("not on image")
            return

        viewport = (
            self.annotation_visualizers[scene_idx_under_mouse]
            .mapToScene(
                self.annotation_visualizers[scene_idx_under_mouse].viewport().geometry()
            )
            .boundingRect()
        )
        for idx, ann_viz in enumerate(self.annotation_visualizers):
            if idx == scene_idx_under_mouse:
                continue
            ann_viz.setSceneRect(viewport)

    def wheelEvent(self, event: QWheelEvent):
        scene_idx_under_mouse = self.get_annotation_visualizer_idx_under_mouse()
        if scene_idx_under_mouse is None:
            print("Zooming only on images")
            return

        if event.angleDelta().y() < 0:
            if self.zoom_level == 0:
                print("cant zoom further out")
                return
            elif self.zoom_level == 1:
                self.zoom_level = 0
                self.fit_annotation_visualizers_to_view()
                return
            else:
                self.zoom_level -= 1
                factor = 1 / self.zoom_factor

        if event.angleDelta().y() > 0:
            self.zoom_level += 1
            factor = self.zoom_factor

        # scale and save viewport and transformation of scene under mouse
        self.annotation_visualizers[scene_idx_under_mouse].scale(factor, factor)
        viewport = (
            self.annotation_visualizers[scene_idx_under_mouse]
            .mapToScene(
                self.annotation_visualizers[scene_idx_under_mouse].viewport().geometry()
            )
            .boundingRect()
        )
        # apply to others
        for idx, ann_viz in enumerate(self.annotation_visualizers):
            if idx == scene_idx_under_mouse:
                continue
            ann_viz.scale(factor, factor)
            ann_viz.setSceneRect(viewport)

    def get_annotation_visualizer_idx_under_mouse(self) -> int:
        scene_idx_under_mouse = None

        for idx, ann_viz in enumerate(self.annotation_visualizers):
            if ann_viz.underMouse():
                scene_idx_under_mouse = idx
        return scene_idx_under_mouse

    def fit_annotation_visualizers_to_view(self):
        for ann_viz in self.annotation_visualizers:
            ann_viz.fitInView()

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

    def open_load_folder_dialog(self):
        filepath = QFileDialog.getExistingDirectory(
            parent=self,
            caption="Open Folder",
            directory="${HOME}/data",
        )
        return filepath

    def run(self):
        self.show()

    def convert_ndarray_to_qimage(self, img: np.ndarray) -> QImage:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        pixmap = QPixmap.fromImage(q_img)
        self.annotation_visualizers[idx].set_pixmap(pixmap=pixmap)

    def resizeEvent(self, event):
        vis_width, vis_height = self.calcluate_size_of_annotation_visualizers()

        idx = 0
        for i in range(2):
            for j in range(2):
                self.annotation_visualizers[idx].setGeometry(
                    self.offset + i * vis_width,
                    self.offset + j * vis_height,
                    vis_width,
                    vis_height,
                )
                self.annotation_visualizers[idx].fitInView()
                idx += 1


class InteractiveGraphicsView(QGraphicsView):
    coordinatesChanged: pyqtSignal = pyqtSignal(QPoint)
    mouseWheelScroll: pyqtSignal = pyqtSignal(QWheelEvent)
    mouseMove: pyqtSignal = pyqtSignal(QMouseEvent)
    clicked: pyqtSignal = pyqtSignal(bool)
    mousePressSignal: pyqtSignal = pyqtSignal(QMouseEvent)

    def __init__(self, parent, width: int = 1024, height: int = 1024) -> None:
        super().__init__(parent=parent)

        self.width = width
        self.height = height
        self.zoom_val = 0
        self.mouse_down = False
        self.graphics_scene = QGraphicsScene(self)
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setShapeMode(QGraphicsPixmapItem.ShapeMode.BoundingRectShape)

        self.graphics_scene.addItem(self.pixmap_item)
        self.setScene(self.graphics_scene)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.Shape.NoFrame)

    def set_pixmap(self, pixmap: QPixmap):
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.pixmap_item.setPixmap(pixmap)

    def wheelEvent(self, event: QWheelEvent):
        # and overwrite superclass method to avoid scrolling on mouse wheel
        self.mouseWheelScroll.emit(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView()

    def leaveEvent(self, event):
        self.coordinatesChanged.emit(QPoint())
        super().leaveEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        self.updateCoordinates(event.position().toPoint())
        if self.mouse_down:
            self.mouseMove.emit(event)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        self.mouse_down = True
        self.mousePressSignal.emit(event)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.mouse_down = False
        super().mouseReleaseEvent(event)

    def updateCoordinates(self, pos=None):
        if self.pixmap_item.isUnderMouse():
            if pos is None:
                pos = self.mapFromGlobal(QCursor.pos())
            point = self.mapToScene(pos).toPoint()
        else:
            point = QPoint()
        self.coordinatesChanged.emit(point)

    def fitInView(self):
        rect = QRectF(self.pixmap_item.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(
                viewrect.width() / scenerect.width(),
                viewrect.height() / scenerect.height(),
            )
            self.scale(factor, factor)
            self.zoom_val = 0


class OptionsWindow(QMainWindow):
    def __init__(self, options: list[str]) -> None:
        super().__init__(parent=None)
        box_height = 50
        box_width = 100

        self.setWindowTitle("Options")
        self.resize(2 * box_width, 3 * box_height + 10)

        self.comboboxes = []

        for i in range(2):
            for j in range(2):
                combobox = QtWidgets.QComboBox(self)
                combobox.setGeometry(
                    i * box_width, j * box_height, box_width, box_height
                )
                [combobox.addItem(option) for option in options]
                self.comboboxes.append(combobox)
        self.ok_button = QPushButton(parent=self, text="OK")
        self.ok_button.setGeometry(0, box_height * 2 + 10, box_width, box_height)

        self.ok_button.setIcon(QtGui.QIcon.fromTheme("edit-undo"))
        self.abort_button = QPushButton(parent=self, text="Abort")
        self.abort_button.setGeometry(
            box_width, box_height * 2 + 10, box_width, box_height
        )
