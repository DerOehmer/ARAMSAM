import numpy as np
import cv2

from PyQt6 import QtWidgets
from PyQt6 import QtGui
from screeninfo import get_monitors
import math
import json
import os

from PyQt6.QtCore import Qt, QPoint, QRectF, pyqtSignal, QRect, QPointF
from PyQt6.QtGui import (
    QPixmap,
    QImage,
    QKeyEvent,
    QWheelEvent,
    QMouseEvent,
    QBrush,
    QColor,
    QCursor,
    QFont,
    QPainter,
    QPainterPath,
    QPolygonF,
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
    QCheckBox,
    QProgressDialog,
    QApplication,
    QDialog,
)


class UserInterface(QMainWindow):
    load_img_signal: pyqtSignal = pyqtSignal(int)
    load_img_folder_signal: pyqtSignal = pyqtSignal(int)
    mouse_position: pyqtSignal = pyqtSignal(tuple)
    preview_annotation_point_signal: pyqtSignal = pyqtSignal(int)
    layout_options_signal: pyqtSignal = pyqtSignal(list)
    output_dir_signal: pyqtSignal = pyqtSignal(str)
    sam_path_signal: pyqtSignal = pyqtSignal(str)
    save_signal: pyqtSignal = pyqtSignal(int)

    def __init__(self, ui_options: dict = None, experiment_mode: str = None) -> None:
        super().__init__(parent=None)

        self.ui_options = ui_options
        self.experiment_mode = experiment_mode

        monitors = get_monitors()
        width = 1920
        height = 1080
        for m in monitors:
            if m.width < width:
                width = m.width
            if m.height < height:
                height = m.height

        self.height_offset = 70
        self.buttons_min_width = 100
        self.buttons_spacing = 10
        self.resize(width, height)
        self.menu = self.menuBar().addMenu("Menu")
        self.menu_open = self.menuBar().addMenu("Open")
        self.menu_settings = self.menuBar().addMenu("Settings")
        self.menu.addAction("Save", self.save)
        self.menu.addAction("Save As", self.save_as)
        self.menu.addAction("Exit", self.exit)
        self.menu_open.addAction("Load Image", self.load_img)
        self.menu_open.addAction("Load Folder", self.load_img_folder)
        self.menu_settings.addAction("Change Layout", self.open_layout_settings_box)
        self.menu_settings.addAction(
            "Set Ouput Directory", self.open_ouput_dir_selection
        )
        self.menu_settings.addAction("Set SAM Model", self._open_sam_model_selection)
        self.zoom_level = 0
        self.zoom_factor = 1.25
        self.current_viewport = None
        self.last_panned_img = None

        self.loading_window = None
        self.basic_loading_window = None

        self.construct_ui()
        self.showMaximized()

    def construct_ui(self):
        vis_width, vis_height = self.calcluate_size_of_annotation_visualizers()

        self.annotation_visualizers: list[InteractiveGraphicsView] = []
        for i in range(2):
            for j in range(2):
                annotation_visualizer = InteractiveGraphicsView(
                    self,
                    ann_viz_id=i * 2 + j,
                    width=vis_width,
                    height=vis_height,
                )
                annotation_visualizer.setGeometry(
                    i * vis_width,
                    self.height_offset + j * vis_height,
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
                annotation_visualizer.new_scene_initialized.connect(
                    self.new_pixmap_displayed_in_annotation_vizualizers_callback
                )
                self.annotation_visualizers.append(annotation_visualizer)

        self.fit_annotation_visualizers_to_view()

        self.good_mask_button = QPushButton(text="good mask (n)", parent=self)
        self.good_mask_button.move(self.buttons_spacing, int(self.height_offset / 2))
        self.good_mask_button.setMinimumWidth(self.buttons_min_width)

        self.bad_mask_button = QPushButton(text="bad mask (m)", parent=self)
        self.bad_mask_button.move(
            2 * self.buttons_spacing + 1 * self.buttons_min_width,
            int(self.height_offset / 2),
        )
        self.bad_mask_button.setMinimumWidth(self.buttons_min_width)

        self.back_button = QPushButton(text="back (b)", parent=self)
        self.back_button.move(
            3 * self.buttons_spacing + 2 * self.buttons_min_width,
            int(self.height_offset / 2),
        )
        self.back_button.setMinimumWidth(self.buttons_min_width)

        self.manual_annotation_button = QPushButton(text="manual", parent=self)
        self.manual_annotation_button.setCheckable(True)
        self.manual_annotation_button.move(
            4 * self.buttons_spacing + 3 * self.buttons_min_width,
            int(self.height_offset / 2),
        )
        self.manual_annotation_button.setStyleSheet(
            """
            QPushButton {
                background-color: #455364;   /* Set the background color */
                color: white;                /* Set the text color */
            }
            QPushButton:checked {
                background-color: #0c6af7;  /* Checked state */
                color: white;
            
            }
            QPushButton:hover {
                background-color: #54687A;   /* Change background on hover */
            }
            QPushButton:disabled {
                color: #788D9C;   /* Disabled state */
            }

            """
        )
        self.manual_annotation_button.setMinimumWidth(self.buttons_min_width)

        self.draw_poly_button = QPushButton(text="draw polygon", parent=self)
        self.draw_poly_button.setCheckable(True)
        self.draw_poly_button.move(
            5 * self.buttons_spacing + 4 * self.buttons_min_width,
            int(self.height_offset / 2),
        )
        self.draw_poly_button.setStyleSheet(
            """
            QPushButton {
                background-color: #455364;   /* Set the background color */
                color: white;                /* Set the text color */
            }
            QPushButton:checked {
                background-color: #0c6af7;  /* Checked state */
                color: white;
            
            }
            QPushButton:hover {
                background-color: #54687A;   /* Change background on hover */
            }
            QPushButton:disabled {
                color: #788D9C;   /* Disabled state */
            }

            """
        )
        self.draw_poly_button.setMinimumWidth(self.buttons_min_width)

        self.delete_button = QPushButton(text="delete", parent=self)
        self.delete_button.setCheckable(True)
        self.delete_button.move(
            6 * self.buttons_spacing + 5 * self.buttons_min_width,
            int(self.height_offset / 2),
        )
        self.delete_button.setStyleSheet(
            """
            QPushButton {
                background-color: #455364;   /* Set the background color */
                color: white;                /* Set the text color */
            }
            QPushButton:checked {
                background-color: #e35727;  /* Checked state */
                color: white;
            
            }
            QPushButton:hover {
                background-color: #54687A;   /* Change background on hover */
            }
            QPushButton:disabled {
                color: #788D9C;   /* Disabled state */
            }

        """
        )
        self.delete_button.setMinimumWidth(self.buttons_min_width)

        self.performing_embedding_label = QLabel(text="No image loaded", parent=self)
        self.performing_embedding_label.move(
            9 * self.buttons_spacing + 8 * self.buttons_min_width,
            int(self.height_offset / 2),
        )
        self.performing_embedding_label.setMinimumWidth(
            self.buttons_min_width * 6,
        )
        if self.experiment_mode is None:

            self.next_img_button = QPushButton(text="next image", parent=self)
            self.next_img_button.move(
                7 * self.buttons_spacing + 6 * self.buttons_min_width,
                int(self.height_offset / 2),
            )
            self.next_img_button.setMinimumWidth(self.buttons_min_width)

            self.labelCoords = QLabel(self, text="Pixel Pos.")
            self.labelCoords.move(
                8 * self.buttons_spacing + 7 * self.buttons_min_width,
                int(self.height_offset / 2),
            )
            self.labelCoords.show()

            self.sam2_checkbox = QCheckBox(text="SAM2", parent=self)
            self.sam2_checkbox.move(
                15 * self.buttons_spacing + 14 * self.buttons_min_width,
                int(self.height_offset / 2),
            )
            self.sam2_checkbox.setChecked(True)

        if self.experiment_mode in ["structured", "polygon", "tutorial"]:
            self.next_method_button = QPushButton("Next", self)
            self.next_method_button.setFont(QFont("Arial", 16, QFont.Weight.Bold))
            self.next_method_button.move(
                int(7.5 * self.buttons_spacing) + int(6.5 * self.buttons_min_width),
                int(self.height_offset / 2),
            )
            self.next_method_button.setMinimumWidth(100)
            self.next_method_button.setStyleSheet(
                """
                QPushButton {
                    background-color: green;  /* Set background color */
                    color: white;             /* Set text color */
                    border-radius: 10px;      /* Rounded corners */
                }
                QPushButton:hover {
                    background-color: darkgreen;  /* Change color on hover */
                }
                """
            )

            self.experiment_instructions_label = QLabel(
                text="Find instructions here", parent=self
            )
            instruction_x = 15 * self.buttons_spacing + 14 * self.buttons_min_width
            self.experiment_instructions_label.move(
                instruction_x,
                int(self.height_offset / 2),
            )
            self.experiment_instructions_label.setMinimumWidth(
                self.width() - instruction_x,
            )

    def calcluate_size_of_annotation_visualizers(self) -> tuple[int]:
        vis_width = int(self.width() / 2)
        vis_height = int((self.height() - self.height_offset) / 2)
        return vis_width, vis_height

    def exit(self):
        ret = QMessageBox.question(self, "Exit", "Are you sure you want to exit?")
        if ret == QMessageBox.StandardButton.Yes:
            self.close()

    def save(self):
        self.save_signal.emit(1)

    def save_as(self):
        user_selected_dir = self.open_ouput_dir_selection()
        if user_selected_dir:
            self.save()

    def load_img(self):
        self.load_img_signal.emit(1)

    def load_img_folder(self):
        self.load_img_folder_signal.emit(1)

    def open_ouput_dir_selection(self) -> bool:
        filepath = QFileDialog.getExistingDirectory(
            parent=self,
            caption="Select Output Folder",
            directory="${HOME}/output",
        )
        if filepath is not None:
            self.output_dir_signal.emit(filepath)
            return True
        else:
            return False

    def _open_sam_model_selection(self):
        ret = QMessageBox.question(
            self,
            "SAM Model Selection",
            "Are you sure you want to discard everything and load a different SAM model?",
        )
        if ret == QMessageBox.StandardButton.Yes:
            sam_path = QFileDialog.getOpenFileName(
                parent=self,
                caption="Select SAM Model",
                directory="${HOME}",
                filter="*.pth",
            )
            if sam_path is not None:
                self.sam_path_signal.emit(sam_path[0])

    def open_layout_settings_box(self):
        self.options_window = OptionsWindow(
            layout_options=self.ui_options["layout_settings_options"]
        )
        self.options_window.show()
        self.options_window.layout_settings_options_signal.connect(
            self._recv_layout_options_changes
        )

    def open_save_annots_box(self):
        ret = QMessageBox.question(
            self,
            "Save Annotations",
            "Do you want to save the annotations before moving on to the next image?",
        )
        if ret == QMessageBox.StandardButton.Yes:
            self.save()
        elif ret == QMessageBox.StandardButton.No:
            return

    def _recv_layout_options_changes(self, layout_options: list[str]):
        self.ui_options["layout_settings_options"]["current"] = layout_options
        self.layout_options_signal.emit(layout_options)

    def handleCoords(self, point: QPoint):
        if point.isNull():
            if self.experiment_mode is None:
                self.labelCoords.clear()
            return
        if self.experiment_mode is None:
            self.labelCoords.setText(f"{point.x()}, {point.y()}")
        self.mouse_position.emit((point.x(), point.y()))

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
        if self.zoom_level == 0:
            print("no pan when fully zoomed out")
            return

        scene_idx_under_mouse = self.get_annotation_visualizer_idx_under_mouse()
        if scene_idx_under_mouse is None:
            print("not on image")
            return

        if self.last_panned_img != scene_idx_under_mouse:
            for idx, ann_viz in enumerate(self.annotation_visualizers):
                ann_viz.fitInView()
                zoom = self.zoom_factor**self.zoom_level
                ann_viz.scale(zoom, zoom)
            self.last_panned_img = scene_idx_under_mouse

        self.current_viewport = (
            self.annotation_visualizers[scene_idx_under_mouse]
            .mapToScene(
                self.annotation_visualizers[scene_idx_under_mouse].viewport().geometry()
            )
            .boundingRect()
        )
        for idx, ann_viz in enumerate(self.annotation_visualizers):
            if idx == scene_idx_under_mouse:
                continue
            ann_viz.setSceneRect(self.current_viewport)

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
                factor = 1 / self.zoom_factor
                self.zoom_level -= 1

        if event.angleDelta().y() > 0:
            self.zoom_level += 1
            factor = self.zoom_factor

        # scale and save viewport and transformation of scene under mouse
        self.annotation_visualizers[scene_idx_under_mouse].scale(factor, factor)
        self.current_viewport = (
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
            ann_viz.setSceneRect(self.current_viewport)

    def new_pixmap_displayed_in_annotation_vizualizers_callback(self, ann_viz_id: int):
        if self.zoom_level == 0:
            self.fit_annotation_visualizers_to_view()
        else:
            self.annotation_visualizers[ann_viz_id].scale(
                self.zoom_factor**self.zoom_level, self.zoom_factor**self.zoom_level
            )
            if self.current_viewport is not None:
                self.annotation_visualizers[ann_viz_id].setSceneRect(
                    self.current_viewport
                )

    def get_annotation_visualizer_idx_under_mouse(self) -> int:
        scene_idx_under_mouse = None

        for idx, ann_viz in enumerate(self.annotation_visualizers):
            if ann_viz.underMouse():
                scene_idx_under_mouse = idx
        return scene_idx_under_mouse

    def fit_annotation_visualizers_to_view(self):
        for ann_viz in self.annotation_visualizers:
            ann_viz.fitInView()

    def center_all_annotation_visualizers(self, center_p):
        self.fit_annotation_visualizers_to_view()
        for ann_viz in self.annotation_visualizers:
            ann_viz.scale(
                self.zoom_factor**self.zoom_level, self.zoom_factor**self.zoom_level
            )
        self.annotation_visualizers[0].centerOn(*center_p)
        self.current_viewport = (
            self.annotation_visualizers[0]
            .mapToScene(self.annotation_visualizers[0].viewport().geometry())
            .boundingRect()
        )
        for ann_viz in self.annotation_visualizers:
            ann_viz.setSceneRect(self.current_viewport)

    def create_message_box(
        self, crticial: bool = False, text: str = "", wait_for_user: bool = False
    ):
        self.msg_box = QMessageBox(self)
        icon = QMessageBox.Icon.Critical if crticial else QMessageBox.Icon.Information
        self.msg_box.setIcon(icon)
        self.msg_box.setText(text)
        if wait_for_user:
            self.msg_box.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            self.msg_box.setDefaultButton(QMessageBox.StandardButton.No)
            result = self.msg_box.exec()
            if result == QMessageBox.StandardButton.Yes:
                return True
            elif (
                result == QMessageBox.StandardButton.No
                or result == QMessageBox.StandardButton.NoButton
            ):
                print("No button clicked")
                return False
        else:
            self.msg_box.show()

    def create_info_box(
        self, crticial: bool = False, text: str = "", wait_for_user: bool = False
    ):
        self.info_box = QMessageBox(self)
        icon = QMessageBox.Icon.Critical if crticial else QMessageBox.Icon.Information
        self.info_box.setIcon(icon)
        self.info_box.setText(text)
        if wait_for_user:
            self.info_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            self.info_box.setDefaultButton(QMessageBox.StandardButton.Ok)
            result = self.info_box.exec()
            if result == QMessageBox.StandardButton.Ok:
                return True
            elif result == QMessageBox.StandardButton.NoButton:
                print("No button clicked")
                return False
        else:
            self.info_box.setStandardButtons(QMessageBox.StandardButton.NoButton)
            self.info_box.show()

    def create_basic_loading_window(self):
        self.basic_loading_window = QMessageBox(self)
        self.basic_loading_window.setIcon(QMessageBox.Icon.Information)
        self.basic_loading_window.setText("Loading, please wait...")
        self.basic_loading_window.show()

    def close_basic_loading_window(self):
        if self.basic_loading_window is not None:
            self.basic_loading_window.close()
            self.basic_loading_window = None

    def create_loading_window(
        self, label_text: str, max_val: int = 100, initial_val: int = 0
    ):
        self.loading_window = QProgressDialog(self)
        self.loading_window.setWindowTitle("Loading...")
        self.loading_window.setLabelText(label_text)
        self.loading_window.setCancelButtonText(None)  # Hide the cancel button
        self.loading_window.setRange(initial_val, max_val)
        self.loading_window.setWindowModality(Qt.WindowModality.WindowModal)

    def update_loading_window(self, val: int):
        if isinstance(val, tuple):
            val = int(val[0] / val[1] * 100)

        if self.loading_window is not None:
            self.loading_window.setValue(val)
            if val >= self.loading_window.maximum():
                self.loading_window.close()

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

    def update_main_pix_map(self, idx: int, img: np.ndarray | None):
        if img is None:
            self.annotation_visualizers[idx].set_pixmap(pixmap=None)
        else:
            q_img = self.convert_ndarray_to_qimage(img=img)
            pixmap = QPixmap.fromImage(q_img)
            self.annotation_visualizers[idx].set_pixmap(pixmap=pixmap)

    def resizeEvent(self, event):
        vis_width, vis_height = self.calcluate_size_of_annotation_visualizers()

        idx = 0
        for i in range(2):
            for j in range(2):
                self.annotation_visualizers[idx].setGeometry(
                    i * vis_width,
                    self.height_offset + j * vis_height,
                    vis_width,
                    vis_height,
                )
                self.annotation_visualizers[idx].fitInView()
                idx += 1

    def set_cursor(self, cross: bool = False):
        if cross:
            QApplication.setOverrideCursor(Qt.CursorShape.CrossCursor)
        else:
            QApplication.restoreOverrideCursor()

    def start_tutorial(self, mode: str):
        """
        Starts the tutorial overlay.
        Parameters:
        - mode: Can be "ui_overview" or "kernel_examples".
        """
        if mode == "ui_overview":
            settings_file = "ExperimentData/TutorialSettings/ui_overview.json"
            select_masks = False

        elif mode == "kernel_examples":
            settings_file = "ExperimentData/TutorialSettings/kernel_examples.json"
            select_masks = True

        else:
            raise ValueError("Invalid tutorial mode.")

        if not os.path.exists(settings_file):
            raise FileNotFoundError("Tutorial settings file not found.")

        with open(settings_file, "r") as file:
            tut_steps = json.load(file)

        self.tutorial_overlay = TutorialOverlay(
            parent=self, tutorial_steps=tut_steps, select_masks=select_masks
        )
        self.tutorial_overlay.exec()


class TutorialOverlay(QDialog):
    def __init__(
        self, parent=None, tutorial_steps: list[dict] = None, select_masks: bool = False
    ):
        super().__init__(parent)
        self.parent = parent
        self.resize(self.parent.size())
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.move(0, 0)

        self.select_masks = select_masks
        self.steps = tutorial_steps
        self.current_step = 0

        # Navigation buttons
        self.button_width = 70
        self.next_tutorial_button = QPushButton("Next", self)
        self.next_tutorial_button.clicked.connect(self.next_step)
        self.next_tutorial_button.move(700, 550)
        self.next_tutorial_button.setMinimumWidth(self.button_width)

        self.prev_tutorial_button = QPushButton("Previous", self)
        self.prev_tutorial_button.clicked.connect(self.prev_step)
        self.prev_tutorial_button.move(600, 550)
        self.prev_tutorial_button.setMinimumWidth(self.button_width)

        self.close_tutorial_button = QPushButton("Close", self)
        self.close_tutorial_button.clicked.connect(self.close)
        self.close_tutorial_button.move(750, 550)
        self.close_tutorial_button.setMinimumWidth(self.button_width)
        self.close_tutorial_button.setVisible(False)

        self.arrow_buffer_space = 20
        self.arrow_length = 100

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.steps:
            if self.current_step == len(self.steps) - 1:
                self.next_tutorial_button.setVisible(False)
                self.close_tutorial_button.setVisible(True)
            else:
                self.next_tutorial_button.setVisible(True)

            self.resize(self.parent.size())
            step = self.steps[self.current_step]
            # widget = step["widget"]
            widget = getattr(self.parent, step["widget"])
            if step["object_index"] is not None:
                widget = widget[step["object_index"]]
            widget_rect = widget.geometry()

            # Map widget rect to overlay coordinates
            global_pos = widget.mapToGlobal(QPoint(0, 0))
            overlay_pos = self.mapFromGlobal(global_pos)
            widget_rect = QRect(overlay_pos, widget_rect.size())

            # Create a transparent path over the widget
            path = QPainterPath()
            path.addRect(QRectF(self.rect()))
            path.addRoundedRect(QRectF(widget_rect.adjusted(-10, -10, 10, 10)), 10, 10)

            painter.fillPath(path, QColor(0, 0, 0, 180))

            widget_center = widget_rect.center()
            if step["arrow_position"] == "bottom":
                arrow_end = QPointF(
                    widget_center.x(), widget_rect.bottom() + self.arrow_buffer_space
                )
                arrow_start = QPointF(
                    widget_center.x(),
                    widget_rect.bottom() + self.arrow_buffer_space + self.arrow_length,
                )
                self.draw_arrow(painter, arrow_start, arrow_end)

            elif step["arrow_position"] == "top":
                arrow_end = QPointF(
                    widget_center.x(), widget_rect.top() - self.arrow_buffer_space
                )
                arrow_start = QPointF(
                    widget_center.x(),
                    widget_rect.top() - self.arrow_buffer_space - self.arrow_length,
                )
                self.draw_arrow(painter, arrow_start, arrow_end)

            elif step["arrow_position"] == "none":
                arrow_start = QPointF(
                    widget_center.x(), widget_rect.bottom() + self.arrow_buffer_space
                )

            # Draw the tutorial text
            text = step["text"]
            painter.setPen(QColor(255, 255, 255))
            painter.setBrush(QColor(0, 0, 0, 200))
            text_rect = self.move_instructions(
                widget_center,
                int(arrow_start.y()),
                arrow_below_text=(step["arrow_position"] == "bottom"),
            )
            painter.drawRoundedRect(text_rect, 10, 10)
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                text,
            )

    def move_instructions(
        self,
        widget_center: QPoint,
        arrow_ymin: int,
        arrow_below_text: bool = True,
        text_box_width: int = 600,
        text_box_height: int = 100,
        edge_buffer: int = 10,
    ):
        min_cx = edge_buffer + int(0.5 * text_box_width)
        max_cx = self.width() - edge_buffer - int(0.5 * text_box_width)

        if widget_center.x() < min_cx:
            text_aleft = edge_buffer

        elif widget_center.x() > max_cx:
            text_aleft = self.width() - edge_buffer - text_box_width
        else:
            text_aleft = widget_center.x() - int(0.5 * text_box_width)
        if arrow_below_text:
            text_atop = arrow_ymin + edge_buffer
        else:
            text_atop = (
                arrow_ymin
                - text_box_height
                - 2 * edge_buffer
                - self.next_tutorial_button.height()
            )
        box_center_x = text_aleft + int(0.5 * text_box_width)
        button_y = text_atop + text_box_height + edge_buffer
        button_x_base = box_center_x - int(0.5 * self.button_width)

        self.prev_tutorial_button.move(button_x_base - 100, button_y)
        self.next_tutorial_button.move(button_x_base, button_y)
        self.close_tutorial_button.move(button_x_base + 100, button_y)

        text_rect = QRect(text_aleft, text_atop, text_box_width, text_box_height)
        return text_rect

    def draw_arrow(self, painter: QPainter, start: QPointF, end: QPointF):
        """
        Draws an arrow from start to end points.

        :param painter: QPainter object
        :param start: QPointF, starting point of the arrow
        :param end: QPointF, ending point of the arrow
        """
        pen_thickness = 4
        # Draw the main line
        pen = QtGui.QPen()
        pen.setWidth(pen_thickness)
        pen.setColor(QtGui.QColor("white"))
        painter.setPen(pen)
        if start.y() > end.y():
            line_end = QPointF(end.x(), end.y() + pen_thickness)
        elif start.y() < end.y():
            line_end = QPointF(end.x(), end.y() - pen_thickness)
        painter.drawLine(start, line_end)
        # painter.setPen(QtGui.QPen())

        # Calculate the arrowhead
        arrow_size = 20
        angle = 30  # degrees

        # Calculate the direction of the line

        line_angle = math.atan2(end.y() - start.y(), end.x() - start.x())

        # Calculate the points for the arrowhead
        angle_rad = math.radians(angle)
        # First arrowhead point
        arrow_p1 = QPointF(
            end.x() - arrow_size * math.cos(line_angle - angle_rad),
            end.y() - arrow_size * math.sin(line_angle - angle_rad),
        )
        # Second arrowhead point
        arrow_p2 = QPointF(
            end.x() - arrow_size * math.cos(line_angle + angle_rad),
            end.y() - arrow_size * math.sin(line_angle + angle_rad),
        )

        # Create a polygon for the arrowhead
        arrow_head = QPolygonF([end, arrow_p1, arrow_p2])

        # Fill the arrowhead
        painter.setBrush(QColor(255, 255, 255))
        painter.drawPolygon(arrow_head)

    def next_step(self):
        if self.current_step < len(self.steps) - 1:
            if self.select_masks:
                if self.current_step % 2 == 0:
                    self.parent.good_mask_button.click()
                else:
                    self.parent.bad_mask_button.click()
            self.current_step += 1
            self.update()

    def prev_step(self):
        if self.select_masks:
            self.parent.back_button.click()
        if self.current_step > 0:
            self.current_step -= 1
            self.update()


class InteractiveGraphicsView(QGraphicsView):
    coordinatesChanged: pyqtSignal = pyqtSignal(QPoint)
    mouseWheelScroll: pyqtSignal = pyqtSignal(QWheelEvent)
    mouseMove: pyqtSignal = pyqtSignal(QMouseEvent)
    clicked: pyqtSignal = pyqtSignal(bool)
    mousePressSignal: pyqtSignal = pyqtSignal(QMouseEvent)
    new_scene_initialized: pyqtSignal = pyqtSignal(int)

    def __init__(
        self, parent, ann_viz_id: int, width: int = 1024, height: int = 1024
    ) -> None:
        super().__init__(parent=parent)

        self.width = width
        self.height = height
        self.id = ann_viz_id
        self.mouse_l_down = False
        self.graphics_scene = QGraphicsScene(self)
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setShapeMode(QGraphicsPixmapItem.ShapeMode.BoundingRectShape)
        self.pixmap_item_is_set = False

        self.init_scene()

    def init_scene(self):
        self.graphics_scene.addItem(self.pixmap_item)
        self.setScene(self.graphics_scene)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.Shape.NoFrame)

        self.pixmap_item_is_set = False
        self.new_scene_initialized.emit(self.id)

    def reset_scene(self):
        self.graphics_scene.removeItem(self.pixmap_item)
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setShapeMode(QGraphicsPixmapItem.ShapeMode.BoundingRectShape)
        self.init_scene()

    def set_pixmap(self, pixmap: QPixmap | None):
        if pixmap is None:
            if self.pixmap_item_is_set:
                self.reset_scene()
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.pixmap_item.setPixmap(pixmap)
            if not self.pixmap_item_is_set:
                self.fitInView()
                self.new_scene_initialized.emit(self.id)
            self.pixmap_item_is_set = True

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
        super().mouseMoveEvent(event)
        self.updateCoordinates(event.position().toPoint())
        if self.mouse_l_down:
            self.mouseMove.emit(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button().name == "LeftButton":
            self.mouse_l_down = True
        self.mousePressSignal.emit(event)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button().name == "LeftButton":
            self.mouse_l_down = False
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


class OptionsWindow(QMainWindow):
    layout_settings_options_signal: pyqtSignal = pyqtSignal(list)

    def __init__(self, layout_options: list[str]) -> None:
        super().__init__(parent=None)
        box_height = 40
        box_width = 150

        self.setWindowTitle("Options")
        self.resize(2 * box_width, 3 * box_height + 10)

        self.comboboxes: list[QtWidgets.QComboBox] = []
        for i in range(2):
            for j in range(2):
                combobox = QtWidgets.QComboBox(self)
                combobox.setGeometry(
                    i * box_width, j * box_height, box_width, box_height
                )
                [combobox.addItem(option) for option in layout_options["options"]]
                combobox.setCurrentText(layout_options["current"][i * 2 + j])
                self.comboboxes.append(combobox)
        self.ok_button = QPushButton(parent=self, text="OK")
        self.ok_button.setGeometry(
            int(box_width / 2), box_height * 2 + 10, box_width, box_height
        )
        self.ok_button.setIcon(QtGui.QIcon.fromTheme("edit-undo"))
        self.ok_button.clicked.connect(self._accept)

    def _accept(self):
        combobox = QtWidgets.QComboBox(self)
        options = []
        for combobox in self.comboboxes:
            options.append(combobox.currentText())

        self.layout_settings_options_signal.emit(options)
