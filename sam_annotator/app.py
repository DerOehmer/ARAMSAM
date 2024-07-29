import sys
import time
import numpy as np
import traceback

from os import listdir
from os.path import isfile, join, basename
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QRunnable, QThreadPool, pyqtSignal, QThreadPool, QObject
from natsort import natsorted
from segment_anything import SamPredictor

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
        self.threadpool = QThreadPool()
        self.img_fnames = []

        self.ui.test_button.clicked.connect(self.segment_anything)
        self.ui.good_mask_button.clicked.connect(self.add_good_mask)
        self.ui.bad_mask_button.clicked.connect(self.add_bad_mask)
        self.ui.back_button.clicked.connect(self.last_mask)
        self.ui.manual_annotation_button.clicked.connect(self.manual_annotation)
        self.ui.next_img_button.clicked.connect(self.select_next_img)

        self.ui.mouse_position.connect(self.mouse_move_on_img)
        self.ui.load_img_signal.connect(self.load_img)
        self.ui.load_img_folder_signal.connect(self.load_img_folder)
        self.ui.preview_annotation_point_signal.connect(
            self.add_sam_preview_annotation_point
        )

        self.manual_sam_preview_updates_per_sec = 10
        self.last_sam_preview_time_stamp = time.time_ns()

    def run(self) -> None:
        self.ui.run()
        sys.exit(self.application.exec())

    def load_img(self, _) -> None:
        print("loading new image")
        img_fpath = self.ui.open_img_load_file_dialog()
        self.img_fnames.append(img_fpath)
        self.select_next_img()

    def load_img_folder(self, _) -> None:
        print("loading folder")
        img_dir = self.ui.open_load_folder_dialog()
        img_fnames = [
            join(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f))
        ]
        self.img_fnames.extend(img_fnames)
        self.img_fnames = natsorted(self.img_fnames)
        self.select_next_img()

    def select_next_img(self):
        if self.img_fnames:
            next_img_name = Path(self.img_fnames.pop())
            self.annotator.create_new_annotation(next_img_name)
            self.update_ui_imgs()
            self.embed_img(basename(next_img_name))
        else:
            print("No image left in the queue")

    def embed_img(self, img_name: str):
        worker = PyQtWorker(
            sam_predictor=self.annotator.sam.predictor,
            img=self.annotator.annotation.img,
            img_name=img_name,
        )
        worker.signals.result.connect(self.receive_predictor)
        worker.signals.finished.connect(self.embedding_done)
        self.threadpool.start(worker)

        embedding_threads = self.threadpool.activeThreadCount()
        if embedding_threads > 1:
            self.ui.performing_embedding_label.setText(
                f"Embedding {embedding_threads} images"
            )
        else:
            self.ui.performing_embedding_label.setText(f"Embed img ... - {img_name}")

    def embedding_done(self, img_name: str):
        # TODO: if multiple images should be embedded in the background
        # filter and map them by the image name here
        self.annotator.sam.is_img_embedded = True

        embedding_threads = self.threadpool.activeThreadCount()
        if embedding_threads > 0:
            self.ui.performing_embedding_label.setText(
                f"Embedding {embedding_threads} images"
            )
        else:
            self.ui.performing_embedding_label.setText(f"Embedding done! - {img_name}")

    def receive_predictor(self, predictor: SamPredictor):
        self.annotator.sam.predictor = predictor

    def segment_anything(self):
        self.annotator.predict_with_sam()
        self.update_ui_imgs()

    def update_ui_imgs(self):
        mviss = self.annotator.annotation.mask_visualizations
        fields = ["img", "mask", "img_sam_preview", "mask_collection_cnt"]
        if len(fields) > 4:
            print(
                f"Too many fields selected for visualization ({len(fields)}) expected 4"
            )
        for idx, field in enumerate(fields):
            if getattr(mviss, field) is not None:
                self.ui.update_main_pix_map(idx=idx, img=getattr(mviss, field))

    def add_good_mask(self):
        new_center = self.annotator.good_mask()
        if new_center is None:
            self.ui.create_message_box(False, "All masks are done")
        # TODO: use center for centering large images
        self.update_ui_imgs()

    def add_bad_mask(self):
        new_center = self.annotator.bad_mask()
        if new_center is None:
            self.ui.create_message_box(False, "All masks are done")
        # TODO: use center for centering large images
        self.update_ui_imgs()

    def last_mask(self):
        done = self.annotator.update_mask_idx(self.annotator.mask_idx - 1)
        # TODO: error handling if mask idx is out of bounds
        self.update_ui_imgs()

    def manual_annotation(self):
        self.annotator.toggle_manual_annotation()

    def mouse_move_on_img(self, point: tuple[int]):
        current_time = time.time_ns()
        delta = current_time - self.last_sam_preview_time_stamp
        if delta * 1e-9 > 1 / self.manual_sam_preview_updates_per_sec:
            self.mouse_pos = point
            self.annotator.predict_sam_manually(point)
            self.last_sam_preview_time_stamp = current_time
            self.update_ui_imgs()

    def add_sam_preview_annotation_point(self, label: int):
        if not self.annotator.manual_annotation_enabled:
            return
        if label == -1:
            if self.annotator.manual_mask_points:
                self.annotator.manual_mask_points.pop()
                self.annotator.manual_mask_point_labels.pop()
        else:
            self.annotator.manual_mask_points.append(self.mouse_pos)
            self.annotator.manual_mask_point_labels.append(label)

        self.annotator.predict_sam_manually(self.mouse_pos)
        self.update_ui_imgs()


# https://www.pythonguis.com/tutorials/multithreading-pyqt6-applications-qthreadpool/
class PyQtWorker(QRunnable):
    finished: pyqtSignal = pyqtSignal(str)
    error: pyqtSignal = pyqtSignal(tuple)
    result: pyqtSignal = pyqtSignal(SamPredictor)

    def __init__(self, sam_predictor: SamPredictor, img: np.ndarray, img_name: str):
        super(PyQtWorker, self).__init__()
        self.signals = WorkerSignals()
        self.sam_predictor = sam_predictor
        self.img = img
        self.img_name = img_name

    def run(self):
        try:
            self.sam_predictor.set_image(self.img)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(self.sam_predictor)
        finally:
            self.signals.finished.emit(self.img_name)


class WorkerSignals(QObject):
    finished: pyqtSignal = pyqtSignal(str)
    error: pyqtSignal = pyqtSignal(tuple)
    result: pyqtSignal = pyqtSignal(object)
