import sys
import os
import time
import numpy as np
import traceback
import qdarkstyle

from os import listdir
from os.path import isfile, join, basename
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QRunnable, QThreadPool, pyqtSignal, QThreadPool, QObject
from natsort import natsorted

from sam_annotator.run_sam import CustomSamPredictor
from sam_annotator.gui import UserInterface
from sam_annotator.annotator import Annotator, PanoImageAligner


class App:
    """
    Contains the UI with main thread and subprocesses for the annotation
    """

    def __init__(self, ui_options: dict = None) -> None:
        self.application = QApplication([])
        self.application.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
        self.ui = UserInterface(ui_options=ui_options)
        self.annotator = Annotator()
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(1)
        self.img_fnames = []
        self.output_dir = None

        self.ui.good_mask_button.clicked.connect(self.add_good_mask)
        self.ui.bad_mask_button.clicked.connect(self.add_bad_mask)
        self.ui.back_button.clicked.connect(self.last_mask)
        self.ui.manual_annotation_button.clicked.connect(self.manual_annotation)
        self.ui.draw_poly_button.clicked.connect(self.draw_polygon)
        self.ui.next_img_button.clicked.connect(self.select_next_img)

        self.ui.mouse_position.connect(self.mouse_move_on_img)
        self.ui.load_img_signal.connect(self.load_img)
        self.ui.load_img_folder_signal.connect(self.load_img_folder)
        self.ui.output_dir_signal.connect(self.change_output_dir)
        self.ui.sam_path_signal.connect(self.changed_sam_model)
        self.ui.save_signal.connect(self.save_output)
        self.ui.preview_annotation_point_signal.connect(
            self.add_sam_preview_annotation_point
        )

        self.ui.layout_options_signal.connect(self._ui_config_changed)

        self.fields = ui_options["layout_settings_options"]["default"]

        self.manual_sam_preview_updates_per_sec = 10
        self.last_sam_preview_time_stamp = time.time_ns()
        self.pano_aligner = None
        self.sam2 = False

    def run(self) -> None:
        self.ui.run()
        sys.exit(self.application.exec())

    def set_sam(self):
        self.sam2 = self.ui.sam2_checkbox.isChecked()
        self.annotator.set_sam_version(sam2=self.sam2)

    def save_output(self, _=None):
        if self.output_dir is None:
            print("Select output directory before saving")
            self.ui._open_ouput_dir_selection()
            self.ui.save()
            return
        if not Path(self.output_dir).exists():
            print(f"Path {self.output_dir} does not exist")
            self.ui._open_ouput_dir_selection()
            self.ui.save()
            return

        output_exists = self.annotator.save_annotations(self.output_dir)
        if output_exists:
            self.ui.create_message_box(
                True,
                "Annotations already exist. Maybe outpupath should be provided together with input path, so programme can check whether this img has already been annotated?",
            )

    def change_output_dir(self, out_dir: str):
        self.output_dir = out_dir

    def changed_sam_model(self, model_path: str):
        if "vit_b" in model_path:
            model_type = "vit_b"
        elif "vit_h" in model_path:
            model_type = "vit_h"
        elif "vit_l" in model_path:
            model_type = "vit_l"
        else:
            print(f"could not infer model type from model path {model_path}")
            print("model path must include one of [vit_b, vit_h, vit_l]")
        self.annotator = Annotator(sam_ckpt=model_path, sam_model_type=model_type)

        ### this is lazy and shouled be changed! ###
        self.ui.construct_ui()
        ### this is lazy and shouled be changed! ###

        # TODO: kill & collect old threads first!
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(1)
        self.img_fnames = []
        self.output_dir = None

    def load_img(self, _) -> None:
        self.set_sam()
        print("loading new image")
        img_fpath = self.ui.open_img_load_file_dialog()
        if img_fpath == "":
            return
        self.img_fnames.append(img_fpath)
        self.select_next_img()

    def load_img_folder(self, _) -> None:
        self.set_sam()
        print("loading folder")
        img_dir = self.ui.open_load_folder_dialog()
        if img_dir == "":
            return
        img_fnames = [
            join(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f))
        ]
        self.img_fnames.extend(img_fnames)
        self.img_fnames = natsorted(self.img_fnames)
        self.select_next_img()

    def select_next_img(self):
        if self.annotator.annotation is not None:
            self.ui.open_save_annots_box()

        if not self.img_fnames:
            print("No image left in the queue")
            return
        if self.threadpool.activeThreadCount() > 0:
            print(
                "Wait for embedding calculation to be finished before skipping to next img"
            )
            return
        img_name = Path(self.img_fnames.pop())
        if self.img_fnames:
            next_img_name = Path(self.img_fnames[-1])
        else:
            next_img_name = None

        self.propagate_good_masks()

        embed_current, embed_next = self.annotator.create_new_annotation(
            filepath=img_name, next_filepath=next_img_name
        )
        if self.sam2:
            self.embed_img_pair()
        else:
            if embed_current:
                self.embed_img(basename(img_name))
            else:
                self.annotator.update_sam_features_to_current_annotation()
                self.segment_anything()
            if embed_next:
                self.embed_img(basename(next_img_name))
        self.annotator.init_time_stamp()

    def propagate_good_masks(self):
        if (
            not self.annotator.sam.predictor.is_image_set
            and self.annotator.annotation is None
        ) or not self.annotator.annotation.good_masks:
            return
        if self.sam2:
            self.annotator.track_good_masks()
        else:
            if self.pano_aligner is None:
                self.pano_aligner = PanoImageAligner()
            self.pano_aligner.add_image(
                self.annotator.annotation.img, self.annotator.annotation.good_masks
            )

    def embed_img_pair(self):
        # SAM2
        if self.annotator.next_annotation is not None:
            img_pair = [
                self.annotator.annotation.img,
                self.annotator.next_annotation.img,
            ]
        else:
            img_pair = [self.annotator.annotation.img]
        self.annotator.sam.set_features(img_pair)
        self.segment_anything()

    def embed_img(self, img_name: str):
        current_ann_name = self.annotator.get_annotation_img_name()
        next_ann_name = self.annotator.get_next_annotation_img_name()

        img_to_embed = None

        if current_ann_name:
            if current_ann_name == img_name:
                img_to_embed = self.annotator.annotation.img
                delay = 0.0

        if next_ann_name:
            if next_ann_name == img_name:
                img_to_embed = self.annotator.next_annotation.img
                delay = 1.5

        if img_to_embed is None:
            print(f"Could not find img name ({img_name}) in annotations")
            print(f"Found {current_ann_name} and {next_ann_name}")
            raise ValueError("Embedding could not be matched to images")

        worker = PyQtWorker(
            sam_predictor=self.annotator.sam.predictor,
            img=img_to_embed,
            img_name=img_name,
            delay=delay,
        )
        worker.signals.result.connect(self.receive_embedding_from_thread)
        worker.signals.finished.connect(self.embedding_done)
        self.threadpool.start(worker)

        embedding_threads = self.threadpool.activeThreadCount()
        self.ui.performing_embedding_label.setText(
            f"Embedding {embedding_threads} images"
        )

    def embedding_done(self, img_name: str):
        print(f"Embedding of {img_name} done")
        embedding_threads = self.threadpool.activeThreadCount()
        if embedding_threads > 0:
            self.ui.performing_embedding_label.setText(
                f"Embedding {embedding_threads} images"
            )
        else:
            self.ui.performing_embedding_label.setText(f"Embeddings done!")

    def receive_embedding_from_thread(self, result: tuple):
        features, original_size, input_size, img_name = result
        current_ann_name = self.annotator.get_annotation_img_name()
        if current_ann_name:
            if current_ann_name == img_name:
                self.annotator.annotation.set_sam_parameters(
                    features=features,
                    original_size=original_size,
                    input_size=input_size,
                )

                self.annotator.update_sam_features_to_current_annotation()
                self.segment_anything()

                return

        next_ann_name = self.annotator.get_next_annotation_img_name()
        if next_ann_name:
            if next_ann_name == img_name:
                self.annotator.next_annotation.set_sam_parameters(
                    features=features,
                    original_size=original_size,
                    input_size=input_size,
                )
                return
        print(f"Warning could not find annotation object with fitting name {img_name}")

    def segment_anything(self):
        now = time.time()
        self.annotator.predict_with_sam(self.pano_aligner)
        duration = time.time() - now
        print(f"SAM inference {duration}")

        now = time.time()
        self.update_ui_imgs()
        duration = time.time() - now
        print(f"update ui {duration}")

    def _ui_config_changed(self, fields: list[str]):
        assert (
            len(fields) == 4
        ), f"Too many fields selected for visualization ({len(fields)}) expected 4"
        self.fields = fields
        self.update_ui_imgs()

    def update_ui_imgs(self):
        mviss = self.annotator.annotation.mask_visualizations
        for idx, field in enumerate(self.fields):
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
        # done = self.annotator.update_mask_idx(self.annotator.mask_idx - 1)
        self.annotator.step_back()
        # TODO: error handling if mask idx is out of bounds
        self.update_ui_imgs()

    def manual_annotation(self):
        self.annotator.toggle_manual_annotation()

    def draw_polygon(self):
        self.annotator.toggle_polygon_drawing()

    def mouse_move_on_img(self, point: tuple[int]):
        current_time = time.time_ns()
        delta = current_time - self.last_sam_preview_time_stamp
        if delta * 1e-9 > 1 / self.manual_sam_preview_updates_per_sec:
            self.mouse_pos = point
            self.annotator.predict_sam_manually(point)
            self.last_sam_preview_time_stamp = current_time
            self.update_ui_imgs()

    def add_sam_preview_annotation_point(self, label: int):
        if (
            not self.annotator.manual_annotation_enabled
            and not self.annotator.polygon_drawing_enabled
        ):
            return
        if label == -1:
            if self.annotator.manual_mask_points:
                self.annotator.manual_mask_points.pop()
                self.annotator.manual_mask_point_labels.pop()
        else:
            self.annotator.manual_mask_points.append(self.mouse_pos)
            self.annotator.manual_mask_point_labels.append(label)

        if self.annotator.manual_annotation_enabled:
            self.annotator.predict_sam_manually(self.mouse_pos)
        elif self.annotator.polygon_drawing_enabled:
            self.annotator.mask_from_polygon()

        self.update_ui_imgs()


# https://www.pythonguis.com/tutorials/multithreading-pyqt6-applications-qthreadpool/
class PyQtWorker(QRunnable):
    finished: pyqtSignal = pyqtSignal(str)
    error: pyqtSignal = pyqtSignal(tuple)
    result: pyqtSignal = pyqtSignal(tuple)

    def __init__(
        self,
        sam_predictor: CustomSamPredictor,
        img: np.ndarray,
        img_name: str,
        delay: float = 0.0,
    ):
        super(PyQtWorker, self).__init__()
        self.signals = WorkerSignals()
        self.sam_predictor = sam_predictor
        self.img = img
        self.img_name = img_name
        self.delay = delay

    def run(self):
        try:
            now = time.time()
            features, original_size, input_size = (
                self.sam_predictor.get_img_ebedding_without_overiding_current_embedding(
                    self.img
                )
            )
            duration = time.time() - now
            print(f"embedding took {duration}")
            result = (features, original_size, input_size, self.img_name)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit(self.img_name)


class Sam2PropagationWorker(QRunnable):
    finished: pyqtSignal = pyqtSignal(str)
    error: pyqtSignal = pyqtSignal(tuple)
    result: pyqtSignal = pyqtSignal(tuple)

    def __init__(
        self,
        delay: float = 0.0,
    ):
        super(PyQtWorker, self).__init__()
        self.signals = WorkerSignals()
        self.delay = delay

    def run(self):
        try:
            now = time.time()
            features, original_size, input_size = (
                self.sam_predictor.get_img_ebedding_without_overiding_current_embedding(
                    self.img
                )
            )
            duration = time.time() - now
            print(f"embedding took {duration}")
            result = (features, original_size, input_size, self.img_name)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit(self.img_name)


class WorkerSignals(QObject):
    finished: pyqtSignal = pyqtSignal(str)
    error: pyqtSignal = pyqtSignal(tuple)
    result: pyqtSignal = pyqtSignal(tuple)
