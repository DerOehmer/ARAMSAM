import sys
import time
import numpy as np
import traceback
import qdarkstyle

from os import listdir
from os.path import isfile, join, basename, isdir, splitext, basename
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import (
    QRunnable,
    pyqtSignal,
    QThreadPool,
    QObject,
    QMutex,
    pyqtSlot,
)
from natsort import natsorted

from sam_annotator.run_sam import BackgroundThreadSamPredictor, Sam2Inference
from sam_annotator.gui import UserInterface
from sam_annotator.annotator import Annotator
from sam_annotator.mask_visualizations import MaskData, AnnotationObject
from sam_annotator.tracker import PanoImageAligner


class App:
    """
    Contains the UI with main thread and subprocesses for the annotation
    """

    def __init__(self, ui_options: dict = None, experiment_mode: str = None) -> None:
        self.application = QApplication([])
        self.application.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
        self.ui = UserInterface(ui_options=ui_options, experiment_mode=experiment_mode)
        self.annotator = Annotator()
        self.threadpool = QThreadPool.globalInstance()
        self.threadpool.setMaxThreadCount(1)
        self.img_fnames = []
        self.output_dir = None
        self.experiment_mode = experiment_mode

        self.ui.good_mask_button.clicked.connect(self.add_good_mask)
        self.ui.bad_mask_button.clicked.connect(self.add_bad_mask)
        self.ui.back_button.clicked.connect(self.previous_mask)
        self.ui.manual_annotation_button.clicked.connect(self.manual_annotation)
        self.ui.draw_poly_button.clicked.connect(self.draw_polygon)
        self.ui.delete_button.clicked.connect(self.select_masks_to_delete)

        if self.experiment_mode == "structured":
            self.ui.next_method_button.clicked.connect(self.next_method)
            self.experiment_step: int = 1
        elif self.experiment_mode == "polygon":
            self.ui.next_method_button.clicked.connect(self.next_indicated_polygon_img)
        else:
            self.ui.next_img_button.clicked.connect(self.select_next_img)

        self.ui.mouse_position.connect(self.manage_mouse_move)
        self.ui.load_img_signal.connect(self.load_img)
        self.ui.load_img_folder_signal.connect(self.load_img_folder)
        self.ui.output_dir_signal.connect(self.change_output_dir)
        self.ui.sam_path_signal.connect(self.changed_sam_model)
        self.ui.save_signal.connect(self.save_output)
        self.ui.preview_annotation_point_signal.connect(self.manage_mouse_action)

        self.ui.layout_options_signal.connect(self._ui_config_changed)

        self.fields = ui_options["layout_settings_options"]["default"]

        self.manual_sam_preview_updates_per_sec: int = 5
        self.last_sam_preview_time_stamp: int = time.time_ns()
        self.bbox_tracker: object = None
        self.sam_gen: int = None

        self.mask_track_batch_size: int = 10
        self.propagated_mids: set[int] = set()
        self.mutex = QMutex()

    def run(self) -> None:
        self.ui.run()
        sys.exit(self.application.exec())

    def set_sam(self):
        if self.experiment_mode is None:
            background_embedding = True
            if self.ui.sam2_checkbox.isChecked():
                self.sam_gen = 2
            else:
                self.sam_gen = 1
        elif self.experiment_mode == "structured":
            background_embedding = False

        if self.sam_gen is None:
            raise ValueError("SAM generation has not been set")
        self.annotator.set_sam_version(
            sam_gen=self.sam_gen, background_embedding=background_embedding
        )

    def save_output(self, _=None):
        if self.output_dir is None:
            print("Select output directory before saving")
            self.ui.open_ouput_dir_selection()
            self.ui.save()
            return
        if not Path(self.output_dir).exists():
            print(f"Path {self.output_dir} does not exist")
            self.ui.open_ouput_dir_selection()
            self.ui.save()
            return

        output_exists = self.annotator.save_annotations(self.output_dir)
        if output_exists:
            self.ui.create_message_box(
                True,
                "Incomplete outputfolder detected. Files have been overwritten",
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

    def _pop_img_fnames(self) -> tuple[Path, Path]:
        img_name = Path(self.img_fnames.pop())
        if self.img_fnames:
            next_img_name = Path(self.img_fnames[-1])
        else:
            next_img_name = None
        return img_name, next_img_name

    def select_next_img(self):
        if not self.img_fnames:
            print("No image left in the queue")
            if self.experiment_mode == "structured":
                self.ui.create_message_box(False, "No image left in the queue")
                self.ui.save()
                self.ui.close()
            return
        if self.threadpool.activeThreadCount() > 0:
            print(
                "Wait for embedding calculation to be finished before skipping to next img"
            )
            self.threadpool.waitForDone(-1)
        img_name, next_img_name = self._pop_img_fnames()

        # check if img_name is already annotated in output_dir
        current_img_done, next_img_done = self.check_annotations_done(
            img_name, next_img_name
        )
        if current_img_done and next_img_done:
            if self.experiment_mode == "structured":
                raise ValueError("This directory is already occupied with annotations")
            print(f"Both {img_name} and {next_img_name} are already annotated")
            return self.select_next_img()

        elif current_img_done and not next_img_done:
            if self.experiment_mode == "structured":
                raise ValueError("This directory is already occupied with annotations")
            print(
                f"{img_name} is already annotated but {next_img_name} is not. Loading previous annotations"
            )
            self.load_previous_annotations(img_name, next_img_name)
            img_name, next_img_name = self._pop_img_fnames()

        elif self.annotator.annotation is not None:
            if self.experiment_mode is None:
                self.ui.open_save_annots_box()
            else:
                self.ui.save()
        self.propagate_good_masks()

        embed_current, embed_next = self.annotator.create_new_annotation(
            filepath=img_name, next_filepath=next_img_name
        )

        if self.sam_gen == 2:
            self.embed_img_pair()
        elif self.sam_gen == 1:
            if (
                embed_current
                or (
                    current_img_done
                    and not next_img_done  # if previous annotations wer just loaded from disk, embedding is still required
                )
                or self.experiment_mode == "structured"
            ):
                self.embed_img(basename(img_name))
            else:
                self.annotator.update_sam_features_to_current_annotation()
                self.segment_anything()
            if embed_next and self.experiment_mode is None:
                self.embed_img(basename(next_img_name))
        self.threadpool.waitForDone(-1)
        user_ready = self.ui.create_message_box(
            False,
            "Experiment is about to start. Click Yes once you are ready",
            wait_for_user=True,
        )
        if user_ready:
            self.annotator.init_time_stamp()
            self.annotator.update_collections(self.annotator.annotation)
            self.update_ui_imgs()

        else:
            self.ui.close()
            raise KeyboardInterrupt("User is not ready")

    def check_annotations_done(
        self, img_name: str, next_img_name: str
    ) -> tuple[bool, bool]:

        if self.output_dir is None:
            self.ui.open_ouput_dir_selection()
        img_name_done = self._annot_log_exists(img_name)
        next_img_name_done = self._annot_log_exists(next_img_name)
        return img_name_done, next_img_name_done

    def _annot_log_exists(self, img_name: str | None) -> bool:
        if img_name is None:
            return True

        img_path = str(img_name)
        img_id = splitext(basename(img_path))[0]
        annot_id = f"{img_id}_annots"
        annot_path = join(self.output_dir, annot_id)
        if isdir(annot_path) and isfile(join(annot_path, "log.json")):
            return True
        return False

    def load_previous_annotations(self, img_name: str, next_img_name: str):
        img_path = str(img_name)
        img_id = splitext(basename(img_path))[0]
        annot_id = f"{img_id}_annots"
        annot_path = join(self.output_dir, annot_id)
        annot_masks_path = join(annot_path, "masks")
        if not isdir(annot_masks_path):
            raise FileNotFoundError(f"Could not find masks directory in {annot_path}")

        self.annotator.create_new_annotation(
            filepath=img_name, next_filepath=next_img_name
        )
        self.annotator.annotation.load_masks_from_dir(
            Path(annot_masks_path), self.annotator.mask_id_handler
        )
        if self.sam_gen == 2:
            self.embed_img_pair(do_amg=False)

    def propagate_good_masks(self):
        if (
            (
                not self.annotator.sam.predictor.is_image_set
                and self.annotator.annotation is None
            )
            or not self.annotator.annotation.good_masks
            or self.annotator.next_annotation is None
        ):
            return
        if self.sam_gen == 2:

            self.start_mask_batch_thread(track_remaining=True)

        elif self.sam_gen == 1:
            if self.bbox_tracker is None:
                self.bbox_tracker = PanoImageAligner()
            self.bbox_tracker.add_annotation(self.annotator.annotation)

        if self.sam_gen == 2:
            self.annotator.convey_color_to_next_annot(
                self.annotator.next_annotation.masks
            )
        if self.annotator.time_stamp is None:
            self.annotator.init_time_stamp()

    def embed_img_pair(self, do_amg: bool = True):
        # SAM2
        if self.annotator.next_annotation is not None:
            img_pair = [
                self.annotator.annotation.img,
                self.annotator.next_annotation.img,
            ]
        else:
            img_pair = [self.annotator.annotation.img]
        img_embed_worker = Sam2ImgPairEmbeddingWorker(
            self.annotator.sam, img_pair, self.mutex
        )
        img_embed_worker.signals.finished.connect(self.embedding_done)
        img_embed_worker.signals.error.connect(self.print_thread_error)
        if self.experiment_mode is None:
            self.ui.performing_embedding_label.setText(
                f"Embedding {len(img_pair)} images"
            )
        self.threadpool.start(img_embed_worker)
        self.threadpool.waitForDone(-1)
        if do_amg:
            self.segment_anything()

    def embed_img(self, img_name: str):
        current_ann_name = self.annotator.get_annotation_img_name()
        next_ann_name = self.annotator.get_next_annotation_img_name()

        img_to_embed = None

        if self.experiment_mode == "structured":
            img_to_embed = self.annotator.annotation.img
            delay = 0.0
        else:
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

        worker = Sam1EmbeddingWorker(
            sam_predictor=self.annotator.sam.predictor,
            img=img_to_embed,
            img_name=img_name,
            mutex=self.mutex,
            delay=delay,
        )
        worker.signals.result.connect(self.receive_embedding_from_thread)
        worker.signals.finished.connect(self.embedding_done)
        self.threadpool.start(worker)

        embedding_threads = self.threadpool.activeThreadCount()
        if self.experiment_mode is None:
            self.ui.performing_embedding_label.setText(
                f"Embedding {embedding_threads} images"
            )

    def start_mask_batch_thread(self, track_remaining: bool = False):

        if self.mask_track_batch_size is None:
            print("Propagating masks in main thread")
            self.annotator.sam.set_masks(self.annotator.annotation.good_masks)
            prop_mask_objs: list[MaskData] = self.annotator.sam.propagate_to_next_img()
            prop_mask_objs = self.annotator.convey_color_to_next_annot(prop_mask_objs)
            self.annotator.next_annotation.add_masks(prop_mask_objs, decision=True)
        else:
            # Batching of masks for propagation to next image
            unpropagated_masks = [
                mobj
                for mobj in self.annotator.annotation.good_masks
                if mobj.mid not in self.propagated_mids
            ]

            if len(unpropagated_masks) >= self.mask_track_batch_size or (
                track_remaining and len(unpropagated_masks) > 0
            ):
                self.threadpool.waitForDone(-1)
                self.propagated_mids.update(
                    {int(mobj.mid) for mobj in unpropagated_masks}
                )
                s2p_worker = Sam2PropagationWorker(
                    sam2_predictor=self.annotator.sam,
                    next_annotation=self.annotator.next_annotation,
                    unpropagated_masks=unpropagated_masks,
                    batch_size=self.mask_track_batch_size,
                    track_remaining=track_remaining,
                    mutex=self.mutex,
                )
                if self.experiment_mode is None:
                    self.ui.performing_embedding_label.setText(
                        f"Propagating {len(unpropagated_masks)} masks"
                    )
                s2p_worker.signals.finished.connect(self.propagation_done)
                s2p_worker.signals.error.connect(self.print_thread_error)
                self.threadpool.start(s2p_worker)

            if track_remaining:
                self.ui.create_loading_window("Propagating masks")

                while self.threadpool.activeThreadCount() > 0:
                    if self.mutex.tryLock(100):
                        self.ui.update_loading_window(
                            (
                                len(self.annotator.next_annotation.masks),
                                len(self.annotator.annotation.good_masks),
                            )
                        )
                        self.mutex.unlock()
                        time.sleep(0.1)
                self.threadpool.waitForDone(-1)
                self.ui.update_loading_window(100)
                self.ui.loading_window = None
                self._purge_falsely_propagated_masks()
                self.propagated_mids = set()

    def print_thread_error(self, error: tuple):
        exctype, value, traceback_str = error
        print(traceback_str)
        raise exctype(value)

    def _purge_falsely_propagated_masks(self):
        good_mask_ids = [mobj.mid for mobj in self.annotator.annotation.good_masks]
        bad_mask_ids = [
            mobj
            for mobj in self.annotator.next_annotation.masks
            if mobj.mid not in good_mask_ids
        ]
        self.annotator.next_annotation.masks = [
            mobj
            for mobj in self.annotator.next_annotation.masks
            if mobj.mid in good_mask_ids
        ]
        print(f"{len(bad_mask_ids)} falsely propagated masks have been purged")

    def embedding_done(self, img_name: str | int):
        if self.experiment_mode == "structured":
            return
        if isinstance(img_name, int):
            self.ui.performing_embedding_label.setText(
                f"{img_name} images successfully embedded"
            )
        else:
            print(f"Embedding of {img_name} done")
            embedding_threads = self.threadpool.activeThreadCount()
            if embedding_threads > 0:
                self.ui.performing_embedding_label.setText(
                    f"Embedding {embedding_threads} images"
                )
            else:
                self.ui.performing_embedding_label.setText(f"Embeddings done!")

    def propagation_done(self, maskn):
        if self.experiment_mode is None:
            self.ui.performing_embedding_label.setText(f"Propagated {maskn} masks")

    def receive_embedding_from_thread(self, result: tuple):
        if result[0] is None:
            self.segment_anything()
            return
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
        self.annotator.predict_with_sam(self.bbox_tracker)
        duration = time.time() - now
        print(f"SAM inference {duration}")

        now = time.time()
        first_mask_center = self.annotator.annotation.masks[0].center
        self.update_ui_imgs(center=first_mask_center)
        duration = time.time() - now
        print(f"update ui {duration}")
        if self.sam_gen == 2 and self.annotator.next_annotation is not None:
            self.start_mask_batch_thread()

    def _ui_config_changed(self, fields: list[str]):
        assert (
            len(fields) == 4
        ), f"Too many fields selected for visualization ({len(fields)}) expected 4"
        self.fields = fields
        self.update_ui_imgs()

    def update_ui_imgs(self, center: tuple | None | str = None):
        mviss = self.annotator.annotation.mask_visualizations
        for idx, field in enumerate(self.fields):
            self.ui.update_main_pix_map(idx=idx, img=getattr(mviss, field))

        if type(center) == tuple:
            self.ui.center_all_annotation_visualizers(center)

    def add_good_mask(self):
        if self.sam_gen == 2 and self.annotator.next_annotation is not None:
            self.start_mask_batch_thread()
        new_center = self.annotator.good_mask()
        if new_center is None:
            if (
                self.experiment_mode == "structured"
                and self.annotator.polygon_drawing_enabled == False
                and self.annotator.manual_annotation_enabled == False
            ):
                self.ui.create_message_box(
                    False,
                    "All proposed masks are done. Press the Next button if you want to continue with selecting masks interactively.",
                )
            print("All proposed masks are done")
            self.update_ui_imgs()

        else:
            self.update_ui_imgs(center=new_center)

    def add_bad_mask(self):
        new_center = self.annotator.bad_mask()
        if new_center is None:
            if (
                self.experiment_mode == "structured"
                and self.annotator.polygon_drawing_enabled == False
                and self.annotator.manual_annotation_enabled == False
            ):
                self.ui.create_message_box(
                    False,
                    "All proposed masks are done. Press the Next button if you want to continue with selecting masks interactively.",
                )
            print("All proposed masks are done")
            self.update_ui_imgs()
        else:
            self.update_ui_imgs(center=new_center)

    def previous_mask(self):
        new_center = self.annotator.step_back()
        self.annotator.update_collections(self.annotator.annotation)
        self.update_ui_imgs(center=new_center)

    def manual_annotation(self):
        self.annotator.toggle_manual_annotation()
        self.ui.draw_poly_button.setChecked(False)
        self.ui.delete_button.setChecked(False)
        self.ui.set_cursor(self.annotator.manual_annotation_enabled)
        self.update_ui_imgs()

    def draw_polygon(self):
        self.annotator.toggle_polygon_drawing()
        self.ui.manual_annotation_button.setChecked(False)
        self.ui.delete_button.setChecked(False)
        self.ui.set_cursor(self.annotator.polygon_drawing_enabled)
        self.update_ui_imgs()

    def select_masks_to_delete(self):

        # restoring previous state after mask deletion
        if self.annotator.previoius_toggle_state is not None:
            man_state = self.annotator.previoius_toggle_state["manual"]
            poly_state = self.annotator.previoius_toggle_state["polygon"]
        else:
            man_state, poly_state = False, False
        self.ui.manual_annotation_button.setChecked(man_state)
        self.ui.draw_poly_button.setChecked(poly_state)
        self.annotator.toggle_mask_deletion()
        self.update_ui_imgs()

    def manage_mouse_move(self, point: tuple[int]):
        height, width = self.annotator.annotation.img.shape[:2]
        if point[0] < 0 or point[0] >= width:
            return
        if point[1] < 0 or point[1] >= height:
            return

        current_time = time.time_ns()
        delta = current_time - self.last_sam_preview_time_stamp
        self.mouse_pos = point
        if delta * 1e-9 > 1 / self.manual_sam_preview_updates_per_sec:
            self.last_sam_preview_time_stamp = current_time
            if self.annotator.manual_annotation_enabled:
                self.annotator.predict_sam_manually(point)
            elif self.annotator.mask_deletion_enabled:
                self.annotator.highlight_mask_at_point(point)
            else:
                return

            self.update_ui_imgs()

    def manage_mouse_action(self, label: int):
        if (
            self.annotator.manual_annotation_enabled
            or self.annotator.polygon_drawing_enabled
        ):
            self.add_sam_preview_annotation_point(label)
        elif self.annotator.mask_deletion_enabled:
            self.delete_mask_at_point(label)
        else:
            return

    def add_sam_preview_annotation_point(self, label: int):
        if label == -1:
            if self.annotator.manual_mask_points:
                self.annotator.manual_mask_points.pop()
                self.annotator.manual_mask_point_labels.pop()
        else:
            self.annotator.manual_mask_points.append(self.mouse_pos)
            self.annotator.manual_mask_point_labels.append(label)
            if (
                len(self.annotator.manual_mask_points) == 1
                and self.experiment_mode == "polygon"
            ):
                self.annotator.init_time_stamp()

        if self.annotator.manual_annotation_enabled:
            if self.mutex.tryLock(100):
                self.annotator.predict_sam_manually(self.mouse_pos)
                self.mutex.unlock()
        elif self.annotator.polygon_drawing_enabled:
            self.annotator.mask_from_polygon()

        self.update_ui_imgs()

    def delete_mask_at_point(self, label: int):
        mid = self.annotator.highlight_mask_at_point(self.mouse_pos)
        if mid is None:
            return
        elif label == 1:
            self.annotator.delete_mask(mid)
            self.annotator.update_collections(self.annotator.annotation)
        self.update_ui_imgs()

    def next_indicated_polygon_img(self):
        print("Next polygon img")
        if len(self.annotator.annotation.good_masks) != 2:
            self.ui.create_message_box(False, "Please select exactly 2 masks")
            return
        self.select_next_img()
        self.ui.draw_poly_button.click()

    def next_method(self):
        print("Next method")
        if (
            not self.annotator.manual_annotation_enabled
            and not self.annotator.polygon_drawing_enabled
            and self.experiment_step == 0
        ):
            self.experiment_step = 1
            self.ui.performing_embedding_label.setText(
                f"Step 1/3: Select the good proposed masks"
            )
            # reset
            self.ui.manual_annotation_button.setDisabled(False)
            self.ui.manual_annotation_button.click()
            self.ui.manual_annotation_button.click()
            self.ui.manual_annotation_button.setDisabled(True)
            self.ui.good_mask_button.setDisabled(False)
            self.ui.bad_mask_button.setDisabled(False)
            self.ui.back_button.setDisabled(False)
            self.ui.delete_button.setDisabled(False)

        elif (
            not self.annotator.manual_annotation_enabled
            and not self.annotator.polygon_drawing_enabled
            and self.experiment_step == 1
        ):
            self.experiment_step = 2
            self.ui.performing_embedding_label.setText(
                f"Step 2/3: Annotate masks interactively with SAM{self.sam_gen}"
            )
            self.ui.manual_annotation_button.setDisabled(False)
            self.ui.manual_annotation_button.click()
            self.ui.manual_annotation_button.setDisabled(True)
        elif (
            self.annotator.manual_annotation_enabled
            and not self.annotator.polygon_drawing_enabled
            and self.experiment_step == 2
        ):
            self.experiment_step = 3
            self.ui.performing_embedding_label.setText(
                f"Step 3/3: Draw polygon masks for remaining objects"
            )
            self.ui.draw_poly_button.setDisabled(False)
            self.ui.draw_poly_button.click()
            self.ui.draw_poly_button.setDisabled(True)
        elif (
            not self.annotator.manual_annotation_enabled
            and self.annotator.polygon_drawing_enabled
            and self.experiment_step == 3
        ):
            self.experiment_step = 0
            self.select_next_img()
            self.ui.performing_embedding_label.setText(
                f"Step 0/3: Check whether all masks have been propagated correctly"
            )
            self.ui.good_mask_button.setDisabled(True)
            self.ui.bad_mask_button.setDisabled(True)
            self.ui.back_button.setDisabled(True)
            self.ui.delete_button.click()
            self.ui.delete_button.setDisabled(True)


# https://www.pythonguis.com/tutorials/multithreading-pyqt6-applications-qthreadpool/
class Sam1EmbeddingWorker(QRunnable):
    finished: pyqtSignal = pyqtSignal(str)
    error: pyqtSignal = pyqtSignal(tuple)
    result: pyqtSignal = pyqtSignal(tuple)

    def __init__(
        self,
        sam_predictor: BackgroundThreadSamPredictor,
        img: np.ndarray,
        img_name: str,
        mutex: QMutex,
        delay: float = 0.0,
    ):
        super().__init__()
        self.signals = WorkerSignals()
        self.sam_predictor = sam_predictor
        self.img = img
        self.img_name = img_name
        self.delay = delay
        self.mutex = mutex

    @pyqtSlot()
    def run(self):
        try:
            self.mutex.lock()
            now = time.time()
            embedding_result = self.sam_predictor.embed_img(self.img)
            if embedding_result is not None:
                features, original_size, input_size = embedding_result
                result = (features, original_size, input_size, self.img_name)
            else:
                result = (None, None, None, self.img_name)
            duration = time.time() - now
            print(f"embedding took {duration}")
            self.mutex.unlock()
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit(self.img_name)


class Sam2PropagationWorker(QRunnable):

    def __init__(
        self,
        sam2_predictor: Sam2Inference,
        next_annotation: AnnotationObject,
        unpropagated_masks: list[MaskData],
        batch_size: int,
        track_remaining: bool,
        mutex: QMutex,
    ):
        super().__init__()
        self.signals = Sam2WorkerSignals()
        self.sam2_predictor = sam2_predictor
        self.next_annotation = next_annotation
        self.unpropagated_masks = unpropagated_masks
        self.batch_size = batch_size
        self.track_remaining = track_remaining
        self.mutex = mutex

    @pyqtSlot()
    def run(self):
        try:
            for mask_batch_idx in range(
                0, len(self.unpropagated_masks), self.batch_size
            ):

                mask_batch = self.unpropagated_masks[
                    mask_batch_idx : mask_batch_idx + self.batch_size
                ]

                self.mutex.lock()
                assert len(mask_batch) > 0 and mask_batch[0].mask is not None

                now = time.time()
                mask_objs = self.sam2_predictor.prop_thread_func(mask_batch)
                self.next_annotation.add_masks(mask_objs, decision=True)
                duration = time.time() - now
                print(
                    f"Propagating of mask batch containig {len(mask_batch)} masks took {duration}"
                )
                self.mutex.unlock()
                time.sleep(
                    0.01
                )  # give some time to the main thread to access data for loading window

        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))

        finally:
            self.signals.finished.emit(len(self.unpropagated_masks))


class Sam2ImgPairEmbeddingWorker(QRunnable):

    def __init__(
        self,
        sam2: Sam2Inference,
        img_pair: list[np.ndarray],
        mutex: QMutex,
    ):
        super().__init__()
        self.signals = Sam2WorkerSignals()
        self.sam2 = sam2
        self.img_pair = img_pair
        self.mutex = mutex

    @pyqtSlot()
    def run(self):
        try:
            self.mutex.lock()
            now = time.time()
            self.sam2.set_features(self.img_pair)
            duration = time.time() - now
            print(f"Embedding images with Sam2 took {duration}")
            self.mutex.unlock()

        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))

        finally:
            self.signals.finished.emit(len(self.img_pair))


class WorkerSignals(QObject):
    finished: pyqtSignal = pyqtSignal(str)
    error: pyqtSignal = pyqtSignal(tuple)
    result: pyqtSignal = pyqtSignal(tuple)


class Sam2WorkerSignals(QObject):
    finished: pyqtSignal = pyqtSignal(int)
    error: pyqtSignal = pyqtSignal(tuple)
