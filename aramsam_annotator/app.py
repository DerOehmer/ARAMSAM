from functools import cached_property
from aramsam_annotator.controllers.navigation import NavigationController
from aramsam_annotator.controllers.inference import InferenceController
from aramsam_annotator.controllers.interaction import InteractionController
from aramsam_annotator.controllers.experiments import ExperimentController
from aramsam_annotator.controllers.presentation import PresentationController
import sys
import inspect
import traceback
import time
import qdarkstyle
import shutil
import tempfile
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThreadPool, QMutex, QEventLoop, QTimer

from aramsam_annotator.gui import UserInterface
from aramsam_annotator.annotator import Annotator
from aramsam_annotator.configs import AramsamConfigs
from aramsam_annotator.backend.tasks import TaskDispatcher
from aramsam_annotator.workers import BackendWorker, ModelLoadingWorker


class App:
    """Composition root and stable callback API for the Qt interface.

    Controllers coordinate workflows; Annotator owns annotation state. Explicit
    forwarding methods preserve integrations without dynamic attribute proxies.
    """

    def __init__(
        self,
        ui_options: dict = None,
        experiment_mode: str = None,
        experiment_progress: tuple = None,
        configs: AramsamConfigs = None,
    ) -> None:
        self.configs = configs if configs is not None else AramsamConfigs()
        if ui_options is None:
            from aramsam_annotator.main import create_vis_options
            options, defaults, current = create_vis_options(self.configs)
            ui_options = {"layout_settings_options": {
                "options": options, "default": defaults, "current": current,
            }}
        self.temp_dir = tempfile.mkdtemp(prefix="aramsam-")
        self.application = QApplication.instance() or QApplication([])
        self.application.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
        ui_options["class"] = self.configs.class_dict
        self.ui = UserInterface(ui_options=ui_options, experiment_mode=experiment_mode)
        self.annotator = Annotator(self.configs)
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(1)
        self.img_fnames = []
        self.output_dir = None
        self.experiment_mode = experiment_mode
        self.experiment_progress = experiment_progress
        self.tutorial_flag = False

        self.ui.auto_save_box.setChecked(self.configs.save_data.auto_save)
        self.ui.mask_propagation_box.setChecked(self.configs.mask_propagation)
        self._connect_ui(self.ui.good_mask_button.clicked, self.add_good_mask)
        self._connect_ui(self.ui.bad_mask_button.clicked, self.add_bad_mask)
        self._connect_ui(self.ui.back_button.clicked, self.previous_mask)
        self._connect_ui(self.ui.manual_annotation_button.clicked, self.manual_annotation)
        self._connect_ui(self.ui.draw_button.clicked, self.draw_polygon)
        self._connect_ui(self.ui.delete_button.clicked, self.select_masks_to_delete)
        self._connect_ui(self.ui.auto_save_box.checkStateChanged, self.auto_save_changed)
        self._connect_ui(self.ui.auto_embed_box.checkStateChanged, self.auto_embed_changed)
        self._connect_ui(self.ui.mask_propagation_box.checkStateChanged, self.mask_propagation_changed)

        if self.experiment_mode == "structured":
            self._connect_ui(self.ui.next_method_button.clicked, self.next_method)
            self.proposed_masks_instructions()

        elif self.experiment_mode == "polygon":
            self._connect_ui(self.ui.next_method_button.clicked, self.next_indicated_polygon_img)

        elif self.experiment_mode is None:
            self._connect_ui(self.ui.next_img_button.clicked, self.select_next_img)
            self._connect_ui(self.ui.previous_img_button.clicked, self.select_previous_img)

        if self.experiment_mode is not None:
            self.experiment_step: int = 1
            self.experiment_progress = experiment_progress

        self._connect_ui(self.ui.mouse_position, self.manage_mouse_move)
        self._connect_ui(self.ui.load_img_signal, self.load_img)
        self._connect_ui(self.ui.load_img_folder_signal, self.load_img_folder)
        self._connect_ui(self.ui.output_dir_signal, self.change_output_dir)
        self._connect_ui(self.ui.sam_path_signal, self.changed_sam_model)
        self._connect_ui(self.ui.save_signal, self.save_output)
        self._connect_ui(self.ui.preview_annotation_point_signal, self.manage_mouse_action)
        self._connect_ui(self.ui.layout_options_signal, self._ui_config_changed)
        self._connect_ui(self.ui.shutdown_signal, self.shutdown)
        self._connect_ui(self.ui.text_prompt_signal, self.start_text_prompt)
        self.ui.set_text_prompt_available(self.configs.sam_configs.gen == 3)
        self.ui.text_prompt_confidence.setValue(self.configs.sam_configs.confidence_threshold)

        self.fields = ui_options["layout_settings_options"]["default"]

        self.manual_sam_preview_updates_per_sec: int = 10
        self.last_sam_preview_time_stamp: int = time.time_ns()
        self.bbox_tracker: object = None
        self.sam_gen: int = None

        self.mask_track_batch_size: int = 10
        self.propagated_mids: set[int] = set()
        self.mutex = QMutex()
        self.mouse_pos = None
        self._model_settings = None

    def _connect_ui(self, signal, callback):
        """Catch UI callback errors at the Qt boundary, where exceptions are fatal."""
        parameter_count = len(inspect.signature(callback).parameters)

        def guarded(*args):
            try:
                return callback(*args[:parameter_count])
            except Exception as error:
                self.print_thread_error((type(error), error, traceback.format_exc()))

        signal.connect(guarded)

    def run(self) -> None:
        self.ui.run()
        sys.exit(self.application.exec())

    @cached_property
    def tasks(self):
        return TaskDispatcher(self.threadpool)

    def wait_for_workers(self, message="Processing images...", delay_ms=200):
        if "tasks" in self.__dict__:
            self._wait_with_loading(message, delay_ms)
        else:
            self.threadpool.waitForDone(-1)

    def _wait_with_loading(self, message, delay_ms):
        window = None
        timer = QTimer()
        timer.setSingleShot(True)

        def show_loading():
            nonlocal window
            if self.ui.basic_loading_window is None:
                self.ui.create_basic_loading_window(text=message)
                window = self.ui.basic_loading_window

        timer.timeout.connect(show_loading)
        timer.start(delay_ms)
        try:
            self.tasks.wait()
        finally:
            timer.stop()
            if window is not None and self.ui.basic_loading_window is window:
                self.ui.close_basic_loading_window()

    def run_backend_task(self, operation, message="Processing images...", delay_ms=200):
        """Run non-Qt work off the UI thread, preserving synchronous callers."""
        results, errors = [], []
        self.tasks.submit(BackendWorker(operation), result=lambda value: results.append(value[0]),
                          error=lambda error: errors.append(error[1]))
        self._wait_with_loading(message, delay_ms)
        if errors:
            raise errors[0]
        return results[0] if results else None

    def shutdown(self, _=None) -> None:
        if "tasks" in self.__dict__:
            self.tasks.close()
        else:
            self.threadpool.waitForDone(-1)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def set_sam(self):
        if self.experiment_mode == "polygon":
            return
        generation = self.sam_gen or self.configs.sam_configs.gen
        background = self.configs.sam_background_embedding if self.experiment_mode is None else False
        settings = (generation, background, self.configs.sam_configs.model_ckpt_p,
                    self.configs.sam_configs.model_type, self.annotator.device,
                    self.configs.sam_configs.bpe_path, self.configs.sam_configs.confidence_threshold)
        if self.annotator.sam is not None and settings == getattr(self, "_model_settings", None):
            return
        self.wait_for_workers()
        self.annotator.sam_ckpt = self.configs.sam_configs.model_ckpt_p
        self.annotator.sam_model_type = self.configs.sam_configs.model_type
        self.ui.create_basic_loading_window(
            text=f"Loading SAM{generation} model. Initial startup may take a while...",
        )
        # Preserve set_sam's synchronous API for navigation and model switching,
        # while Qt continues painting and animating the modal loading dialog.
        loop = QEventLoop()
        errors = []

        def failed(error):
            errors.append(error[1])
            loop.quit()

        try:
            worker = ModelLoadingWorker(self.annotator, generation, background)
            # Completion must also quit the local loop if shutdown invalidates
            # the dispatcher's normal result callbacks during initialization.
            worker.signals.finished.connect(loop.quit)
            self.tasks.submit(worker, error=failed)
            loop.exec(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            # Release the receiver even when its error signal quit the loop
            # before the worker's completion signal was delivered.
            self.tasks.drain()
            if errors:
                raise errors[0]
        finally:
            self.ui.close_basic_loading_window()
        if generation == 2 and self.annotator.device == "cuda":
            # SAM2 establishes thread-local autocast during construction. Its UI
            # predictions need the same precision context on the GUI thread.
            self.annotator.sam._init_mixed_precision()
        self.sam_gen = generation
        self.ui.set_text_prompt_available(generation == 3)
        self._model_settings = settings
        for annotation in (self.annotator.annotation, self.annotator.next_annotation):
            if annotation is not None:
                annotation.set_sam_parameters(None, None, None)

    def auto_save_changed(self):
        box_checked = self.ui.auto_save_box.isChecked()
        if box_checked:
            self.configs.save_data.auto_save = True
        else:
            self.configs.save_data.auto_save = False

    def mask_propagation_changed(self):
        self.configs.mask_propagation = self.ui.mask_propagation_box.isChecked()
        if self.configs.mask_propagation:
            return
        # Finish any running batch before removing its unvisited-image results.
        # Keep masks that came from disk or another annotation source.
        self.wait_for_workers()
        self.bbox_tracker = None
        target = self.annotator.next_annotation
        if target is not None:
            def generated(mask):
                return (mask.mid in self.propagated_mids
                        and mask.origin in ("Sam2_tracking", "Sam3_tracking"))

            retained = [(mask, decision) for mask, decision in zip(target.masks, target.mask_decisions)
                        if not generated(mask)]
            target.masks = [mask for mask, _ in retained]
            target.mask_decisions = [decision for _, decision in retained]
            target.good_masks = [mask for mask in target.good_masks if not generated(mask)]
        self.propagated_mids = set()

    def auto_embed_changed(self):
        if self.annotator.annotation is None:
            return
        box_checked = self.ui.auto_embed_box.isChecked()

        if box_checked:
            self.ui.manual_annotation_button.setEnabled(True)
            if self.sam_gen == 2:
                self.embed_img_pair(do_amg=False)
            else:
                self.embed_img(self.annotator.get_annotation_img_name())
        else:
            self.ui.manual_annotation_button.setChecked(False)
            if self.annotator.manual_annotation_enabled:
                self.manual_annotation()
            self.ui.manual_annotation_button.setEnabled(False)

    def save_output(self, _=None):
        return self.navigation.save_output(_)

    def change_output_dir(self, out_dir: str):
        return self.navigation.change_output_dir(out_dir)

    def changed_sam_model(self, model_path: str):
        if not model_path:
            return
        model_type = next((kind for kind in ("vit_b", "vit_h", "vit_l") if kind in model_path), None)
        generation = 3 if "sam3" in Path(model_path).name.lower() else 1
        if model_type is None and generation != 3:
            self.ui.create_message_box(True, "Choose a SAM1 vit_b/vit_h/vit_l checkpoint or sam3.pt. Configure SAM2 in configs.yaml.")
            return
        self.wait_for_workers()
        old_settings = (self.configs.sam_configs.gen, self.configs.sam_configs.model_ckpt_p,
                        self.configs.sam_configs.model_type, self.sam_gen)
        self.configs.sam_configs.gen = self.sam_gen = generation
        self.configs.sam_configs.model_ckpt_p = model_path
        self.configs.sam_configs.model_type = model_type
        try:
            self.set_sam()
        except Exception:
            (self.configs.sam_configs.gen, self.configs.sam_configs.model_ckpt_p,
             self.configs.sam_configs.model_type, self.sam_gen) = old_settings
            raise
        if self.annotator.annotation is not None:
            self.embed_img(self.annotator.annotation.img_name)

    def start_text_prompt(self, prompt: str, confidence: float):
        return self.inference.start_text_prompt(prompt, confidence)

    def load_img(self, _) -> None:
        return self.navigation.load_img(_)

    def _is_img_file(self, path: str) -> bool:
        return self.navigation._is_img_file(path)

    def load_img_folder(self, _) -> None:
        return self.navigation.load_img_folder(_)

    def _pop_img_fnames(self) -> tuple[Path, Path]:
        return self.navigation._pop_img_fnames()

    def select_previous_img(self):
        return self.navigation.select_previous_img()

    def select_next_img(self):
        return self.navigation.select_next_img()

    def start_user_annotation(self):
        return self.presentation.start_user_annotation()

    def check_annotations_done(
        self, img_name: str, next_img_name: str
    ) -> tuple[bool, bool]:

        return self.navigation.check_annotations_done(img_name, next_img_name)

    def _annot_log_exists(self, img_name: str | None) -> bool:
        return self.navigation._annot_log_exists(img_name)

    def load_previous_annotations(self, img_name: str, next_img_name: str):
        return self.navigation.load_previous_annotations(img_name, next_img_name)

    def propagate_good_masks(self):
        return self.inference.propagate_good_masks()

    def embed_img_pair(self, do_amg=None):
        return self.inference.embed_img_pair(do_amg)

    def embed_img(self, img_name: str):
        return self.inference.embed_img(img_name)

    def start_mask_batch_thread(self, track_remaining: bool = False):

        return self.inference.start_mask_batch_thread(track_remaining)

    def print_thread_error(self, error: tuple):
        return self.inference.print_thread_error(error)

    def _purge_falsely_propagated_masks(self):
        return self.inference._purge_falsely_propagated_masks()

    def embedding_done(self, img_name: str | int):
        return self.inference.embedding_done(img_name)

    def object_proposal_done(self, _):
        return self.inference.object_proposal_done(_)

    def propagation_done(self, maskn):
        return self.inference.propagation_done(maskn)

    def receive_embedding_from_thread(self, result: tuple):
        return self.inference.receive_embedding_from_thread(result)

    def receive_sam2_embedding(self, do_amg: bool):
        return self.inference.receive_sam2_embedding(do_amg)

    def propose_masks(self):
        return self.inference.propose_masks()

    def start_yolo_worker(self):
        return self.inference.start_yolo_worker()

    def receive_yolo_results(self, result: list):
        return self.inference.receive_yolo_results(result)

    def start_amg_worker(self):
        return self.inference.start_amg_worker()

    def receive_amg_results(self, result: tuple):
        return self.inference.receive_amg_results(result)

    def _ui_config_changed(self, fields: list[str]):
        return self.presentation._ui_config_changed(fields)

    def update_ui_imgs(self, center: tuple | None | str = None):
        return self.presentation.update_ui_imgs(center)

    def add_good_mask(self):
        return self.interaction.add_good_mask()

    def add_bad_mask(self):
        return self.interaction.add_bad_mask()

    def previous_mask(self):
        return self.interaction.previous_mask()

    def proposed_masks_instructions(self):
        return self.interaction.proposed_masks_instructions()

    def manual_annotation(self):
        return self.interaction.manual_annotation()

    def draw_polygon(self):
        return self.interaction.draw_polygon()

    def select_masks_to_delete(self):

        # restoring previous state after mask deletion
        return self.interaction.select_masks_to_delete()

    def manage_mouse_move(self, point: tuple[int]):
        return self.interaction.manage_mouse_move(point)

    def manage_mouse_action(self, label: int):
        return self.interaction.manage_mouse_action(label)

    def add_sam_preview_annotation_point(self, label: int):
        return self.interaction.add_sam_preview_annotation_point(label)

    def delete_mask_at_point(self, label: int):
        return self.interaction.delete_mask_at_point(label)

    def next_indicated_polygon_img(self):
        return self.experiments.next_indicated_polygon_img()

    def next_method(self):
        return self.experiments.next_method()

    @cached_property
    def navigation(self):
        return NavigationController(self)

    @cached_property
    def inference(self):
        return InferenceController(self)

    @cached_property
    def interaction(self):
        return InteractionController(self)

    @cached_property
    def experiments(self):
        return ExperimentController(self)

    @cached_property
    def presentation(self):
        return PresentationController(self)
