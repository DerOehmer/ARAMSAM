"""Image selection and persistence workflows for the existing interface."""
from pathlib import Path

from natsort import natsorted

from aramsam_annotator.backend.session import ImageQueue, image_key
from aramsam_annotator.backend.storage import AnnotationRepository


class NavigationController:
    def __init__(self, context):
        self.context = context
        self.changing_image = False
        self.image_history = []
        self.revisit_images = set()

    @property
    def repository(self):
        return AnnotationRepository(self.context.configs.save_data)

    def save_output(self, _=None):
        app = self.context
        if (app.experiment_mode == 'tutorial' or app.tutorial_flag
                or app.annotator.annotation is None or not app.configs.save_data.do_save):
            return
        if app.output_dir is None or not Path(app.output_dir).is_dir():
            app.ui.open_ouput_dir_selection()
        # A cancelled directory chooser must not recurse through ui.save().
        if app.output_dir is None or not Path(app.output_dir).is_dir():
            return
        suffix = None
        if app.experiment_mode == 'structured':
            suffix = f'structured_sam{app.sam_gen}'
        elif app.experiment_mode == 'polygon':
            suffix = 'polygon'
        app.annotator.save_annotations(app.output_dir, save_suffix=suffix)

    def change_output_dir(self, out_dir):
        if out_dir:
            self.context.output_dir = out_dir

    def load_img(self, _):
        app = self.context
        path = app.ui.open_img_load_file_dialog()
        if not path:
            return
        app.wait_for_workers()
        app.set_sam()
        app.img_fnames.append(path)
        app.select_next_img()

    def _is_img_file(self, path):
        path = Path(path)
        return path.is_file() and path.suffix.lower() in ('.png', '.jpg', '.jpeg')

    def load_img_folder(self, _):
        app = self.context
        folder = app.ui.open_load_folder_dialog()
        if app.experiment_mode not in ('tutorial', 'structured', 'polygon'):
            if not folder:
                return
            images = [str(path) for path in Path(folder).iterdir() if self._is_img_file(path)]
            if not images:
                return
            app.img_fnames = natsorted([*app.img_fnames, *images])
        app.wait_for_workers()
        app.set_sam()
        app.select_next_img()

    def _pop_img_fnames(self):
        app = self.context
        return ImageQueue(app.img_fnames, app.temp_dir, app.configs.img_tiles).pop_pair()

    def select_previous_img(self):
        app = self.context
        if self.changing_image or len(self.image_history) < 2:
            return
        self.changing_image = True
        try:
            app.wait_for_workers()
            if app.configs.save_data.auto_save:
                app.ui.save()
            else:
                app.ui.open_save_annots_box()
            image = self.image_history[-2]
            current = self.image_history[-1]
            # Read a fresh record so disk annotations replace model proposals.
            from aramsam_annotator.backend.annotations import AnnotationObject
            annotation = AnnotationObject(image)
            if self.repository.exists(app.output_dir, image):
                self.repository.load(annotation, app.output_dir, app.annotator.mask_id_handler)
            app.annotator.next_annotation = app.annotator.annotation
            app.annotator.annotation = annotation
            embed_current, embed_next = app.annotator.create_new_annotation(image, current)
            app.img_fnames.append(str(current))
            self.revisit_images.add(image_key(current))
            self.image_history.pop()
            app.bbox_tracker = None
            app.propagated_mids = set()
            app.ui.reset_ui()
            app.ui.enable_push_buttons()
            self._update_previous_button()
            self._activate_image(image, current, embed_current, embed_next)
        finally:
            self.changing_image = False

    def _update_previous_button(self):
        app = self.context
        if app.experiment_mode is None:
            app.ui.previous_img_button.setEnabled(len(self.image_history) > 1)

    def select_next_img(self):
        if self.changing_image:
            return
        self.changing_image = True
        try:
            return self._select_next_img()
        finally:
            self.changing_image = False

    def _select_next_img(self):
        app = self.context
        app.wait_for_workers()
        if not app.img_fnames:
            self._finish_queue()
            return
        if app.annotator.annotation is not None:
            if app.configs.save_data.auto_save:
                app.ui.save()
            else:
                app.ui.open_save_annots_box()

        # Iteration handles large completed folders without growing the call stack.
        while app.img_fnames:
            queue_before = list(app.img_fnames)
            try:
                image, following = app._pop_img_fnames()
                revisiting = image_key(image) in self.revisit_images
                current_done, next_done = ((False, False) if revisiting
                                           else app.check_annotations_done(image, following))
                if current_done and app.experiment_mode == 'structured':
                    raise ValueError('This directory is already occupied with annotations')
                if current_done and next_done:
                    continue
                if current_done:
                    # Restore the final completed predecessor for mask propagation.
                    app.load_previous_annotations(image, following)
                    app.wait_for_workers()
                    image, following = app._pop_img_fnames()
                previous_next = app.annotator.next_annotation
                previous_path = (getattr(previous_next, 'filepath', None)
                                 or getattr(previous_next, 'img_name', None))
                if not revisiting and previous_path is not None and image_key(previous_path) == image_key(image):
                    app.propagate_good_masks()
                else:
                    app.bbox_tracker = None
                embed_current, embed_next = app.annotator.create_new_annotation(image, following)
                if revisiting and self.repository.exists(app.output_dir, image):
                    self.repository.load(app.annotator.annotation, app.output_dir, app.annotator.mask_id_handler)
                    app.annotator.mask_idx = len(app.annotator.annotation.masks)
            except Exception:
                app.img_fnames[:] = queue_before
                raise
            app.ui.reset_ui()
            app.ui.enable_push_buttons()
            app.annotator.reset_toggles(toggles_only=True)
            app.propagated_mids = set()
            self.image_history.append(image)
            self.revisit_images.discard(image_key(image))
            self._update_previous_button()
            self._activate_image(image, following, embed_current, embed_next)
            return
        self._finish_queue()

    def _activate_image(self, image, following, embed_current, embed_next):
        app = self.context
        if app.sam_gen == 2:
            if getattr(app.annotator.annotation, "proposals_ready", False) is True:
                app.embed_img_pair(do_amg=False)
            else:
                app.embed_img_pair()
        elif app.sam_gen == 1:
            if app.experiment_mode == 'tutorial':
                app.ui.close_basic_loading_window()
                app.start_user_annotation()
            elif (embed_current or app.annotator.annotation.features is None
                  or app.experiment_mode == 'structured'
                  or not app.configs.sam_background_embedding):
                app.embed_img(str(image))
            else:
                app.annotator.update_sam_features_to_current_annotation()
                app.propose_masks()
                if (embed_next and following is not None and app.experiment_mode is None
                        and app.configs.sam_background_embedding):
                    app.embed_img(str(following))
        elif app.experiment_mode == 'polygon':
            app.ui.performing_embedding_label.setText('Draw 3 polygon masks at the indicated kernels')
            app.ui.close_basic_loading_window()
            app.start_user_annotation()

    def _finish_queue(self):
        app = self.context
        app.ui.disable_push_buttons()
        self._update_previous_button()
        if app.experiment_mode in ('structured', 'polygon'):
            app.ui.save()
            if app.experiment_progress is not None and app.experiment_progress[0] == app.experiment_progress[1]:
                app.ui.create_message_box(
                    False,
                    'Congratulations! You have finished the experiment. Thank you for your participation! Tell the experiment supervisor that you are done and click Yes.',
                    wait_for_user=True,
                )
            app.ui.close()
            return
        message = ('No more images left in current directory. Check output directory whether '
                   'annotations were already done. Load a new folder/image to proceed')
        if not app.configs.save_data.auto_save:
            message += '\nClick Yes to save current annotations.'
        reply = app.ui.create_message_box(False, message, wait_for_user=True)
        if app.annotator.annotation is not None and (app.configs.save_data.auto_save or reply):
            app.ui.save()

    def check_annotations_done(self, img_name, next_img_name):
        app = self.context
        if app.output_dir is None:
            app.ui.open_ouput_dir_selection()
        return app._annot_log_exists(img_name), app._annot_log_exists(next_img_name)

    def _annot_log_exists(self, img_name):
        return self.repository.exists(self.context.output_dir, img_name)

    def load_previous_annotations(self, img_name, next_img_name):
        app = self.context
        app.annotator.create_new_annotation(img_name, next_img_name)
        self.repository.load(app.annotator.annotation, app.output_dir, app.annotator.mask_id_handler)
        app.annotator.mask_idx = len(app.annotator.annotation.masks)
        if app.sam_gen == 2:
            app.embed_img_pair(do_amg=False)
