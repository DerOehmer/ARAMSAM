"""SAM/YOLO inference and propagation orchestration."""
import time
from pathlib import Path
from aramsam_annotator.tracker import PanoImageAligner
from aramsam_annotator.mask_visualizations import MaskData
from aramsam_annotator.workers import (AMGWorker, Sam2ImgPairEmbeddingWorker, Sam1EmbeddingWorker, Sam2PropagationWorker, YoloPredicitonWorker)


class InferenceController:
    def __init__(self, context):
        self.context = context

    def propagate_good_masks(self):
        context = self.context
        if (context.annotator.annotation is None
                or not context.annotator.annotation.good_masks
                or context.annotator.next_annotation is None):
            return
        if context.sam_gen == 2:

            context.start_mask_batch_thread(track_remaining=True)

        elif context.sam_gen == 1:
            if context.bbox_tracker is None:
                context.bbox_tracker = PanoImageAligner()
            context.bbox_tracker.add_annotation(context.annotator.annotation)

        if context.sam_gen == 2:
            context.annotator.convey_color_to_next_annot(
                context.annotator.next_annotation.masks
            )
        if context.annotator.time_stamp is None:
            context.annotator.init_time_stamp()

    def embed_img_pair(self, do_amg=None):
        context = self.context
        if context.ui.auto_embed_box.isChecked() is False:
            context.start_user_annotation()
            return
        context.ui.manual_annotation_button.setEnabled(False)
        if context.annotator.next_annotation is not None:
            img_pair = [
                context.annotator.annotation.img,
                context.annotator.next_annotation.img,
            ]
        else:
            img_pair = [context.annotator.annotation.img]
        img_embed_worker = Sam2ImgPairEmbeddingWorker(
            context.annotator.sam, img_pair,
            (context.configs.do_amg or context.configs.yolo_model_ckpt_p is not None)
            if do_amg is None else do_amg, context.mutex
        )
        if context.experiment_mode is None:
            context.ui.performing_embedding_label.setText(
                f"Embedding {len(img_pair)} images"
            )
        self._submit(img_embed_worker, context.receive_sam2_embedding, context.embedding_done)

    def embed_img(self, img_name: str):
        context = self.context
        if context.ui.auto_embed_box.isChecked() is False:
            context.start_user_annotation()
            return
        candidates = (context.annotator.annotation, context.annotator.next_annotation)
        target = next((annotation for annotation in candidates
                       if annotation is not None and self._matches(annotation, img_name)), None)
        if target is None:
            raise ValueError(f"Embedding could not be matched to image: {img_name}")
        if target is context.annotator.next_annotation and not context.configs.sam_background_embedding:
            return
        if target is context.annotator.annotation:
            context.ui.manual_annotation_button.setEnabled(False)
        img_to_embed = target.img
        delay = 0.0
        # Use full path identities for real annotations; retain name-only API callers.
        identity = getattr(target, "filepath", target.img_name)
        worker = Sam1EmbeddingWorker(
            sam_predictor=context.annotator.sam.predictor,
            img=img_to_embed,
            img_name=identity,
            mutex=context.mutex,
            delay=delay,
        )
        self._submit(worker, context.receive_embedding_from_thread, context.embedding_done,
                     target=target)

        embedding_threads = context.threadpool.activeThreadCount()
        if context.experiment_mode is None:
            context.ui.performing_embedding_label.setText(
                f"Embedding {embedding_threads} images"
            )
            if target is context.annotator.annotation:
                context.ui.create_basic_loading_window(
                    text="Please wait... Processing image with SAM"
                )

    def start_mask_batch_thread(self, track_remaining: bool = False):

        context = self.context
        if context.mask_track_batch_size is None:
            print("Propagating masks in main thread")
            context.annotator.sam.set_masks(context.annotator.annotation.good_masks)
            prop_mask_objs: list[MaskData] = context.annotator.sam.propagate_to_next_img()
            prop_mask_objs = context.annotator.convey_color_to_next_annot(prop_mask_objs)
            context.annotator.next_annotation.add_masks(prop_mask_objs, decision=True)
        else:
            # Batching of masks for propagation to next image
            unpropagated_masks = [
                mobj
                for mobj in context.annotator.annotation.good_masks
                if mobj.mid not in context.propagated_mids
            ]

            if len(unpropagated_masks) >= context.mask_track_batch_size or (
                track_remaining and len(unpropagated_masks) > 0
            ):
                context.threadpool.waitForDone(-1)
                context.propagated_mids.update(
                    {int(mobj.mid) for mobj in unpropagated_masks}
                )
                s2p_worker = Sam2PropagationWorker(
                    sam2_predictor=context.annotator.sam,
                    next_annotation=context.annotator.next_annotation,
                    unpropagated_masks=unpropagated_masks,
                    batch_size=context.mask_track_batch_size,
                    track_remaining=track_remaining,
                    mutex=context.mutex,
                )
                if context.experiment_mode is None:
                    context.ui.performing_embedding_label.setText(
                        f"Propagating {len(unpropagated_masks)} masks"
                    )
                self._submit(s2p_worker, None, context.propagation_done)

            if track_remaining:
                context.ui.create_loading_window("Propagating masks")

                context.wait_for_workers()
                context.ui.update_loading_window(100)
                context.ui.loading_window = None
                context._purge_falsely_propagated_masks()
                context.propagated_mids = set()

    def print_thread_error(self, error: tuple):
        context = self.context
        _, value, traceback_str = error
        print(traceback_str)
        context.ui.close_basic_loading_window()
        context.ui.performing_embedding_label.setText("Processing failed")
        context.ui.create_message_box(True, str(value))

    def _purge_falsely_propagated_masks(self):
        context = self.context
        target = context.annotator.next_annotation
        if target is None:
            return
        good_ids = {mask.mid for mask in context.annotator.annotation.good_masks}
        retained = [(mask, decision) for mask, decision in zip(target.masks, target.mask_decisions)
                    if mask.mid in good_ids]
        target.masks = [mask for mask, _ in retained]
        target.mask_decisions = [decision for _, decision in retained]
        target.good_masks = [mask for mask in target.good_masks if mask.mid in good_ids]

    def embedding_done(self, img_name: str | int):
        context = self.context
        if context.experiment_mode in ["structured", "tutorial"]:
            return
        if isinstance(img_name, int):
            context.ui.performing_embedding_label.setText(
                f"{img_name} images successfully embedded"
            )
        else:
            print(f"Embedding of {img_name} done")
            embedding_threads = context.threadpool.activeThreadCount()
            if embedding_threads > 0:
                context.ui.performing_embedding_label.setText(
                    f"Embedding {embedding_threads} images"
                )
            else:
                context.ui.performing_embedding_label.setText(f"Embeddings done!")
                # Prefetch completion must not reset the current annotation timer.
                context.update_ui_imgs()

    def object_proposal_done(self, _):
        context = self.context
        context.ui.close_basic_loading_window()
        print("Object proposal done")

    def propagation_done(self, maskn):
        context = self.context
        if context.experiment_mode is None:
            context.ui.performing_embedding_label.setText(f"Propagated {maskn} masks")


    @staticmethod
    def _matches(annotation, identity):
        filepath = getattr(annotation, "filepath", None)
        if filepath is not None and str(identity) != Path(str(identity)).name:
            return Path(filepath).resolve() == Path(identity).resolve()
        return annotation.img_name == identity

    def receive_embedding_from_thread(self, result: tuple):
        context = self.context
        features, original_size, input_size, identity = result
        current = context.annotator.annotation
        following = context.annotator.next_annotation
        if current is not None and self._matches(current, identity):
            context.ui.close_basic_loading_window()
            if features is not None:
                current.set_sam_parameters(features=features, original_size=original_size, input_size=input_size)
                context.annotator.update_sam_features_to_current_annotation()
            context.propose_masks()
            if (context.experiment_mode is None and context.configs.sam_background_embedding
                    and following is not None and following.features is None):
                context.embed_img(getattr(following, "filepath", following.img_name))
        elif following is not None and self._matches(following, identity) and features is not None:
            following.set_sam_parameters(features=features, original_size=original_size, input_size=input_size)

    def receive_sam2_embedding(self, do_amg: bool):
        context = self.context
        context.ui.close_basic_loading_window()
        print("Will now do SAM2 embedding", do_amg)
        if do_amg:
            context.propose_masks()
        else:
            context.start_user_annotation()

    def propose_masks(self):
        context = self.context
        if getattr(context.annotator.annotation, "proposals_ready", False) is True:
            context.start_user_annotation()
        elif context.configs.do_amg:
            context.start_amg_worker()
        elif context.configs.yolo_model_ckpt_p is not None:
            context.start_yolo_worker()
        else:
            context.start_user_annotation()
        return

    def start_yolo_worker(self):
        context = self.context
        context.annotator.prepare_yolo()
        worker = YoloPredicitonWorker(context.annotator.yolo, context.mutex)
        self._submit(worker, context.receive_yolo_results, context.object_proposal_done)
        context.ui.create_basic_loading_window(
            text="Please wait... Running Prediction with YOLO model. If you see this\nyou propably don't have a GPU that is recognised by ARAMSAM"
        )

    def receive_yolo_results(self, result: list):
        context = self.context
        context.annotator.process_yolo_bboxes(result)
        context.annotator.annotation.proposals_ready = True
        context.update_ui_imgs()
        if context.annotator.time_stamp is None:
            context.annotator.init_time_stamp()

    def start_amg_worker(self):
        context = self.context
        context.annotator.prepare_amg(context.bbox_tracker)
        worker = AMGWorker(context.annotator.sam, context.mutex)
        self._submit(worker, context.receive_amg_results, context.object_proposal_done)
        context.ui.create_basic_loading_window(
            text="Please wait... This step can take up to 30 seconds"
        )

    def receive_amg_results(self, result: tuple):
        context = self.context
        mask_objs, annotated_image = result
        context.annotator.process_amg_masks(mask_objs, annotated_image)
        context.annotator.annotation.proposals_ready = True

        now = time.time()
        if len(context.annotator.annotation.masks) > 0:
            first_mask_center = context.annotator.annotation.masks[0].center
        else:
            first_mask_center = None
        context.update_ui_imgs(center=first_mask_center)
        duration = time.time() - now
        print(f"update ui {duration}")
        if context.sam_gen == 2 and context.annotator.next_annotation is not None:
            context.start_mask_batch_thread()
        context.start_user_annotation()

    def _submit(self, worker, result, finished, target=None):
        context = self.context
        annotator = context.annotator
        model = annotator.sam
        current, following = annotator.annotation, annotator.next_annotation

        def valid():
            if context.annotator is not annotator or annotator.sam is not model:
                return False
            if target is not None:
                return target is annotator.annotation or target is annotator.next_annotation
            return current is annotator.annotation and following is annotator.next_annotation

        return context.tasks.submit(worker, valid=valid, result=result,
                                    finished=finished, error=context.print_thread_error)
