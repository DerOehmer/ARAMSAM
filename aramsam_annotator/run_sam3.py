"""SAM3 adapter for ARAMSAM's image, proposal, and tracking workflows.

SAM3 stays optional: its package is imported only when generation 3 is selected.
Image states are cached per annotation, independently of the active predictor.
"""
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from aramsam_annotator.mask_visualizations import MaskData
from aramsam_annotator.run_sam import DefaultAMG, load_amg_config


class Sam3Inference:
    interactive_origin = "Sam3_interactive"
    proposed_origin = "Sam3_proposed"

    def __init__(self, mask_id_handler, sam3_checkpoint="sam3.pt", device=None,
                 background_embedding=True, bpe_path=None, confidence_threshold=0.5,
                 amg_config_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if torch.device(self.device).type != "cuda":
            raise RuntimeError(
                "The installed SAM3 package requires CUDA. Set use_gpu: true and "
                "ensure PyTorch can access an NVIDIA GPU, or select SAM1/SAM2."
            )
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.mask_id_handler = mask_id_handler
        self.background_embedding = background_embedding
        # Support the local checkout used by _test_sam3.py, as well as an
        # installed package whose builder supplies the vocabulary itself.
        local_bpe = Path(__file__).resolve().parents[1] / "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        if bpe_path is None and local_bpe.is_file():
            bpe_path = str(local_bpe)
        self.model = build_sam3_image_model(
            checkpoint_path=sam3_checkpoint, bpe_path=bpe_path,
            device=self.device, enable_inst_interactivity=True,
            load_from_HF=False,
        )
        self.processor = Sam3Processor(self.model, device=self.device,
                                       confidence_threshold=confidence_threshold)
        self.tracker = self.model.inst_interactive_predictor.model
        # The upstream tracker constructor enters a persistent autocast context.
        # Restore the caller's context; this adapter scopes precision per call.
        self.tracker.bf16_context.__exit__(None, None, None)
        self.predictor = self  # Existing editor and embedding-worker interface.
        self.state = None
        self.is_image_set = False
        self._next_annotation = None
        self.inference_state = None
        self.amg = DefaultAMG(self, Sam3AutomaticMaskGenerator(self, **(load_amg_config(amg_config_path) or {})))

    def _precision(self):
        # Autocast is thread-local; enter it for every worker and UI prediction.
        if torch.device(self.device).type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    @torch.inference_mode()
    def embed_img(self, image, image_format="RGB"):
        if image_format not in ("RGB", "BGR"):
            raise ValueError("image_format must be RGB or BGR")
        rgb = image[..., ::-1] if image_format == "BGR" else image
        pil_image = Image.fromarray(np.ascontiguousarray(rgb))
        with self._precision():
            state = self.processor.set_image(pil_image)
            # The tracker needs both the backbone features and normalized pixels
            # for memory encoding. Retain these without running another backbone.
            pixels = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).to(self.device)
            state["tracking_image"] = self.processor.transform(pixels).unsqueeze(0)
        return state, image.shape[:2], image.shape[:2]

    def set_features(self, features, original_size=None, input_size=None):
        self.state = features
        self.is_image_set = features is not None
        self.inference_state = None
        self._next_annotation = None

    def reset_image(self):
        self.set_features(None)

    reset_img = reset_image

    def _require_image(self):
        if not self.is_image_set:
            raise ValueError("Image embedding has not been initialized yet")

    @staticmethod
    def _numpy(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            return value.numpy()
        return np.asarray(value)

    @classmethod
    def _mask(cls, value):
        array = cls._numpy(value)
        return (array.reshape(array.shape[-2:]) > 0).astype(np.uint8) * 255

    def _prepare_mask(self, mask):
        mask = self._numpy(mask)
        # Full-resolution binary masks are accepted by ARAMSAM. SAM3 expects
        # low-resolution logits; preserve logits supplied by an API caller.
        if mask.ndim == 2:
            size = self.tracker.sam_prompt_encoder.mask_input_size
            mask = cv2.resize((mask > 0).astype(np.float32), tuple(size[::-1]))
            mask = (mask * 20 - 10)[None]
        return mask.astype(np.float32)

    @torch.inference_mode()
    def _predict(self, pts=None, pts_labels=None, bboxes=None, mask_input=None,
                 multimask_output=False):
        self._require_image()
        if all(value is None for value in (pts, bboxes, mask_input)):
            raise ValueError("Provide points, boxes, or a mask")
        if (pts is None) != (pts_labels is None):
            raise ValueError("Points and labels must be provided together")
        with self._precision():
            return self.model.predict_inst(
                self.state,
                point_coords=None if pts is None else self._numpy(pts).astype(np.float32),
                point_labels=None if pts_labels is None else self._numpy(pts_labels).astype(np.int32),
                box=None if bboxes is None else self._numpy(bboxes).astype(np.float32),
                mask_input=None if mask_input is None else self._prepare_mask(mask_input),
                multimask_output=multimask_output,
            )

    def predict(self, pts=None, pts_labels=None, bboxes=None, mask_input=None):
        masks, _, _ = self._predict(pts, pts_labels, bboxes, mask_input)
        mask = self._mask(masks[0])
        return mask if np.any(mask) else None

    def predict_batch(self, pts=None, pts_labels=None, bboxes=None, mask_input=None):
        inputs = [value for value in (pts, pts_labels, bboxes, mask_input) if value is not None]
        if not inputs:
            raise ValueError("Provide points, boxes, or masks")
        count = len(inputs[0])
        if any(len(value) != count for value in inputs):
            raise ValueError("Prompt batches must have the same length")
        masks = []
        for index in range(count):
            result, _, _ = self._predict(
                None if pts is None else pts[index],
                None if pts_labels is None else pts_labels[index],
                None if bboxes is None else bboxes[index],
                None if mask_input is None else mask_input[index],
            )
            masks.append((self._mask(result[0]) > 0)[None])
        return masks

    def _prep_batch_pts_and_lbls(self, pts, lbls, shp):
        return self._numpy(pts).astype(np.float32), self._numpy(lbls).astype(np.int32)

    def transform_bboxes(self, bboxes, shp):
        return self._numpy(bboxes).astype(np.float32)

    def _torch_to_npmasks(self, masks):
        return [mask for value in masks if np.any(mask := self._mask(value))]

    @torch.inference_mode()
    def predict_text(self, prompt, confidence_threshold=None):
        self._require_image()
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Enter a text prompt")
        if confidence_threshold is not None:
            if not 0 <= confidence_threshold <= 1:
                raise ValueError("Confidence must be between 0 and 1")
            self.processor.set_confidence_threshold(confidence_threshold)
        with self._precision():
            # Prompt-specific state must not contaminate cached image embeddings.
            state = dict(self.state, backbone_out=dict(self.state["backbone_out"]))
            self.processor.reset_all_prompts(state)
            output = self.processor.set_text_prompt(state=state, prompt=prompt)
        return self._torch_to_npmasks(output["masks"])

    def set_tracking_pair(self, current_annotation, next_annotation):
        if self._next_annotation is not next_annotation:
            self.inference_state = None
        self._next_annotation = next_annotation

    @torch.inference_mode()
    def set_masks(self, mask_objs):
        self._require_image()
        if torch.device(self.device).type != "cuda":
            raise RuntimeError("SAM3 mask tracking requires CUDA in the installed SAM3 package")
        if self._next_annotation is None:
            raise ValueError("No next image available for tracking")
        with self._precision():
            if self.inference_state is None:
                following = self._next_annotation.features
                if following is None:
                    following, _, _ = self.embed_img(self._next_annotation.img, "BGR")
                height, width = self.state["original_height"], self.state["original_width"]
                # SAM3's tracker uses a single video resolution. The resulting
                # second-frame masks are resized to that image's actual shape below.
                cache = {index: (state["tracking_image"], state["backbone_out"]["sam2_backbone_out"])
                         for index, state in enumerate((self.state, following))}
                self.inference_state = self.tracker.init_state(
                    video_height=height, video_width=width, num_frames=2,
                    cached_features=cache,
                )
            self.tracker.clear_all_points_in_video(self.inference_state)
            self._tracking_objects = {obj.mid: obj for obj in mask_objs}
            for obj in mask_objs:
                self.tracker.add_new_mask(self.inference_state, 0, obj.mid,
                                          torch.as_tensor(obj.mask > 0))

    @torch.inference_mode()
    def propagate_to_next_img(self):
        result = []
        with self._precision():
            for frame, ids, _, masks, _ in self.tracker.propagate_in_video(
                self.inference_state, start_frame_idx=0, max_frame_num_to_track=1,
                reverse=False, propagate_preflight=True,
            ):
                if frame != 1:
                    continue
                height, width = self._next_annotation.img.shape[:2]
                for mid, value in zip(ids, masks):
                    mask = self._mask(value)
                    if mask.shape != (height, width):
                        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    if np.any(mask):
                        source = self._tracking_objects[mid]
                        result.append(MaskData(mid=mid, mask=mask, origin="Sam3_tracking",
                                               class_id=source.class_id, color_idx=source.color_idx,
                                               time_stamp=1))
        return result

    def prop_thread_func(self, mask_objs):
        self.set_masks(mask_objs)
        return self.propagate_to_next_img()


class Sam3AutomaticMaskGenerator:
    """Generate class-independent proposals from a grid of interactive prompts."""
    def __init__(self, sam, points_per_side=16, points_per_batch=16,
                 pred_iou_thresh=0.8, stability_score_thresh=0.95,
                 stability_score_offset=1.0, box_nms_thresh=0.7,
                 min_mask_region_area=0):
        if points_per_side < 1 or points_per_batch < 1:
            raise ValueError("AMG point counts must be positive")
        self.sam = sam
        self.points_per_side, self.points_per_batch = points_per_side, points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.min_mask_region_area = min_mask_region_area

    def generate(self, image):
        from torchvision.ops import batched_nms

        self.sam._require_image()
        height, width = image.shape[:2]
        axis = (np.arange(self.points_per_side) + 0.5) / self.points_per_side
        xx, yy = np.meshgrid(axis * width, axis * height)
        points = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2).astype(np.float32)
        candidates, qualities, boxes = [], [], []
        for start in range(0, len(points), self.points_per_batch):
            batch = points[start:start + self.points_per_batch]
            masks, scores, logits = self.sam._predict(
                batch, np.ones(batch.shape[:2], np.int32), multimask_output=True)
            masks = masks.reshape(-1, height, width)
            logits = logits.reshape(-1, *logits.shape[-2:])
            for mask, score, low_res in zip(masks, scores.reshape(-1), logits):
                offset = self.stability_score_offset
                union = np.count_nonzero(low_res > -offset)
                stability = np.count_nonzero(low_res > offset) / union if union else 0
                area = np.count_nonzero(mask)
                if score < self.pred_iou_thresh or stability < self.stability_score_thresh or area <= self.min_mask_region_area:
                    continue
                yy, xx = np.nonzero(mask)
                candidates.append(mask > 0)
                qualities.append(float(score))
                boxes.append([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1])
        if not candidates:
            return []
        keep = batched_nms(torch.tensor(boxes, dtype=torch.float32),
                           torch.tensor(qualities), torch.zeros(len(boxes), dtype=torch.int64),
                           self.box_nms_thresh)
        return [{"segmentation": candidates[index]} for index in keep.tolist()]
