import torch
import cv2
import numpy as np
import pandas as pd
import time
import json

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from segment_anything.modeling import Sam

from aramsam_annotator.mask_visualizations import MaskVisualization, MaskData, MaskIdHandler


class SAM2AutomaticMaskGenerator:
    # Placeholder for Typing
    pass


class SamInference:
    def __init__(
        self,
        mask_id_handler: MaskIdHandler,
        sam_checkpoint="sam_vit_b_01ec64.pth",
        model_type="vit_b",
        device="cpu",
        background_embedding=True,
        amg_config_path="ExperimentData/AmgEvaluationData/Sam1_VitH_best_config.json",
    ):
        self.mask_id_handler = mask_id_handler
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type

        self.device = device

        checkpoint = torch.load(
            sam_checkpoint, map_location=torch.device(device), weights_only=True
        )

        # Initialize the model from the registry without loading the checkpoint
        self.sam = sam_model_registry[model_type]()

        # Load the state_dict from the checkpoint
        self.sam.load_state_dict(checkpoint)

        # Move the model to the appropriate device
        self.sam.to(device=self.device)
        if background_embedding:
            self.predictor = BackgroundThreadSamPredictor(self.sam)
        else:
            self.predictor = MainThreadSamPredictor(self.sam)
        # self.custom_amg = CustomAMG(self)
        amg_kwargs = load_amg_config(amg_config_path)
        self.amg = DefaultAMG(self, SamAutomaticMaskGenerator(self.sam, **amg_kwargs))
        self.img = None

    def predict_batch(
        self,
        pts: torch.Tensor = None,
        pts_labels: torch.Tensor = None,
        bboxes: torch.Tensor = None,
        mask_input: list[np.ndarray] = None,
    ):
        assert (
            (pts is not None and pts_labels is not None)
            or (mask_input is not None)
            or (bboxes is not None)
        ), "Either pts and pts_labels or bboxes or masks must be provided"

        if not self.predictor.is_image_set:
            raise ValueError("Image embedding has not been initialized yet")
        if mask_input is not None:
            mask_input = self._prep_batch_masks(mask_input)
        masks, scores, logits = self.predictor.predict_torch(
            point_coords=pts,
            point_labels=pts_labels,
            boxes=bboxes,
            mask_input=mask_input,
            multimask_output=False,
        )
        return masks

    def predict(
        self,
        pts: np.ndarray = None,
        pts_labels: np.ndarray = None,
        bboxes: np.ndarray = None,
        mask_input: np.ndarray = None,
    ):
        if not self.predictor.is_image_set:
            raise ValueError("Image embedding has not been initialized yet")
        assert (
            pts.dtype == np.float32 and pts.shape[-1] == 2
        ), "Expecting numpy array with shape (N, 2) and dtype float32"
        masks, scores, logits = self.predictor.predict(
            point_coords=pts,
            point_labels=pts_labels,
            box=bboxes,
            mask_input=mask_input,
            multimask_output=False,
        )
        return masks[0].astype(np.uint8) * 255

    def reset_img(self):
        self.predictor.reset_image()

    def _prep_batch_pts_and_lbls(self, pts, lbls, shp: tuple[int, int]):
        pt_tensor = torch.tensor(pts, dtype=torch.int32, device=self.device)
        label_tensor = torch.tensor(lbls, dtype=torch.int32, device=self.device)
        transformed_pts = self.predictor.transform.apply_coords_torch(pt_tensor, shp)
        return transformed_pts, label_tensor

    def transform_bboxes(self, bboxes: list[np.ndarray], shp: tuple[int, int]):

        torch_boxes_orig_shape = torch.tensor(
            bboxes, dtype=torch.int32, device=self.device
        )
        return self.predictor.transform.apply_boxes_torch(torch_boxes_orig_shape, shp)

    def _prep_batch_masks(self, masks_np: list[np.ndarray]):
        mask_input = torch.zeros(
            (len(masks_np), 1, 256, 256), dtype=torch.float32, device=self.device
        )
        for i, m in enumerate(masks_np):
            mask_resized = cv2.resize(m, (256, 256), m)
            mask_float = mask_resized.astype(np.float32)  # / 255.0
            mask_tensor = torch.tensor(
                mask_float, dtype=torch.float32, device=self.device
            )
            mask_input[i] = mask_tensor

        return mask_input

    def _torch_to_npmasks(self, mask_tensor: torch.Tensor) -> list[np.ndarray]:
        mask_lst = []
        for mt in mask_tensor:
            mask = mt.cpu().numpy()
            if not np.any(mask):
                continue
            mask = mask.astype(np.uint8) * 255
            mask = np.squeeze(mask, axis=0)
            mask_lst.append(mask)
        return mask_lst


class MainThreadSamPredictor(SamPredictor):
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        self.is_image_set = False
        super().__init__(sam_model=sam_model)

    def embed_img(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ):
        # Get img embedding without overiding the current embedding
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        self.set_image(image)
        self.is_image_set = True
        return None


class BackgroundThreadSamPredictor(SamPredictor):
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        self.is_image_set = False
        super().__init__(sam_model=sam_model)

    def embed_img(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ):
        # Get img embedding without overiding the current embedding
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[
            None, :, :, :
        ]
        print_tensor_gpu_usage(input_image_torch, "input_image_torch")

        features, original_size, input_size = self.to_torch_img(
            input_image_torch, image.shape[:2]
        )
        return features, original_size, input_size

    def to_torch_img(
        self,
        transformed_image: torch.Tensor,
        original_image_size: tuple[int, ...],
    ):
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."

        original_size = original_image_size
        input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        features = self.model.image_encoder(input_image)

        return features, original_size, input_size

    def set_features(self, features, original_size, input_size):
        self.reset_image()
        self.original_size = original_size
        self.input_size = input_size
        self.features = features
        self.is_image_set = True


class Sam2Inference:
    def __init__(
        self,
        mask_id_handler: MaskIdHandler,
        sam2_checkpoint: str = "sam2_hiera_small.pt",
        cfg_path: str = "sam2_hiera_s.yaml",
        background_embedding: bool = True,
        amg_config_path: str = "ExperimentData/AmgEvaluationData/Sam2_hieraS2.1_best_config.json",
    ):
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. SAM2 requires a CUDA enabled GPU."
            )

        self.device = "cuda"

        self._init_mixed_precision()

        self.mask_id_handler = mask_id_handler
        self.predictor: SAM2VideoPredictor = build_sam2_video_predictor(
            cfg_path, sam2_checkpoint, device="cuda"
        )
        # self.custom_amg = CustomAMG(self)
        amg_kwargs = load_amg_config(amg_config_path)
        self.amg = DefaultAMG(
            self, SAM2AutomaticMaskGenerator(self.predictor, **amg_kwargs)
        )
        self.img_predictor = SAM2ImagePredictor(self.predictor)
        self.imgs: list[np.ndarray] = None
        self.inference_state: dict = None
        self.latest_obj_id = 0
        self.predictor.is_image_set = False

    def _init_mixed_precision(self):
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def set_features(self, imgs: list[np.ndarray]):
        if self.predictor.is_image_set:
            print_gpu_usage()
            self.predictor.reset_state(self.inference_state)
            self.img_predictor.reset_predictor()
        self.imgs = imgs
        assert (
            len(imgs) >= 1 and imgs[0].dtype == np.uint8 and imgs[0].shape[2] == 3
        ), "Expecting one or more RGB Numpy arrays with shape (H, W, 3) and dtype uint8"

        self.inference_state = self.predictor.init_state(
            video_path=imgs,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=True,
        )
        self.predictor.is_image_set = True
        frame_idx = 0
        self._set_img_predictor_features(
            self.inference_state["cached_features"][frame_idx][1]
        )

    def _set_img_predictor_features(self, backbone_out):
        h, w, c = self.imgs[0].shape
        self.img_predictor._orig_hw = [(h, w)]

        ### copied from sam2_image_predictor.SAM2ImagePredictor.set_image:
        _, vision_feats, _, _ = self.img_predictor.model._prepare_backbone_features(
            backbone_out
        )
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.img_predictor.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.img_predictor.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(
                vision_feats[::-1], self.img_predictor._bb_feat_sizes[::-1]
            )
        ][::-1]
        self.img_predictor._features = {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
        }
        self.img_predictor._is_image_set = True

    def predict(self, pts: np.ndarray, pts_labels: np.ndarray):
        # single object
        assert (
            pts.dtype == np.float32 and pts.shape[-1] == 2
        ), "Expecting numpy array with shape (N, 2) and dtype float32"
        assert (
            pts_labels.dtype == np.int32 and pts_labels.ndim == 1
        ), "Expecting numpy array with shape (N) and dtype int32"

        mask, scores, logits = self.img_predictor.predict(
            point_coords=pts,
            point_labels=pts_labels,
            multimask_output=False,
        )
        return self._logits_to_npmask(mask)

    def predict_batch(
        self,
        pts: np.ndarray = None,
        pts_labels: np.ndarray = None,
        bboxes: np.ndarray = None,
        mask_input: list[np.ndarray] = None,
    ):
        # multiple objects
        assert (
            (pts is not None and pts_labels is not None)
            or (mask_input is not None)
            or (bboxes is not None)
        ), "Either pts and pts_labels or bboxes or masks must be provided"

        output_masks = []
        for obj_idx in range(pts.shape[0]):
            mask, scores, logits = self.img_predictor.predict(
                point_coords=pts[obj_idx] if pts is not None else None,
                point_labels=pts_labels[obj_idx] if pts_labels is not None else None,
                box=bboxes[obj_idx] if bboxes is not None else None,
                mask_input=mask_input[obj_idx] if mask_input is not None else None,
                multimask_output=False,
            )
            # self.latest_obj_id += 1
            output_masks.append(mask > 0.0)
        return output_masks

    def set_masks(self, mask_objs: list[MaskData]):
        self.predictor.reset_state(self.inference_state)
        for m in mask_objs:
            assert (
                m.mask.dtype == np.uint8 and m.mask.ndim == 2
            ), "Expecting numpy array with 2 dims and dtype bool"
            mask = m.mask.astype(bool)
            self.predictor.add_new_mask(self.inference_state, 0, m.mid, mask)

    def propagate_to_next_img(self) -> list[MaskData]:
        prop_masks: list[MaskData] = []
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(self.inference_state):
            if out_frame_idx == 1:
                for i, out_mid in enumerate(out_obj_ids):
                    np_mask = self._logits_to_npmask(out_mask_logits[i])
                    if np_mask is not None:
                        prop_masks.append(MaskData(out_mid, np_mask, "Sam2_tracking"))
        return prop_masks

    def prop_thread_func(self, mask_objs: list[MaskData]):
        self._init_mixed_precision()
        self.predictor.reset_state(self.inference_state)
        self.set_masks(mask_objs)
        return self.propagate_to_next_img()

    def _logits_to_npmask(self, out_mask_logit: torch.Tensor | np.ndarray):
        if isinstance(out_mask_logit, np.ndarray):
            boolmask = out_mask_logit > 0.0
        else:
            boolmask = (out_mask_logit > 0.0).cpu().numpy()
        if np.any(boolmask):
            npmask = np.transpose(boolmask, (1, 2, 0))
            npmask = np.array(npmask, dtype=np.uint8).reshape(npmask.shape[:2]) * 255
            return npmask
        else:
            return None

    def _prep_batch_pts_and_lbls(
        self, pts: np.ndarray, lbls: np.ndarray, shp: tuple[int, int]
    ):
        return pts.astype(np.float32), lbls


class CustomAMG:
    def __init__(self, sam_cls: SamInference | Sam2Inference) -> None:
        self.sam_cls: SamInference | Sam2Inference = sam_cls

    def __call__(self, roi_pts=False, n_points=100, msize_thresh=10000):
        pts = None
        pt_labels = None
        start_getting_prompt_points = time.time()
        if roi_pts:
            pts, pt_labels = self._get_roi_points(n_points)
        else:
            pts, pt_labels = self._get_pt_grid(n_points)
        transformed_pts, transformed_lbls = self.sam_cls._prep_batch_pts_and_lbls(
            pts, pt_labels, self.imgbgr.shape[:2]
        )
        print(
            f"Time taken for only getting the prompt points: {time.time() - start_getting_prompt_points}"
        )
        startpredtimer = time.time()
        masks = self.sam_cls.predict_batch(transformed_pts, transformed_lbls)
        print(f"Time taken for only the prediction: {time.time() - startpredtimer}")

        xmins = []
        ymins = []
        mask_lst = []
        mask_objs = []

        annotated_image = self.imgbgr.copy()
        maskpostpocessingtimer = time.time()
        for mask in masks:
            if not isinstance(mask, np.ndarray):
                mask = mask.cpu().numpy()
            if np.count_nonzero(mask) < msize_thresh and np.any(mask):
                kernelmask = np.array(mask, dtype=np.uint8) * 255
                kernelmask = np.squeeze(kernelmask, axis=0)
                if np.amax(kernelmask) != 255:
                    continue
                mask_lst.append(kernelmask)
                origin = (
                    "Sam1_proposed"
                    if isinstance(self.sam_cls, SamInference)
                    else "Sam2_proposed"
                )
                mask_objs.append(
                    MaskData(self.sam_cls.mask_id_handler.set_id(), kernelmask, origin)
                )

                ycords, xcords = np.where(kernelmask == 255)
                # annotated_image[ycords, xcords] = b, g, r
                ymin, xmin = np.amin(ycords), np.amin(xcords)

                ymins.append(ymin)
                xmins.append(xmin)
        print(
            f"Time taken for only the mask post processing: {time.time() - maskpostpocessingtimer}"
        )
        startmvistimer = time.time()
        mvis = MaskVisualization()
        mvis.set_annotation(img=annotated_image, mask_objs=mask_objs)
        annotated_image = mvis.get_masked_img()
        mvis_mask_objs = mvis.mask_objs
        print(f"Time taken for only the visualization: {time.time() - startmvistimer}")

        start_sorting_pts = time.time()
        cord_dict = {"ymins": ymins, "xmins": xmins}
        cord_df = pd.DataFrame(cord_dict)
        cord_df = cord_df.sort_values(by=["ymins", "xmins"])
        order = cord_df.index.to_list()
        # masks_np = np.array(mask_lst, dtype=np.uint8)[order]
        ordered_mask_objs = [mvis_mask_objs[i] for i in order]
        print(
            f"Time taken for only sorting the points: {time.time() - start_sorting_pts}"
        )
        return ordered_mask_objs, annotated_image

    def _get_roi_points(self, n=100, lower_hsv=[28, 0, 0], higher_hsv=[179, 255, 255]):
        lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        higher_hsv = np.array(higher_hsv, dtype=np.uint8)
        imgBLUR = cv2.GaussianBlur(self.imgbgr, (3, 3), 0)
        imgHSV = cv2.cvtColor(imgBLUR, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHSV, lower_hsv, higher_hsv)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
        indcs = np.column_stack(np.where(mask == 255))
        if len(indcs) < n:
            n = len(indcs)
        rand_indcs = indcs[np.random.choice(indcs.shape[0], n, replace=False)]
        roi_points = rand_indcs[:, [1, 0]]
        pt_tensor = torch.tensor(
            roi_points.reshape(-1, 1, 2), dtype=torch.int32, device=self.sam_cls.device
        )
        label_tensor = torch.ones((n, 1), dtype=torch.int32, device=self.sam_cls.device)

        return pt_tensor, label_tensor

    def _get_pt_grid(self, n):
        height, width, _ = self.imgbgr.shape
        # Total number of points

        # Calculate the number of rows and columns for the grid
        grid_size = int(np.sqrt(n))
        rows, cols = grid_size, grid_size

        pt_distx = width / cols
        pad_x = int(pt_distx / 2)
        pt_disty = height / rows
        pad_y = int(pt_disty / 2)
        # Generate coordinates for grid points
        row_indices = np.linspace(pad_y, height - 1 - pad_y, rows, dtype=int)
        col_indices = np.linspace(pad_x, width - 1 - pad_x, cols, dtype=int)

        # Create a meshgrid of row and column indices
        grid_y, grid_x = np.meshgrid(col_indices, row_indices)

        # Flatten the meshgrid to get a list of coordinates
        grid_points = np.vstack([grid_y.ravel(), grid_x.ravel()]).T

        pts_np = grid_points.reshape(-1, 1, 2)
        lbls_np = np.ones((rows * cols, 1), dtype=np.int32)

        return pts_np, lbls_np

    def set_visualization_img(self, img):
        self.imgbgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class DefaultAMG:
    def __init__(
        self,
        sam_cls: SamInference | Sam2Inference,
        amg: SamAutomaticMaskGenerator | SAM2AutomaticMaskGenerator,
    ) -> None:
        self.sam_cls: SamInference | Sam2Inference = sam_cls
        self.amg = amg

    def __call__(self, msize_thresh=10000):

        startpredtimer = time.time()
        masks = self.amg.generate(cv2.cvtColor(self.imgbgr, cv2.COLOR_BGR2RGB))
        print(f"Time taken for only the default amg: {time.time() - startpredtimer}")

        xmins = []
        ymins = []
        mask_lst = []
        mask_objs = []

        annotated_image = self.imgbgr.copy()
        maskpostpocessingtimer = time.time()
        for mask in masks:
            if isinstance(mask, dict):
                mask = mask["segmentation"]
            elif not isinstance(mask, np.ndarray):
                mask = mask.cpu().numpy()
            if np.count_nonzero(mask) < msize_thresh and np.any(mask):
                kernelmask = np.array(mask, dtype=np.uint8) * 255
                if len(kernelmask.shape) > 2:
                    kernelmask = np.squeeze(kernelmask, axis=0)
                if np.amax(kernelmask) != 255:
                    continue
                mask_lst.append(kernelmask)
                origin = (
                    "Sam1_proposed"
                    if isinstance(self.sam_cls, SamInference)
                    else "Sam2_proposed"
                )
                mask_objs.append(
                    MaskData(self.sam_cls.mask_id_handler.set_id(), kernelmask, origin)
                )

                ycords, xcords = np.where(kernelmask == 255)
                ymin, xmin = np.amin(ycords), np.amin(xcords)

                ymins.append(ymin)
                xmins.append(xmin)
        print(
            f"Time taken for only the mask post processing: {time.time() - maskpostpocessingtimer}"
        )
        startmvistimer = time.time()
        mvis = MaskVisualization()
        mvis.set_annotation(img=annotated_image, mask_objs=mask_objs)
        annotated_image = mvis.get_masked_img()
        mvis_mask_objs = mvis.mask_objs
        print(f"Time taken for only the visualization: {time.time() - startmvistimer}")

        start_sorting_pts = time.time()
        cord_dict = {"ymins": ymins, "xmins": xmins}
        cord_df = pd.DataFrame(cord_dict)
        cord_df = cord_df.sort_values(by=["ymins", "xmins"])
        order = cord_df.index.to_list()
        ordered_mask_objs = [mvis_mask_objs[i] for i in order]
        print(
            f"Time taken for only sorting the points: {time.time() - start_sorting_pts}"
        )
        return ordered_mask_objs, annotated_image

    def set_visualization_img(self, img):
        self.imgbgr = img


def print_gpu_usage():
    if torch.cuda.is_available():

        print(
            f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0)/1024/1024/1024} GB"
        )
        print(
            f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(0)/1024/1024/1024} GB"
        )
        print(
            f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(0)/1024/1024/1024} GB"
        )
    else:
        print("CUDA is not available. No GPU memory stats to show.")


def print_tensor_gpu_usage(tensor, name=""):
    memory_in_bytes = tensor.element_size() * tensor.nelement()
    memory_in_megabytes = memory_in_bytes / (1024**2)
    print(f"Memory usage of {name}: {memory_in_megabytes} MB")


def load_amg_config(amg_config_path: str):
    with open(amg_config_path, "r") as f:
        amg_config = json.load(f)
    return amg_config
