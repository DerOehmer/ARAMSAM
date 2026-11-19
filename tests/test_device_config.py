import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from aramsam_annotator.annotator import Annotator
from aramsam_annotator.backend.annotations import MaskData
from aramsam_annotator.configs import AramsamConfigs, load_configs_from_yaml
from aramsam_annotator.run_sam import Sam2Inference
from aramsam_annotator.run_yolo import YoloInference


class DeviceConfigTests(unittest.TestCase):
    def test_yaml_gpu_option_and_default(self):
        for yaml_text, expected in [('', True), ('use_gpu: true', True), ('use_gpu: false', False)]:
            with self.subTest(yaml_text=yaml_text), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'configs.yaml'
                path.write_text(yaml_text)
                self.assertEqual(load_configs_from_yaml(path).use_gpu, expected)

    def test_device_selection_and_forwarding_to_sam_and_yolo(self):
        for use_gpu in (False, True):
            for available in (False, True):
                expected = 'cuda' if use_gpu and available else 'cpu'
                with self.subTest(use_gpu=use_gpu, available=available), \
                        patch('aramsam_annotator.annotator.torch.cuda.is_available', return_value=available):
                    annotator = Annotator(AramsamConfigs(use_gpu=use_gpu))
                    self.assertEqual(annotator.device, expected)
                    for generation, cls in [(1, 'SamInference'), (2, 'Sam2Inference'), (3, 'Sam3Inference')]:
                        with patch('aramsam_annotator.annotator.' + cls) as model:
                            annotator.set_sam_version(generation)
                            self.assertEqual(model.call_args.kwargs['device'], expected)
                    annotator.annotation = SimpleNamespace(img=object())
                    with patch('aramsam_annotator.annotator.YoloInference') as yolo:
                        annotator.prepare_yolo()
                        self.assertEqual(yolo.call_args.kwargs['device'], expected)

    def test_sam2_builds_on_selected_device_and_only_uses_cuda_precision_on_gpu(self):
        for device, available, expected in [('cpu', True, 'cpu'), ('cuda', True, 'cuda'),
                                            (None, False, 'cpu'), (None, True, 'cuda')]:
            builder = Mock()
            modules = {
                'sam2': Mock(),
                'sam2.build_sam': SimpleNamespace(build_sam2_video_predictor=builder),
                'sam2.sam2_video_predictor': SimpleNamespace(SAM2VideoPredictor=Mock()),
                'sam2.sam2_image_predictor': SimpleNamespace(SAM2ImagePredictor=Mock()),
                'sam2.automatic_mask_generator': SimpleNamespace(SAM2AutomaticMaskGenerator=Mock()),
            }
            with self.subTest(device=device, available=available), \
                    patch.dict(sys.modules, modules), \
                    patch('aramsam_annotator.run_sam.torch.cuda.is_available', return_value=available), \
                    patch.object(Sam2Inference, '_init_mixed_precision') as precision:
                sam = Sam2Inference(Mock(), device=device)
                self.assertEqual(sam.device, expected)
                self.assertEqual(builder.call_args.kwargs['device'], expected)
                self.assertEqual(precision.call_count, int(expected == 'cuda'))

    def test_yolo_passes_device_to_inference(self):
        for device in ('cpu', 'cuda'):
            with self.subTest(device=device):
                yolo = YoloInference(Mock(), device=device)
                yolo.model = Mock(return_value=[SimpleNamespace(boxes=[])])
                image = object()
                yolo.set_img(image)
                self.assertEqual(yolo.infer_image(), [])
                yolo.model.assert_called_once_with(image, device=device)

    def test_sam2_propagates_masks_on_cpu_without_initializing_cuda(self):
        for device in ('cpu', 'cuda'):
            with self.subTest(device=device):
                sam = Sam2Inference.__new__(Sam2Inference)
                sam.device = device
                sam.inference_state = {'num_frames': 2}
                sam.imgs = None
                sam.predictor = Mock()
                mask = np.zeros((8, 8), np.uint8)
                mask[2:6, 2:6] = 255
                logits = torch.from_numpy(mask > 0).float()[None, None] * 2 - 1
                sam.predictor.propagate_in_video.return_value = iter([
                    (0, [7], logits), (1, [7], logits),
                ])
                source = MaskData(mid=7, origin='Polygon_drawing', mask=mask)

                with patch.object(sam, '_init_mixed_precision') as precision:
                    if device == 'cpu':
                        precision.side_effect = RuntimeError('No CUDA GPUs are available')
                    result = sam.prop_thread_func([source])

                self.assertEqual(precision.call_count, int(device == 'cuda'))
                sam.predictor.add_new_mask.assert_called_once()
                args = sam.predictor.add_new_mask.call_args.args
                self.assertIs(args[0], sam.inference_state)
                self.assertEqual(args[1:3], (0, 7))
                np.testing.assert_array_equal(args[3], mask > 0)
                sam.predictor.propagate_in_video.assert_called_once_with(sam.inference_state)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].mid, 7)
                self.assertEqual(result[0].origin, 'Sam2_tracking')
                np.testing.assert_array_equal(result[0].mask, mask)


if __name__ == '__main__':
    unittest.main()
