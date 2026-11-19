import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QMutex

from aramsam_annotator.annotator import Annotator
from aramsam_annotator.backend.annotations import AnnotationObject, MaskData, MaskIdHandler
from aramsam_annotator.backend.propagation import merge_propagated_masks
from aramsam_annotator.backend.storage import AnnotationRepository
from aramsam_annotator.configs import AramsamConfigs, SaveData
from aramsam_annotator.run_sam import SamInference
from aramsam_annotator.workers import Sam2PropagationWorker, BoxPropagationWorker


class PropagationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / 'image.png'
        cv2.imwrite(str(self.image), np.zeros((32, 32, 3), np.uint8))
        self.handler = MaskIdHandler()

    def mask(self, mid=7, x=2, track_id=None):
        pixels = np.zeros((32, 32), np.uint8)
        pixels[8:16, x:x + 6] = 255
        kwargs = {} if track_id is None else {'track_id': track_id}
        return MaskData(mid=mid, origin='Sam2_tracking', mask=pixels,
                        class_id=4, color_idx=2, time_stamp=1, **kwargs)

    def annotation(self, masks):
        annotation = AnnotationObject(self.image)
        annotation.add_masks(masks, decision=True)
        annotation.good_masks = list(masks)
        return annotation

    def test_native_and_yolo_keep_track_identity_but_allocate_fresh_display_ids(self):
        for style in ('default', 'yolo'):
            for segmentation in (True, False):
                if style == 'default' and not segmentation:
                    continue
                with self.subTest(style=style, segmentation=segmentation):
                    source = self.mask()
                    source.bbox = (2, 8, 8, 16)
                    annotation = self.annotation([source])
                    repository = AnnotationRepository(SaveData(mask_style=style, save_masks=segmentation,
                                                               save_bboxes=not segmentation))
                    output = self.root / f'{style}-{segmentation}'
                    repository.save(annotation, output)
                    loaded = AnnotationObject(self.image)
                    repository.load(loaded, output, self.handler)
                    restored = loaded.good_masks[0]
                    self.assertNotEqual(restored.mid, source.mid)
                    self.assertEqual(restored.track_id, source.track_id)
                    self.assertEqual(restored.class_id, source.class_id)
                    self.assertEqual(restored.origin, source.origin)
                    self.assertEqual(restored.color_idx, source.color_idx)

    def test_yolo_metadata_stays_aligned_when_empty_objects_are_not_exported(self):
        empty, source = self.mask(8), self.mask(9)
        empty.mask[:] = 0
        repository = AnnotationRepository(SaveData())
        output = self.root / 'output'
        repository.save(self.annotation([empty, source]), output)
        loaded = AnnotationObject(self.image)
        repository.load(loaded, output, self.handler)
        self.assertEqual([mask.track_id for mask in loaded.good_masks], [source.track_id])

    def test_stale_or_invalid_yolo_sidecars_do_not_attach_wrong_identities(self):
        repository = AnnotationRepository(SaveData())
        output = self.root / 'output'
        source = self.mask()
        repository.save(self.annotation([source]), output)
        labels = output / 'labels' / 'image.txt'
        original = labels.read_text()
        labels.write_text('9' + original[1:])
        loaded = AnnotationObject(self.image)
        repository.load(loaded, output, self.handler)
        self.assertEqual(loaded.good_masks[0].class_id, 9)
        self.assertNotEqual(loaded.good_masks[0].track_id, source.track_id)
        self.assertTrue(loaded.good_masks[0].track_id.startswith('legacy:'))
        labels.write_text(original)
        for contents in ('not json', '[]', '{"version": 1, "instances": {}}'):
            with self.subTest(contents=contents):
                (output / 'metadata' / 'image.json').write_text(contents)
                repository.load(loaded, output, self.handler)
                self.assertTrue(loaded.good_masks[0].track_id.startswith('legacy:'))

    def test_old_native_and_yolo_files_get_repeatable_local_identities(self):
        for style in ('default', 'yolo'):
            with self.subTest(style=style):
                repository = AnnotationRepository(SaveData(mask_style=style))
                output = self.root / style
                repository.save(self.annotation([self.mask()]), output)
                if style == 'default':
                    metadata_path = output / 'image_annots' / 'annotations.json'
                    records = json.loads(metadata_path.read_text())
                    for record in records:
                        record.pop('track_id')
                    metadata_path.write_text(json.dumps(records))
                else:
                    (output / 'metadata' / 'image.json').unlink()
                first, second = AnnotationObject(self.image), AnnotationObject(self.image)
                repository.load(first, output, self.handler)
                repository.load(second, output, self.handler)
                self.assertNotEqual(first.good_masks[0].mid, second.good_masks[0].mid)
                self.assertEqual(first.good_masks[0].track_id, second.good_masks[0].track_id)
                self.assertTrue(first.good_masks[0].track_id.startswith('legacy:'))

    def test_accepting_an_existing_mask_keeps_its_identity(self):
        annotator = Annotator(AramsamConfigs())
        annotator.create_new_annotation(self.image)
        source = self.mask()
        source.origin = 'Sam1_proposed'
        annotator.annotation.add_masks([source])
        annotator.good_mask(class_id=9)
        self.assertEqual(annotator.annotation.good_masks[0].track_id, source.track_id)

    def test_identity_replaces_all_old_copies_even_when_geometry_and_class_change(self):
        first = self.mask(90, 2, 'same-object')
        duplicate = self.mask(91, 3, 'same-object')
        independent = self.mask(92, 20, 'different-object')
        annotation = self.annotation([first, duplicate, independent])
        prediction = self.mask(7, 20, 'same-object')
        prediction.class_id = 9
        merge_propagated_masks(annotation, [prediction], {7})
        self.assertEqual([mask.mid for mask in annotation.masks], [92, 7])
        self.assertEqual([mask.track_id for mask in annotation.masks], ['different-object', 'same-object'])
        self.assertEqual(annotation.good_masks, [independent])

    def test_attempted_identity_is_removed_even_when_tracking_returns_no_mask(self):
        annotation = self.annotation([self.mask(90, track_id='vanished'),
                                      self.mask(91, track_id='independent')])
        merge_propagated_masks(annotation, [], {7}, source_track_ids={'vanished'})
        self.assertEqual([mask.track_id for mask in annotation.masks], ['independent'])

    def test_batched_tracking_preserves_identity_class_and_color_from_the_source(self):
        sources = [self.mask(7, 2), self.mask(8, 20)]
        annotation = self.annotation([self.mask(90, 2, sources[0].track_id)])
        model = Mock()
        model.prop_thread_func.side_effect = lambda batch: [
            MaskData(mid=obj.mid, origin='Sam2_tracking', mask=np.roll(obj.mask, 2, axis=1))
            for obj in batch]
        worker = Sam2PropagationWorker(model, annotation, sources, 1, True, QMutex())
        worker.execute()
        self.assertEqual(len(annotation.masks), 2)
        for source, result in zip(sources, annotation.masks):
            self.assertEqual(result.track_id, source.track_id)
            self.assertEqual(result.class_id, source.class_id)
            self.assertEqual(result.color_idx, source.color_idx)

    def test_panorama_empty_result_does_not_shift_later_objects_identity(self):
        annotator = Annotator(AramsamConfigs())
        annotator.create_new_annotation(self.image)
        sources = [self.mask(7, 2), self.mask(8, 20)]
        old = self.mask(90, 2, sources[1].track_id)
        annotator.annotation.add_masks([old], decision=True)
        annotator.annotation.good_masks = [old]
        tracker = SimpleNamespace(
            track=Mock(return_value=[[2, 8, 8, 16], [20, 8, 26, 16]]),
            matching_masks=[sources], tracked_sources=sources,
        )
        annotator.sam = Mock()
        annotator.sam.transform_bboxes.side_effect = lambda boxes, shape: boxes
        annotator.sam.predict_batch.return_value = torch.stack([
            torch.zeros((1, 32, 32), dtype=torch.bool), torch.from_numpy(sources[1].mask > 0)[None]])
        annotator.sam._torch_to_npmasks.side_effect = SamInference.__new__(SamInference)._torch_to_npmasks
        annotator._propagate_bboxes(tracker)
        self.assertEqual(len(annotator.annotation.masks), 1)
        result = annotator.annotation.masks[0]
        self.assertEqual(result.mid, sources[1].mid)
        self.assertEqual(result.track_id, sources[1].track_id)
        self.assertEqual(result.class_id, sources[1].class_id)

    def test_deletion_before_or_during_a_worker_prevents_late_results_reappearing(self):
        for kind in ('tracking', 'bounding_box'):
            for timing in ('queued', 'inflight'):
                with self.subTest(kind=kind, timing=timing):
                    source = self.mask()
                    annotator = Annotator(AramsamConfigs())
                    annotator.annotation = self.annotation([source])
                    annotator.next_annotation = self.annotation([self.mask(99, track_id=source.track_id)])
                    model = Mock()

                    def predicted_masks(*args):
                        if timing == 'inflight':
                            annotator.delete_mask(source.mid)
                        if kind == 'tracking':
                            return [MaskData(mid=source.mid, origin='Sam2_tracking', mask=source.mask.copy())]
                        return [source.mask.copy()]

                    model.prop_thread_func.side_effect = predicted_masks
                    model.predict_boxes.side_effect = predicted_masks
                    if kind == 'tracking':
                        worker = Sam2PropagationWorker(model, annotator.next_annotation, [source],
                                                       1, True, QMutex())
                    else:
                        worker = BoxPropagationWorker(model, annotator.annotation,
                                                       annotator.next_annotation, [source], 2, QMutex())
                    if timing == 'queued':
                        annotator.delete_mask(source.mid)
                    worker.execute()
                    self.assertEqual(annotator.next_annotation.masks, [])
                    self.assertEqual(annotator.next_annotation.good_masks, [])
                    if timing == 'queued':
                        model.prop_thread_func.assert_not_called()
                        model.predict_boxes.assert_not_called()

    def test_deletion_exclusions_survive_native_and_yolo_reload(self):
        for style in ('default', 'yolo'):
            with self.subTest(style=style):
                annotation = self.annotation([self.mask()])
                annotation.deleted_track_ids = {'locally-deleted'}
                annotation.suppressed_track_ids = {'deleted-upstream'}
                repository = AnnotationRepository(SaveData(mask_style=style))
                output = self.root / style
                repository.save(annotation, output)
                restored = AnnotationObject(self.image)
                repository.load(restored, output, self.handler)
                self.assertEqual(restored.deleted_track_ids, annotation.deleted_track_ids)
                self.assertEqual(restored.suppressed_track_ids, annotation.suppressed_track_ids)


if __name__ == '__main__':
    unittest.main()
