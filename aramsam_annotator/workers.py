"""Qt worker adapters with one exception-safe lifecycle.

Workers never call UI methods. Inference results are delivered through signals;
all predictor locks are released before success, failure, or completion signals.
"""
from contextlib import contextmanager, nullcontext
import sys
import traceback

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot

from aramsam_annotator.run_sam import BackgroundThreadSamPredictor


@contextmanager
def locked(mutex):
    mutex.lock()
    try:
        yield
    finally:
        mutex.unlock()


class WorkerSignals(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(tuple)
    result = pyqtSignal(tuple)


class YoloSignals(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(tuple)
    result = pyqtSignal(list)


class Sam2EmbeddingWorkerSignals(QObject):
    finished = pyqtSignal(int)
    error = pyqtSignal(tuple)
    result = pyqtSignal(bool)


class Sam2PropagationWorkerSignals(QObject):
    finished = pyqtSignal(int)
    error = pyqtSignal(tuple)


class InferenceWorker(QRunnable):
    """Subclasses supply work; this base guarantees exactly one completion."""
    def __init__(self, signals, completion):
        super().__init__()
        self.signals = signals
        self.completion = completion

    def execute(self):
        raise NotImplementedError

    @pyqtSlot()
    def run(self):
        try:
            result = self.execute()
        except Exception:
            traceback.print_exc()
            error_type, error = sys.exc_info()[:2]
            self.signals.error.emit((error_type, error, traceback.format_exc()))
        else:
            if hasattr(self.signals, 'result'):
                self.signals.result.emit(result)
        finally:
            self.signals.finished.emit(self.completion)


class Sam1EmbeddingWorker(InferenceWorker):
    def __init__(self, sam_predictor, img, img_name, mutex, delay=0.0):
        super().__init__(WorkerSignals(), img_name)
        self.sam_predictor = sam_predictor
        self.img = img
        self.img_name = img_name
        self.mutex = mutex
        self.delay = delay  # Kept for callers of the original API.

    def execute(self):
        # Background SAM1 reads weights without mutating the active predictor.
        guard = (nullcontext() if isinstance(self.sam_predictor, BackgroundThreadSamPredictor)
                 else locked(self.mutex))
        with guard:
            result = self.sam_predictor.embed_img(self.img, image_format='BGR')
        if result is None:
            return None, None, None, self.img_name
        features, original_size, input_size = result
        return features, original_size, input_size, self.img_name


class Sam2ImgPairEmbeddingWorker(InferenceWorker):
    def __init__(self, sam2, img_pair, do_amg, mutex):
        super().__init__(Sam2EmbeddingWorkerSignals(), len(img_pair))
        self.sam2, self.img_pair, self.do_amg, self.mutex = sam2, img_pair, do_amg, mutex

    def execute(self):
        with locked(self.mutex):
            self.sam2.set_features(self.img_pair)
        return self.do_amg


class Sam2PropagationWorker(InferenceWorker):
    def __init__(self, sam2_predictor, next_annotation, unpropagated_masks,
                 batch_size, track_remaining, mutex):
        if batch_size <= 0:
            raise ValueError('Propagation batch size must be positive')
        super().__init__(Sam2PropagationWorkerSignals(), len(unpropagated_masks))
        self.sam2_predictor = sam2_predictor
        self.next_annotation = next_annotation
        self.unpropagated_masks = list(unpropagated_masks)
        self.batch_size, self.track_remaining, self.mutex = batch_size, track_remaining, mutex

    def execute(self):
        for start in range(0, len(self.unpropagated_masks), self.batch_size):
            batch = self.unpropagated_masks[start:start + self.batch_size]
            if any(mask.mask is None for mask in batch):
                raise ValueError('SAM2 propagation requires segmentation masks')
            with locked(self.mutex):
                masks = self.sam2_predictor.prop_thread_func(batch)
                # Retain the captured destination; never look up the current image here.
                self.next_annotation.add_masks(masks, decision=True)


class AMGWorker(InferenceWorker):
    def __init__(self, sam, mutex):
        super().__init__(WorkerSignals(), 'done')
        self.sam, self.mutex = sam, mutex

    def execute(self):
        with locked(self.mutex):
            return self.sam.amg()


class YoloPredicitonWorker(InferenceWorker):
    """The historic spelling remains a supported public name."""
    def __init__(self, yolo, mutex):
        super().__init__(YoloSignals(), 'done')
        self.yolo, self.mutex = yolo, mutex

    def execute(self):
        with locked(self.mutex):
            return self.yolo.infer_image()


YoloPredictionWorker = YoloPredicitonWorker
