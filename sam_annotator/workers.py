from PyQt6.QtCore import (
    QRunnable,
    pyqtSignal,
    QObject,
    QMutex,
    pyqtSlot,
)

from sam_annotator.run_sam import (
    BackgroundThreadSamPredictor,
    Sam2Inference,
    SamInference,
)
from sam_annotator.mask_visualizations import AnnotationObject, MaskData
import numpy as np
import time
import sys
import traceback


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
        self.signals = Sam2PropagationWorkerSignals()
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
        do_amg: bool,
        mutex: QMutex,
    ):
        super().__init__()
        self.signals = Sam2EmbeddingWorkerSignals()
        self.sam2 = sam2
        self.img_pair = img_pair
        self.do_amg = do_amg
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

        else:
            self.signals.result.emit(self.do_amg)

        finally:
            self.signals.finished.emit(len(self.img_pair))


class AMGWorker(QRunnable):

    def __init__(
        self,
        sam: Sam2Inference | SamInference,
        mutex: QMutex,
    ):
        super().__init__()
        self.signals = WorkerSignals()
        self.sam = sam
        self.mutex = mutex

    @pyqtSlot()
    def run(self):
        try:
            self.mutex.lock()
            now = time.time()
            mask_objs, annotated_image = self.sam.amg()
            duration = time.time() - now
            print(f"AMG took {duration} s")
            result = (mask_objs, annotated_image)
            self.mutex.unlock()

        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))

        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit("done")


class WorkerSignals(QObject):
    finished: pyqtSignal = pyqtSignal(str)
    error: pyqtSignal = pyqtSignal(tuple)
    result: pyqtSignal = pyqtSignal(tuple)


class Sam2EmbeddingWorkerSignals(QObject):
    finished: pyqtSignal = pyqtSignal(int)
    error: pyqtSignal = pyqtSignal(tuple)
    result: pyqtSignal = pyqtSignal(bool)


class Sam2PropagationWorkerSignals(QObject):
    finished: pyqtSignal = pyqtSignal(int)
    error: pyqtSignal = pyqtSignal(tuple)
