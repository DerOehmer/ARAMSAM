import unittest
from unittest.mock import Mock, patch

import numpy as np
from PyQt6.QtCore import QMutex, QThread, QThreadPool
from PyQt6.QtWidgets import QApplication

from aramsam_annotator.backend.tasks import TaskDispatcher
from aramsam_annotator.workers import AMGWorker, Sam2ImgPairEmbeddingWorker, Sam2PropagationWorker, YoloPredicitonWorker


class BackendTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_all_worker_failures_release_mutex_and_emit_completion(self):
        factories = [
            lambda model, mutex: AMGWorker(model, mutex),
            lambda model, mutex: Sam2ImgPairEmbeddingWorker(model, [np.zeros((2, 2, 3), np.uint8)], False, mutex),
            lambda model, mutex: YoloPredicitonWorker(model, mutex),
            lambda model, mutex: Sam2PropagationWorker(model, Mock(), [Mock(mask=np.ones((2, 2)))], 1, False, mutex),
        ]
        for factory in factories:
            mutex, model = QMutex(), Mock()
            for name in ('amg', 'set_features', 'infer_image', 'prop_thread_func'):
                getattr(model, name).side_effect = RuntimeError('model failed')
            worker = factory(model, mutex)
            errors, completions = [], []
            worker.signals.error.connect(errors.append)
            worker.signals.finished.connect(completions.append)
            with patch('aramsam_annotator.workers.traceback.print_exc'):
                worker.run()
            self.assertEqual(len(errors), 1)
            self.assertEqual(len(completions), 1)
            self.assertTrue(mutex.tryLock())
            mutex.unlock()

    def test_real_thread_results_are_delivered_on_gui_thread(self):
        pool = QThreadPool()
        dispatcher = TaskDispatcher(pool)
        model = Mock()
        model.amg.return_value = ([], np.zeros((2, 2, 3), np.uint8))
        received = []
        dispatcher.submit(AMGWorker(model, QMutex()), result=lambda result: received.append(QThread.currentThread()))
        dispatcher.drain()
        self.assertEqual(received, [self.application.thread()])
        self.assertFalse(dispatcher.pending)

    def test_stale_results_are_discarded_even_when_worker_already_finished(self):
        pool = QThreadPool()
        dispatcher = TaskDispatcher(pool)
        model = Mock()
        model.amg.return_value = ([], None)
        valid = [True]
        result, finished = Mock(), Mock()
        dispatcher.submit(AMGWorker(model, QMutex()), valid=lambda: valid[0], result=result, finished=finished)
        pool.waitForDone(-1)
        valid[0] = False
        dispatcher.drain()
        result.assert_not_called()
        finished.assert_not_called()
        self.assertFalse(dispatcher.pending)

    def test_drain_includes_followup_work(self):
        dispatcher = TaskDispatcher(QThreadPool())
        model = Mock()
        model.infer_image.return_value = []
        result = Mock()
        def next_task(_):
            dispatcher.submit(YoloPredicitonWorker(model, QMutex()), result=result)
        dispatcher.submit(YoloPredicitonWorker(model, QMutex()), result=next_task)
        dispatcher.drain()
        result.assert_called_once_with([])

    def test_callback_failure_is_reported_and_receiver_released(self):
        dispatcher = TaskDispatcher(QThreadPool())
        model = Mock()
        model.infer_image.return_value = []
        error, finished = Mock(), Mock()
        dispatcher.submit(YoloPredicitonWorker(model, QMutex()),
                          result=Mock(side_effect=ValueError('bad result')), error=error, finished=finished)
        dispatcher.drain()
        error.assert_called_once()
        finished.assert_not_called()
        self.assertFalse(dispatcher.pending)

    def test_close_discards_pending_results_and_rejects_new_work(self):
        dispatcher = TaskDispatcher(QThreadPool())
        model = Mock()
        model.infer_image.return_value = []
        result = Mock()
        dispatcher.submit(YoloPredicitonWorker(model, QMutex()), result=result)
        dispatcher.close()
        result.assert_not_called()
        with self.assertRaises(RuntimeError):
            dispatcher.submit(YoloPredicitonWorker(model, QMutex()))


if __name__ == '__main__':
    unittest.main()
