"""Deliver worker callbacks on the GUI thread and reject stale image results."""
from PyQt6.QtCore import QObject, QCoreApplication, QEvent, Qt, pyqtSlot


class TaskReceiver(QObject):
    def __init__(self, owner, worker, valid, result, finished, error):
        super().__init__(owner)
        self.owner, self.worker, self.valid = owner, worker, valid
        self.on_result, self.on_finished, self.on_error = result, finished, error
        self.failed = False

    @pyqtSlot(tuple)
    @pyqtSlot(list)
    @pyqtSlot(bool)
    def result(self, value):
        if self.valid() and self.on_result is not None:
            try:
                self.on_result(value)
            except Exception as error:
                import traceback
                self.error((type(error), error, traceback.format_exc()))

    @pyqtSlot(tuple)
    def error(self, value):
        self.failed = True
        if self.valid() and self.on_error is not None:
            self.on_error(value)

    @pyqtSlot(str)
    @pyqtSlot(int)
    def finished(self, value):
        try:
            if not self.failed and self.valid() and self.on_finished is not None:
                self.on_finished(value)
        except Exception as error:
            import traceback
            self.error((type(error), error, traceback.format_exc()))
        finally:
            self.owner.pending.discard(self)
            self.deleteLater()


class TaskDispatcher(QObject):
    """Own receivers until completion, including results queued after worker exit."""
    def __init__(self, threadpool):
        super().__init__()
        self.threadpool = threadpool
        self.pending = set()
        self.closed = False

    def submit(self, worker, *, valid=lambda: True, result=None, finished=None, error=None):
        if self.closed:
            raise RuntimeError('Task dispatcher is closed')
        receiver = TaskReceiver(self, worker, lambda: not self.closed and valid(), result, finished, error)
        self.pending.add(receiver)
        if hasattr(worker.signals, 'result'):
            worker.signals.result.connect(receiver.result, Qt.ConnectionType.QueuedConnection)
        worker.signals.error.connect(receiver.error, Qt.ConnectionType.QueuedConnection)
        worker.signals.finished.connect(receiver.finished, Qt.ConnectionType.QueuedConnection)
        self.threadpool.start(worker)
        return receiver

    def drain(self):
        """Finish work and deliver callbacks without processing user input events.

        Callbacks may enqueue follow-up inference, so repeat until both queues
        are empty. This is used at image/model transitions, never for prefetch.
        """
        if self.pending and QCoreApplication.instance() is None:
            raise RuntimeError("A Qt application is required to deliver worker results")
        while self.pending or self.threadpool.activeThreadCount():
            self.threadpool.waitForDone(-1)
            for receiver in tuple(self.pending):
                QCoreApplication.sendPostedEvents(receiver, QEvent.Type.MetaCall)

    def close(self):
        self.closed = True
        self.drain()
