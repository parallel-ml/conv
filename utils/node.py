import time
from multiprocessing import Lock
from queue import Queue


class Node:
    """
        Node class for handling model prediction and get according stats
        for a module.

        Attributes:
            instance: Class attributes to achieve singleton for this node
                        class.
            model: The model created for this node.
            total_time: Total timing of one data packet from being received
                        to being successfully processed.
            prediction_time: Total timing of model inference.
            lock: Ensure the node integrity.
            input: Store the data packets from other nodes.
    """

    instance = None

    @classmethod
    def create(cls, queue_size):
        if cls.instance is None:
            cls.instance = cls(queue_size)
        return cls.instance

    def __init__(self, queue_size):
        self.model = None
        self.total_time = 0.0
        self.utilization_time = 0.0
        self.prediction_time = 0.0
        self.lock = Lock()
        self.input = Queue(queue_size)

    def inference(self):
        # TODO: create model here.

        start = time.time()

        # TODO: do model inference here.

        self.prediction_time += time.time() - start

    def receive(self, msg, req):
        self.acquire_lock()
        start = time.time()
        self.total_time = time.time() if self.total_time == 0.0 else self.total_time

        # TODO: reassemble data packets.

        self.utilization_time += time.time() - start
        self.release_lock()

    def utilization(self):
        return self.utilization_time / (time.time() - self.total_time)

    def overhead(self):
        return (self.utilization_time - self.prediction_time) / self.utilization_time

    def acquire_lock(self):
        self.lock.acquire()

    def release_lock(self):
        self.lock.release()
