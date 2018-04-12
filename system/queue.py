"""
    Customized wrapper for Python deque data structure.
"""
from collections import deque
import numpy as np
import time


class Queue:
    """
        A wrapper class for deque data structure, which will limit the max size
        of deque and also report deque overflow frequency.

        Attributes:
            op: Total operations.
            over: # enqueue that will cause queue exceeds its max size.
            under: # dequeue on an empty queue.
    """
    def __init__(self, size):
        self.max_size = size
        self.enqueue_op = 0
        self.dequeue_op = 0
        self.over = 0
        self.under = 0
        self.queue = deque()

    def enqueue(self, data):
        self.enqueue_op += 1
        if len(self.queue) < self.max_size:
            self.queue.append(data)
        else:
            self.over += 1

    def dequeue(self):
        self.dequeue_op += 1
        self.under += 1 if len(self.queue) == 0 else 0
        while len(self.queue) == 0:
            time.sleep(0.1)
        return self.queue.popleft()

    def force_enqueue(self, data):
        """
            Replace the elements in deque by new data.

            Arguments:
                data: a list of data elements.
        """
        self.enqueue_op += len(data)
        self.over += len(data) + len(self.queue) - self.max_size
        if len(data) > self.max_size:
            self.queue = deque(data[-self.max_size:])
        else:
            queue = deque(data)
            for _ in range(self.max_size - len(data)):
                queue.appendleft(self.queue.pop())
            self.queue = queue

    @property
    def overflow(self):
        return np.float32(self.over) / self.enqueue_op

    @property
    def underflow(self):
        return np.float32(self.under) / self.dequeue_op

    def log(self):
        print 'overflow:  {:.1f} %'.format(self.overflow * 100)
        print 'underflow: {:.1f} %'.format(self.underflow * 100)
