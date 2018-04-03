"""
    Customized wrapper for Python deque data structure.
"""
from collections import deque


class Queue:
    """
        A wrapper class for deque data structure, which will limit the max size
        of deque and also report deque overflow frequency.

        Attributes:
            op: Total operations.
            overflow: # enqueue that will cause queue exceeds its max size.
            underflow: # dequeue on an empty queue.
    """
    def __init__(self, size):
        self.max_size = size
        self.op = 0
        self.overflow = 0
        self.underflow = 0
        self.queue = deque()

    def enqueue(self, data):
        self.op += 1
        if len(self.queue) < self.max_size:
            self.queue.append(data)
        else:
            self.overflow += 1

    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.popleft()
        self.underflow += 1

    def force_enqueue(self, data):
        """
            Replace the elements in deque by new data.

            Arguments:
                data: a list of data elements.
        """
        self.op += len(data)
        self.overflow += len(data) + len(self.queue) - self.max_size
        if len(data) > self.max_size:
            self.queue = deque(data[-self.max_size:])
        else:
            queue = deque(data)
            for _ in range(self.max_size - len(data)):
                queue.appendleft(self.queue.pop())
            self.queue = queue

    def log(self):
        print 'overflow:  {:.1f} %'.format(1.0 * self.overflow / self.op * 100)
        print 'underflow: {:.1f} %'.format(1.0 * self.underflow / self.op * 100)
