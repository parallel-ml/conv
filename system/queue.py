"""
    Customized wrapper for Python deque data structure.
"""
from collections import deque, defaultdict
import time
from threading import Thread


class Queue:
    """
        A wrapper class for deque data structure, which will limit the max size
        of deque and also report deque overflow frequency.

        Attributes:
            op: Total operations.
            over: # enqueue that will cause queue exceeds its max size.
            under: # dequeue on an empty queue.
    """
    def __init__(self, size=10):
        self.size = size
        self.queue = deque()
        self.table = dict()
        for i in range(size + 1):
            self.table[i] = 0
        Thread(target=self.stats).start()

    def enqueue(self, data):
        if len(self.queue) < self.size:
            self.queue.append(data)

    def dequeue(self):
        while len(self.queue) == 0:
            time.sleep(0.1)
        return self.queue.popleft()

    def log(self):
        result = '++++++++++++++++++++++++++++++++++++++++\n'
        result += '+                                      +\n'
        total = sum(self.table.values())
        for key, value in self.table.items():
            result += '+{:>19d}: {:6.3f}           +\n'.format(key, value * 1.0 / total)
        result += '+                                      +\n'
        result = '++++++++++++++++++++++++++++++++++++++++\n'
        return result

    def stats(self):
        while True:
            self.table[len(self.queue)] += 1
            time.sleep(0.001)

