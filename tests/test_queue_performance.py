from system.queue import Queue
from collections import deque
import time


def test_performance():
    q = Queue(10)
    start = time.time()
    for n in range(20):
        q.enqueue(n)
    q.dequeue(5)
    q.dequeue(10)
    customized_queue_time = time.time() - start

    dq = deque([], 10)
    start = time.time()
    for n in range(20):
        dq.append(n)
    for n in range(5):
        dq.popleft()
    for n in range(10):
        dq.popleft()
    deque_time = time.time() - start

    assert deque_time - customized_queue_time < 10