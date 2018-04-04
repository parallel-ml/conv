from utils.queue import Queue


def test_simple_queue():
    q = Queue(2)
    q.force_enqueue([1, 2, 3, 4, 5])
    for _ in range(5):
        q.dequeue()
    assert q.overflow == 0.6
    assert q.underflow == 0.6
