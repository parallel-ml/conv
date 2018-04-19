from multiprocessing import Queue
import time


class Initializer:
    """
        Singleton factory for initializer. The Initializer module has two timers.
        The node_timer is for recording statistics for block1 layer model inference
        time. The timer is for recording the total inference time from last
        fully connected layer.
        Attributes:
            queue: Queue for storing available block1 models devices.
    """
    instance = None

    @classmethod
    def create_init(cls):
        """ Utilize singleton design pattern to create single instance. """
        if cls.instance is None:
            cls.instance = Initializer()
        return cls.instance

    def __init__(self):
        self.queue = Queue()
        self.start = 0.0
        self.count = 0

    def receive(self):
        self.start = time.time() if self.start == 0.0 else self.start
        self.count += 1

    def stats(self):
        print '++++++++++++++++++++++++++++++++++++++++'
        print '+                                      +'
        print '+{:>19s}: {:6.3f}           +'.format('frame rate', self.frame_rate)
        print '+                                      +'
        print '++++++++++++++++++++++++++++++++++++++++'

    @property
    def frame_rate(self):
        return self.count / (time.time() - self.start)
