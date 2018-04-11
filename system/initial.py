from multiprocessing import Queue


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

    def __init__(self):
        self.queue = Queue()

    @classmethod
    def create_init(cls):
        """ Utilize singleton design pattern to create single instance. """
        if cls.instance is None:
            cls.instance = Initializer()
        return cls.instance
