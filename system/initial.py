from multiprocessing import Queue
import time
import yaml
import socket
import os

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)


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
            # read ip resources from config file
            with open(DIR_PATH + '/resource/system/config.json') as f:
                configs = yaml.safe_load(f)

                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    s.connect(('8.8.8.8', 80))
                    ip = s.getsockname()[0]
                except Exception:
                    ip = '127.0.0.1'
                finally:
                    s.close()

                config = configs[ip]
                cls.instance.id = ip
                for device in config['devices']:
                    cls.instance.queue.put(device)

            cls.instance = Initializer()
        return cls.instance

    def __init__(self):
        self.queue = Queue()
        self.start = 0.0
        self.count = 0
        self.id = ''

    def receive(self):
        self.start = time.time() if self.start == 0.0 else self.start
        self.count += 1

    def stats(self):
        result = '++++++++++++++++++++++++++++++++++++++++'
        result += '{:^38s}'.format('CLIENT: ' + self.id)
        result += '+                                      +'
        result += '+{:>19s}: {:6.3f}           +'.format('frame rate', self.frame_rate)
        result += '+                                      +'
        result += '++++++++++++++++++++++++++++++++++++++++'

    @property
    def frame_rate(self):
        return self.count / (time.time() - self.start)
