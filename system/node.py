import time
import os
from multiprocessing import Lock, Queue
from threading import Thread
from system.queue import Queue as queue_wrapper
import socket
import yaml
from keras.models import Sequential
from keras import layers
from keras.layers import InputLayer
import numpy as np

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)


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
            input: Store the data packets from other nodes.
            ip: Store all IP addresses of available devices.
            debug: If print out verbose information.
            lock: Ensure the node integrity.
    """

    instance = None

    @classmethod
    def create(cls, queue_size):
        if cls.instance is None:
            cls.instance = cls(queue_size)

            # Get ip address and create model according to ip config file.
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                ip = s.getsockname()[0]
            except Exception:
                ip = '127.0.0.1'
            finally:
                s.close()

            with open(DIR_PATH + '/resource/system/config.json') as f:
                system_config = yaml.safe_load(f)[ip]

                model = Sequential()

                # The model config is predefined. Extract each layer's config
                # according to the config from system config.
                with open(DIR_PATH + '/resource/model/config.json') as f2:
                    model_config = yaml.safe_load(f2)
                    for layer_name in system_config['model']:
                        class_name = model_config[layer_name]['class_name']
                        config = model_config[layer_name]['config']
                        input_shape = model_config[layer_name]['input_shape']
                        layer = layers.deserialize({
                            'class_name': class_name,
                            'config': config
                        })
                        model.add(InputLayer(input_shape))
                        model.add(layer)

                cls.instance.model = model

                for ip in system_config['devices']:
                    cls.instance.ip.put(ip)

        return cls.instance

    def __init__(self, queue_size):
        self.model = None
        self.total_time = 0.0
        self.utilization_time = 0.0
        self.prediction_time = 0.0
        self.input = queue_wrapper(queue_size)
        self.ip = Queue()
        self.debug = False
        self.lock = Lock()

    def inference(self, X):
        start = time.time()
        X = self.model.predict(np.array([X]))
        self.log('prediction completes')
        self.prediction_time += time.time() - start
        Thread(target=self.send, args=(X,)).start()

    def receive(self, msg, req):
        self.acquire_lock()
        start = time.time()
        self.total_time = time.time() if self.total_time == 0.0 else self.total_time

        self.log('node gets data')

        bytestr = req['input']
        datatype = np.uint8 if req['type'] == 8 else np.float32

        self.log('nodes data assembling finishes')

        input_shape = self.model.input_shape
        X = np.fromstring(bytestr, datatype).reshape(input_shape)
        self.inference(X)

        self.log('inference finishes')

        self.utilization_time += time.time() - start
        self.release_lock()

    def send(self, X):
        # TODO: send output to next layer.
        pass

    def utilization(self):
        return self.utilization_time / (time.time() - self.total_time)

    def overhead(self):
        return (self.utilization_time - self.prediction_time) / self.utilization_time

    def acquire_lock(self):
        self.lock.acquire()

    def release_lock(self):
        self.lock.release()

    def log(self, step, data=''):
        """
            Log function for debug. Turn the flag on to show each step result.
            Args:
                step: Each step names.
                data: Data format or size.
        """
        if self.debug:
            print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
            for k in range(0, len(step), 68):
                print '+{:^68.68}+'.format(step[k:k + 68])
            for k in range(0, len(data), 68):
                print '+{:^68.68}+'.format(data[k:k + 68])
            print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
            print
