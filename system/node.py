import time
import os
from collections import deque
from threading import Thread
from system.queue import Queue as queue_wrapper
import socket
import yaml
from keras.models import Sequential
from keras import layers
from keras.layers import InputLayer
import numpy as np
import tensorflow as tf

import avro.ipc as ipc
import avro.protocol as protocol

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

# data packet format definition
PROTOCOL = protocol.parse(open(DIR_PATH + '/resource/message/image.avpr').read())


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
            graph: Default graph with Tensorflow backend.
            input_shape: Input shape for model on this node.
            merge: Number of previous layers merged into this layer.
            split: Number of next layers to process current data.
    """

    instance = None

    @classmethod
    def create(cls):
        if cls.instance is None:
            cls.instance = cls()

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
                cls.instance.id = ip

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

                        print model_config[layer_name]['input_shape']

                cls.instance.model = model
                cls.log(cls.instance, 'model finishes', model.summary())

                for ip in system_config['devices']:
                    cls.instance.ip.append(ip)

                cls.instance.merge = system_config['merge']
                cls.instance.split = system_config['split']
                cls.instance.op = system_config['op']
                shape = list(model.input_shape[1:])
                shape[-1] = shape[-1] / cls.instance.merge if cls.instance.op == 'cat' else shape[-1]
                cls.instance.input_shape = tuple(shape)

        return cls.instance

    def __init__(self):
        self.model = None
        self.total_time = 0.0
        self.prepare_data = 0.0
        self.prediction_time = 0.0
        self.input = queue_wrapper()
        self.ip = deque([])
        self.id = ''
        self.debug = False
        self.graph = tf.get_default_graph()
        self.input_shape = None
        self.merge = 0
        self.split = 0
        self.op = ''

        Thread(target=self.inference).start()

    def inference(self):
        # wait for the first packet
        while self.total_time == 0.0:
            time.sleep(0.1)

        while True:
            seq = self.input.dequeue(self.merge)

            if self.op == 'cat':
                X = np.concatenate(seq)
            elif self.op == 'add':
                X = np.add(seq[0], seq[1])
            else:
                X = seq[0]

            if X is not None:
                start = time.time()
                with self.graph.as_default():
                    output = self.model.predict(np.array([X]))
                    for _ in range(self.split):
                        Thread(target=self.send, args=(output,)).start()
                self.prediction_time += time.time() - start

    def receive(self, msg, req):
        start = time.time()
        self.total_time = time.time() if self.total_time == 0.0 else self.total_time

        bytestr = req['input']
        datatype = np.uint8 if req['type'] == 8 else np.float32
        X = np.fromstring(bytestr, datatype).reshape(self.input_shape)
        self.input.enqueue(X)
        self.prepare_data += time.time() - start

    def send(self, X):
        ip = self.ip.popleft()
        self.ip.append(ip)

        client = ipc.HTTPTransceiver(ip, 12345)
        requestor = ipc.Requestor(PROTOCOL, client)

        data = dict()
        data['input'] = X.tobytes()
        data['type'] = 32
        requestor.request('forward', data)

        client.close()

    @property
    def utilization(self):
        return np.float32(self.prediction_time) / (time.time() - self.total_time)

    @property
    def overhead(self):
        return np.float32(self.prepare_data) / (time.time() - self.total_time)

    def stats(self):
        with open(DIR_PATH + '/resource/system/stats.txt', 'w+') as f:
            result = '++++++++++++++++++++++++++++++++++++++++\n'
            result += '+                                      +\n'
            result += '+{:^38s}+\n'.format('SERVER: ' + self.id)
            result += '+                                      +\n'
            result += '+{:>19s}: {:6.3f}           +\n'.format('overhead', self.overhead)
            result += '+{:>19s}: {:6.3f}           +\n'.format('utilization', self.utilization)
            result += self.input.log()
            f.write(result)

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
