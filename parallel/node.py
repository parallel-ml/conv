import argparse
import os
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
from multiprocessing import Queue
from threading import Thread, Lock

import avro.ipc as ipc
import avro.protocol as protocol
import avro.schema as schema
import matplotlib
import numpy as np
import tensorflow as tf
import yaml

import model as ml
import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

matplotlib.use('Agg')

PROTOCOL = protocol.parse(open('resource/image.avpr').read())


class Node(object):
    """ singleton factory with threading safe lock.

    Attributes:
        ip: A dictionary contains Queue of ip addresses for different model type.
        model: loaded model associated to a node.
        graph: default graph used by Tensorflow
        fc_layer_dim: dimension of fully connected layer
        max_layer_dim: dimension of max pooling layer
        debug: flag for debugging
        fc_input: input for fully connected layer
        max_input: input for max pooling layer
        result_q: Queue for put result
        lock: threading lock for safe usage of this class. The lock is used
                for safe model forwarding. If the model is processing input and
                it gets request from other devices, the new request will wait
                until the previous model forwarding finishes.

    """

    instance = None

    def __init__(self):
        self.ip = dict()
        self.model = None
        self.graph = tf.get_default_graph()
        self.fc_layer_dim = 7680
        self.max_layer_dim = 16
        self.debug = False
        self.fc_input = None
        self.max_input = None
        self.result_q = Queue()
        self.lock = Lock()

    def log(self, step, data=''):
        if self.debug:
            util.step(step, data)

    def acquire_lock(self):
        self.lock.acquire()

    def release_lock(self):
        self.lock.release()

    @classmethod
    def create(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance


class Responder(ipc.Responder):
    """ responder called by handler when got request """

    def __init__(self):
        ipc.Responder.__init__(self, PROTOCOL)

    def invoke(self, msg, req):
        """ process response

        invoke handles the request and get response for the request. This is the key
        of each node. All model forwarding and output redirect are done here.

        Args:
            msg: meta data
            req: contains data packet

        Returns:
            a string of data

        Raises:
            AvroException: if the data does not have correct syntac defined in Schema

        """
        node = Node.create()

        if msg.name == 'forward':
            try:
                with node.graph.as_default():
                    bytestr = req['input']

                    if req['name'] == 'spatial':
                        node.acquire_lock()
                        node.log('get spatial request')
                        X = np.fromstring(bytestr, np.uint8).reshape(12, 16, 3)
                        node.model = ml.load_spatial() # if node.model is None else node.model
                        output = node.model.predict(np.array([X]))
                        node.release_lock()
                        node.log('finish spatial forward')
                        Thread(target=self.send, args=(output, 'fc')).start()
                        return

                    elif req['name'] == 'temporal':
                        node.acquire_lock()
                        node.log('get temporal request')
                        X = np.fromstring(bytestr, np.float32).reshape(12, 16, 6)
                        node.model = ml.load_temporal() # if node.model is None else node.model
                        output = node.model.predict(np.array([X]))
                        node.release_lock()
                        node.log('finish temporal forward')
                        Thread(target=self.send, args=(output, 'fc')).start()
                        return

                    elif req['name'] == 'maxpool':
                        node.acquire_lock()
                        node.log('get max pool request')
                        X = np.fromstring(bytestr, np.uint8)
                        X = X.reshape(1, X.size)
                        node.max_input = X if node.max_input is None else np.concatenate((node.max_input, X), axis=0)
                        if node.max_input.shape[0] < node.max_layer_dim:
                            return ' '
                        node.model = ml.load_maxpool(N=node.max_layer_dim) # if node.model is None else node.model
                        output = node.model.predict(np.array([node.max_input]))
                        node.release_lock()
                        node.max_input = None
                        node.log('max pool forward')
                        Thread(target=self.send, args=(output, 'fc')).start()
                        return

                    elif req['name'] == 'fc':
                        node.acquire_lock()
                        X = np.fromstring(bytestr, np.float32)
                        X = X.reshape(X.size)
                        # concatenate inputs from spatial and temporal
                        # ex: (1, 256) + (1, 256) = (1, 512)
                        node.fc_input = X if node.fc_input is None else np.concatenate((node.fc_input, X))
                        node.log('get FC request', node.fc_input.shape)
                        if node.fc_input.size < node.fc_layer_dim:
                            node.release_lock()
                            return ' '
                        node.model = ml.load_fc(node.fc_layer_dim) # if node.model is None else node.model
                        output = node.model.predict(np.array([node.fc_input]))
                        node.fc_input = None
                        node.log('finish FC forward')
                        node.release_lock()
                        Thread(target=self.send, args=(output, 'initial')).start()
                        return

            except Exception, e:
                node.log('Error', e.message)
        else:
            raise schema.AvroException('unexpected message:', msg.getname())

    def send(self, X, name):
        """ send data to other devices

        Send data to other devices. The data packet contains data and model name.
        Ip address of next device pop from Queue of a ip list.

        Args:
             X: numpy array
             name: next device model name

        """
        node = Node.create()
        queue = node.ip[name]
        address = queue.get()

        port = 12345
        if node.debug:
            port = 9999 if name == 'initial' else 12345
        client = ipc.HTTPTransceiver(address, port)
        requestor = ipc.Requestor(PROTOCOL, client)

        data = dict()
        data['input'] = X.tostring()
        data['name'] = name
        node.log('finish assembly')
        output = requestor.request('forward', data)
        node.result_q.put(output)
        node.log('node gets request back')
        client.close()
        queue.put(address)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """ handle request from other devices.

        do_POST is automatically called by ThreadedHTTPServer. It creates a new
        responder for each request. The responder generates response and write
        response to data sent back.

        """
        self.responder = Responder()
        call_request_reader = ipc.FramedReader(self.rfile)
        call_request = call_request_reader.read_framed_message()
        resp_body = self.responder.respond(call_request)
        self.send_response(200)
        self.send_header('Content-Type', 'avro/binary')
        self.end_headers()
        resp_writer = ipc.FramedWriter(self.wfile)
        resp_writer.write_framed_message(resp_body)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """handle requests in separate thread"""


def main(cmd):
    node = Node.create()

    node.fc_layer_dim = cmd.fc_dim
    node.max_layer_dim = cmd.max_dim
    node.debug = cmd.debug

    # read ip resources from config file
    with open('resource/ip') as file:
        address = yaml.safe_load(file)
        node.ip['fc'] = Queue()
        node.ip['maxpool'] = Queue()
        node.ip['initial'] = Queue()
        for addr in address['fc']:
            if addr == '#':
                break
            node.ip['fc'].put(addr)
        for addr in address['maxpool']:
            if addr == '#':
                break
            node.ip['maxpool'].put(addr)
        for addr in address['initial']:
            if addr == '#':
                break
            node.ip['initial'].put(addr)

    server = ThreadedHTTPServer(('0.0.0.0', 12345), Handler)
    server.allow_reuse_address = True
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fc_dim', metavar='\b', action='store', default=7680, type=int,
                        help='Choose fc layer input dimension')
    parser.add_argument('--max_dim', metavar='\b', action='store', default=16, type=int,
                        help='Choose maxpooling layer input dimension')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='Set to debug mode')
    cmd = parser.parse_args()
    main(cmd)
