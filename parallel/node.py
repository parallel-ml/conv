import argparse
import os
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
from multiprocessing import Queue
from threading import Thread

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


class Singleton(object):
    """ implementation of singleton design, idea is always return a same class at module level """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance


class Node(Singleton):
    """ class wraps all data used by a node on 1 device """

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

    def log(self, step, data=''):
        if self.debug:
            util.step(step, data)


class Responder(ipc.Responder):
    """ responder called by handler when got request """

    def __init__(self):
        ipc.Responder.__init__(self, PROTOCOL)

    def invoke(self, msg, req):
        node = Node()

        if msg.name == 'forward':
            try:
                with node.graph.as_default():
                    bytestr = req['input']

                    if req['name'] == 'spatial':
                        node.log('get spatial request')
                        X = np.fromstring(bytestr, np.uint8).reshape(12, 16, 3)
                        node.model = ml.load_spatial() if node.model is None else node.model
                        output = node.model.predict(np.array([X]))
                        node.log('finish spatial forward')
                        Thread(target=self.send, args=(output, 'fc')).start()
                        return node.result_q.get()

                    elif req['name'] == 'temporal':
                        node.log('get temporal request')
                        X = np.fromstring(bytestr, np.float32).reshape(12, 16, 6)
                        node.model = ml.load_temporal() if node.model is None else node.model
                        output = node.model.predict(np.array([X]))
                        node.log('finish temporal forward')
                        Thread(target=self.send, args=(output, 'fc')).start()
                        return node.result_q.get()

                    elif req['name'] == 'maxpool':
                        node.log('get max pool request')
                        X = np.fromstring(bytestr, np.uint8)
                        X = X.reshape(1, X.size)
                        node.max_input = X if node.max_input is None else np.concatenate((node.max_input, X), axis=0)
                        if node.max_input.shape[0] < node.max_layer_dim:
                            return ' '
                        node.model = ml.load_maxpool(N=node.max_layer_dim) if node.model is None else node.model
                        output = node.model.predict(np.array([node.max_input]))
                        node.max_input = None
                        node.log('max pool forward')
                        Thread(target=self.send, args=(output, 'fc')).start()
                        return node.result_q.get()

                    elif req['name'] == 'fc':
                        node.log('get FC request')
                        X = np.fromstring(bytestr, np.float32)
                        X = X.reshape(X.size)
                        node.fc_input = X if node.fc_input is None else np.concatenate((node.fc_input, X))
                        if node.fc_input.size < node.fc_layer_dim:
                            return ' '
                        node.model = ml.load_fc(node.fc_layer_dim) if node.model is None else node.model
                        output = node.model.predict(np.array([node.fc_input]))
                        node.fc_input = None
                        node.log('finish FC forward')
                        return output.tobytes()

            except Exception, e:
                node.log('Error', str(e))
        else:
            raise schema.AvroException('unexpected message:', msg.getname())

    def send(self, X, name):
        node = Node()
        queue = node.ip[name]
        address = queue.get()
        client = ipc.HTTPTransceiver(address, 12345)
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
    node = Node()

    node.fc_layer_dim = cmd.fc_dim
    node.max_layer_dim = cmd.max_dim
    node.debug = cmd.debug

    with open('resource/ip') as file:
        address = yaml.safe_load(file)
        node.ip['fc'] = Queue()
        node.ip['maxpool'] = Queue()
        for addr in address['fc']:
            node.ip['fc'].put(addr)
        for addr in address['maxpool']:
            node.ip['maxpool'].put(addr)

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
