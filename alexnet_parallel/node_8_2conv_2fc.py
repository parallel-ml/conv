import argparse
import os
import time
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
from collections import deque
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


"""
    Alexnet is divided into several blocks in 8 nodes config
    block1: conv1
    block2: conv2, conv3, conv4, conv5, flatten
    block3: fc1
    block4: fc2, fc3
"""


class Node(object):
    """ singleton factory with threading safe lock.

    Attributes:
        ip: A dictionary contains Queue of ip addresses for different models type.
        model: loaded models associated to a node.
        graph: default graph used by Tensorflow
        debug: flag for debugging
        lock: threading lock for safe usage of this class. The lock is used
                for safe models forwarding. If the models is processing input and
                it gets request from other devices, the new request will wait
                until the previous models forwarding finishes.
        name: model name
        total: total time of getting frames
        count: total number of frames gets back

    """

    instance = None

    def __init__(self):
        self.ip = dict()
        self.model = None
        self.graph = tf.get_default_graph()
        self.debug = False
        self.lock = Lock()
        self.name = 'unknown'
        self.total = 0
        self.count = 1

    def log(self, step, data=''):
        if self.debug:
            util.step(step, data)

    def acquire_lock(self):
        self.lock.acquire()

    def release_lock(self):
        self.lock.release()

    def timer(self, interval):
        self.total += interval
        print '{:s}: {:.3f}'.format(self.name, self.total / self.count)
        self.count += 1

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
        of each node. All models forwarding and output redirect are done here.

        Args:
            msg: meta data
            req: contains data packet

        Returns:
            a string of data

        Raises:
            AvroException: if the data does not have correct syntac defined in Schema

        """
        node = Node.create()
        node.acquire_lock()

        if msg.name == 'forward':
            try:
                with node.graph.as_default():
                    bytestr = req['input']
                    if req['next'] == 'block1':
                        pass

                    elif req['next'] == 'block2':
                        pass

                    elif req['next'] == 'block3':
                        pass

                    else:
                        pass

                node.release_lock()
                return

            except Exception, e:
                node.log('Error', e.message)
        else:
            raise schema.AvroException('unexpected message:', msg.getname())

    def send(self, X, name, tag):
        """ send data to other devices

        Send data to other devices. The data packet contains data and models name.
        Ip address of next device pop from Queue of a ip list.

        Args:
             X: numpy array
             name: next device models name
             tag: mark the current layer label

        """
        node = Node.create()
        queue = node.ip[name]
        address = queue.get()

        port = 9999 if name == 'initial' else 12345
        client = ipc.HTTPTransceiver(address, port)
        requestor = ipc.Requestor(PROTOCOL, client)

        node.name = name

        data = dict()
        data['input'] = X.tostring()
        data['next'] = name
        data['tag'] = tag
        node.log('finish assembly')
        start = time.time()
        requestor.request('forward', data)
        end = time.time()
        node.timer(end - start)

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

    node.debug = cmd.debug

    # read ip resources from config file
    with open('resource/ip') as file:
        address = yaml.safe_load(file)
        pass

    server = ThreadedHTTPServer(('0.0.0.0', 12345), Handler)
    server.allow_reuse_address = True
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='Set to debug mode')
    cmd = parser.parse_args()
    main(cmd)
