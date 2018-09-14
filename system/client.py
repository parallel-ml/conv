"""
    This module is called initial because it initializes all request
    from this node. It will simulates a (224, 224, 3) size image data
    packet and send to the first node in the distributed system and wait
    for the response from the last layer.
"""
import time
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
import threading
from threading import Thread
import os
import signal
from collections import deque

import avro.ipc as ipc
import avro.protocol as protocol
import avro.schema as schema
import numpy as np

from initial import Initializer

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

# data packet format definition
PROTOCOL = protocol.parse(open(DIR_PATH + '/resource/message/image.avpr').read())


def send_request(frame):
    """
        This function sends data to next layer. It will pop an available
        next layer device IP address defined at IP table, and send data
        to that IP. After, it will put the available IP back.
        Args:
            bytestr: The encoded byte string for image.
            mode: Specify next layer option.
    """
    init = Initializer.create()
    queue = init.queue

    addr = queue.get()
    client = ipc.HTTPTransceiver(addr, 12345)
    requestor = ipc.Requestor(PROTOCOL, client)

    data = dict()
    tmp = [str(entry) for entry in frame.shape]
    data['shape'] = ''.join(tmp)
    data['input'] = frame.astype(np.uint8).tobytes()
    data['type'] = 8
    requestor.request('forward', data)

    client.close()
    queue.put(addr)


def master():
    """
        Master function for real time model inference. A basic while loop
        gets one frame at each time. It appends a frame to deque every time
        and pop the least recent one if the length > maximum.
    """
    init = Initializer.create()

    while True:
        # current frame
        ret, frame = 'unknown', np.random.rand(220, 220, 3) * 255
        for _ in range(init.split):
            Thread(target=send_request, args=(frame,)).start()
        time.sleep(0.03)


class Responder(ipc.Responder):
    def __init__(self):
        ipc.Responder.__init__(self, PROTOCOL)

    def invoke(self, msg, req):
        """
            This functino is invoked by do_POST to handle the request. Invoke handles
            the request and get response for the request. This is the key of each node.
            All models forwarding and output redirect are done here. Because the invoke
            method of initializer only needs to receive the data packet, it does not do
            anything in the function and return None.
            Args:
                msg: Meta data.
                req: Contains data packet.
            Returns:
                None
            Raises:
                AvroException: if the data does not have correct syntac defined in Schema
        """
        if msg.name == 'forward':
            init = Initializer.create()
            try:
                init.receive()
                return
            except Exception, e:
                print 'Error', e.message
        else:
            raise schema.AvroException('unexpected message:', msg.getname())


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """
            Handle request from other devices.
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
    """ Handle requests in separate thread. """


def main():
    signal.signal(signal.SIGTERM, terminate)

    server = ThreadedHTTPServer(('0.0.0.0', 12345), Handler)
    server.allow_reuse_address = True
    Thread(target=server.serve_forever, args=()).start()

    master()


def terminate(signum, stack):
    init = Initializer.create()
    init.terminate()
    for thread in threading.enumerate():
        if thread != threading.current_thread():
            thread.join()


if __name__ == '__main__':
    main()
