from collections import deque
from multiprocessing import Queue
from threading import Thread
import time
from SocketServer import ThreadingMixIn
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

import avro.ipc as ipc
import avro.protocol as protocol
import avro.schema as schema
import cv2
import numpy as np
import yaml

PROTOCOL = protocol.parse(open('resource/image.avpr').read())


class Initializer:
    instance = None

    def __init__(self):
        self.spatial_q = Queue()
        self.temporal_q = Queue()
        self.flows = deque()

    @classmethod
    def create_init(cls):
        if cls.instance is None:
            cls.instance = Initializer()
        return cls.instance


def send_request(bytestr, mode='spatial'):
    init = Initializer.create_init()
    queue = init.spatial_q if mode == 'spatial' else init.temporal_q

    addr = queue.get()
    client = ipc.HTTPTransceiver(addr, 12345)
    requestor = ipc.Requestor(PROTOCOL, client)

    data = dict()
    data['input'] = bytestr
    data['name'] = mode

    requestor.request('forward', data)

    client.close()
    queue.put(addr)


def master():
    init = Initializer.create_init()
    frame0 = None
    while True:
        ret, frame = 'unknown', np.random.rand(12, 16, 3) * 255
        frame = frame.astype(dtype=np.uint8)
        image = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame0 is not None:
            init.flows.appendleft(cv2.calcOpticalFlowFarneback(frame0, frame, None, 0.5, 3, 4, 3, 5, 1.1, 0))
            if len(init.flows) == 3:
                Thread(target=send_request, args=(image.tobytes(), 'spatial')).start()
                optical_flow = np.concatenate(init.flows, axis=2)
                Thread(target=send_request, args=(optical_flow.tobytes(), 'temporal')).start()
                init.flows.pop()

        frame0 = frame
        time.sleep(1)


class Responder(ipc.Responder):
    def __init__(self):
        ipc.Responder.__init__(self, PROTOCOL)

    def invoke(self, msg, req):
        if msg.name == 'forward':
            print 'gets response'
            try:
                bytestr = req['input']
                X = np.fromstring(bytestr, np.float32).reshape(1, 51)
                print X
                return
            except Exception, e:
                print 'Error', e.message
        else:
            raise schema.AvroException('unexpected message:', msg.getname())


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


def main():
    init = Initializer.create_init()
    with open('resource/ip') as file:
        address = yaml.safe_load(file)
        for addr in address['spatial']:
            if addr == '#':
                break
            init.spatial_q.put(addr)
        for addr in address['temporal']:
            if addr == '#':
                break
            init.temporal_q.put(addr)

    server = ThreadedHTTPServer(('0.0.0.0', 9999), Handler)
    server.allow_reuse_address = True
    Thread(target=server.serve_forever, args=()).start()

    master()


if __name__ == '__main__':
    main()
