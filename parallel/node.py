import argparse
import os
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
import yaml
from multiprocessing import Queue
from threading import Thread

import avro.ipc as ipc
import avro.protocol as protocol
import avro.schema as schema
import matplotlib
import numpy as np
import tensorflow as tf
import util
import model as ml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

matplotlib.use('Agg')

PROTOCOL = protocol.parse(open('resource/image.avpr').read())

# global variable declaration
ip, model, graph, fc_dim, max_dim, debug = dict(), None, None, 7680, 16, False

fc_input, max_input = None, None

result_q = Queue()


class Responder(ipc.Responder):
    def __init__(self):
        ipc.Responder.__init__(self, PROTOCOL)

    def invoke(self, msg, req):
        if debug:
            util.step('gets message', req['name'])

        if msg.name == 'forward':
            try:
                global graph, model, fc_dim, max_dim, result_q, fc_input, max_input
                with graph.as_default():
                    bytestr = req['input']

                    if debug:
                        util.step('data', np.fromstring(bytestr))

                    model = None
                    if req['name'] == 'spatial':
                        X = np.fromstring(bytestr, np.uint8).reshape(12, 16, 3)
                        if debug:
                            util.step('spatial, gets input', X.shape)

                        model = ml.load_spatial() if model is None else model
                        output = model.predict(np.array([X]))
                        if debug:
                            util.step('spatial, forward', output.shape)

                        Thread(target=self.send, args=(output, 'fc')).start()

                        return result_q.get()

                    elif req['name'] == 'temporal':
                        X = np.fromstring(bytestr, np.float32).reshape(12, 16, 6)
                        if debug:
                            util.step('temporal, gets input', X.shape)

                        model = ml.load_temporal() if model is None else model
                        output = model.predict(np.array([X]))
                        if debug:
                            util.step('temporal, forward', output.shape)

                        Thread(target=self.send, args=(output, 'fc')).start()

                        return result_q.get()

                    elif req['name'] == 'maxpool':
                        X = np.fromstring(bytestr, np.uint8)
                        X = X.reshape(1, X.size)

                        max_input = X if max_input is None else np.concatenate((max_input, X), axis=0)

                        if debug:
                            util.step('maxpool, gets input', X.shape)

                        if max_input.shape[0] < max_dim:
                            return ' '

                        model = ml.load_maxpool(N=max_dim) if model is None else model
                        output = model.predict(np.array([max_input]))
                        max_input = None
                        if debug:
                            util.step('maxpool, forward', output.shape)

                        Thread(target=self.send, args=(output, 'fc')).start()

                        return result_q.get()

                    elif req['name'] == 'fc':
                        X = np.fromstring(bytestr, np.float32)
                        X = X.reshape(X.size)

                        fc_input = X if fc_input is None else np.concatenate((fc_input, X))

                        if debug:
                            util.step('fc, gets input', X.shape)

                        print fc_input.size

                        if fc_input.size < fc_dim:
                            return ' '

                        model = ml.load_fc(fc_dim) if model is None else model
                        output = model.predict(np.array([fc_input]))
                        fc_input = np.array([])
                        if debug:
                            util.step('fc, forward', output.shape)

                        print output

                        return output.tobytes()

            except Exception, e:
                if debug:
                    print e
        else:
            if debug:
                print '+++++++++++++++++ message ++++++++++++++++++'
                print msg
                print '+++++++++++++++++ request ++++++++++++++++++'
                print req
                raise schema.AvroException('unexpected message:', msg.getname())

    def send(self, X, name):
        global ip, debug, result_q

        queue = None
        if name == 'maxpool':
            queue = ip['maxpool']
        else:
            queue = ip['fc']

        address = queue.get()
        client = ipc.HTTPTransceiver(address, 12345)
        requestor = ipc.Requestor(PROTOCOL, client)

        data = dict()
        data['input'] = X.tostring()
        data['name'] = name

        if debug:
            util.step('finish assembly request', name)

        output = requestor.request('forward', data)
        result_q.put(output)

        if debug:
            util.step('get output back', name)

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
    global ip, graph, fc_dim, max_dim, debug

    fc_dim = cmd.fc_dim
    max_dim = cmd.max_dim
    debug = cmd.debug

    with open('resource/ip') as file:
        address = yaml.safe_load(file)
        ip['fc'] = Queue()
        ip['maxpool'] = Queue()
        for addr in address['fc']:
            ip['fc'].put(addr)
        for addr in address['maxpool']:
            ip['maxpool'].put(addr)

    graph = tf.get_default_graph()

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
