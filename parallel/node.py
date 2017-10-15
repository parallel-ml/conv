import argparse
import os
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn

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
address, model, graph, dim, debug = ('0.0.0.0', 12345), 'spatial', None, 7680, False


class Responder(ipc.Responder):
    def __init__(self):
        ipc.Responder.__init__(self, PROTOCOL)

    def invoke(self, msg, req):
        if debug:
            util.step('gets message', req['name'])

        if msg.name == 'forward':
            try:
                global graph, model, dim
                with graph.as_default():
                    bytestr = req['input']

                    if debug:
                        util.step('data', np.fromstring(bytestr))

                    model = None
                    if req['name'] == 'spatial':
                        X = np.fromstring(bytestr, np.uint8).reshape(12, 16, 3)
                        if debug:
                            util.step('spatial, gets input', X.shape)

                        model = ml.load_spatial()
                        output = model.predict(np.array([X]))
                        if debug:
                            util.step('spatial, forward', output.shape)

                        return self.send(output, 'fc')

                    elif req['name'] == 'temporal':
                        X = np.fromstring(bytestr, np.uint8).reshape(12, 16, 20)
                        if debug:
                            util.step('temporal, gets input', X.shape)

                        model = ml.load_spatial()
                        output = model.predict(np.array([X]))
                        if debug:
                            util.step('temporal, forward', output.shape)
                        return self.send(output, 'fc')

                    elif req['name'] == 'fc':
                        X = np.fromstring(bytestr, np.uint8).reshape(dim)
                        if debug:
                            util.step('fc, gets input', X.shape)

                        model = ml.load_fc(dim)
                        output = model.predict(np.array([X]))
                        if debug:
                            util.step('fc, forward', output.shape)
                        return output.tostring()

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
        global address, debug
        client = ipc.HTTPTransceiver(address[0], address[1])
        requestor = ipc.Requestor(PROTOCOL, client)

        data = dict()
        data['input'] = X.tostring()
        data['name'] = name

        if debug:
            util.step('finish assembly request', name)

        output = requestor.request('forward', data)

        if debug:
            util.step('get output back', name)

        client.close()
        return output


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
    global address, graph, dim, debug

    address = (cmd.address, cmd.port)
    dim = cmd.dim
    debug = cmd.debug

    graph = tf.get_default_graph()

    server = ThreadedHTTPServer(('0.0.0.0', 12345), Handler)
    server.allow_reuse_address = True
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--address', action='store', default='0.0.0.0', metavar='\b',
                        help='Set request ip address binds')
    parser.add_argument('-p', '--port', action='store', default=12345, type=int, metavar='\b',
                        help='Set request server port number')
    parser.add_argument('--dim', metavar='\b', action='store', default=7680, type=int,
                        help='Choose fc layer input dimension')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='Set to debug mode')
    cmd = parser.parse_args()
    main(cmd)
