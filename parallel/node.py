from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn

import avro.ipc as ipc
import avro.protocol as protocol
import avro.schema as schema
import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')

import argparse
from model import load_fc, load_temporal, load_spatial

PROTOCOL = protocol.parse(open('resource/image.avpr').read())


class ImageResponder(ipc.Responder):
    def __init__(self):
        ipc.Responder.__init__(self, PROTOCOL)

    def invoke(self, msg, req):
        """
        automatically called by do_POST method
        :param msg: message name
        :param req: request sent by client
        :return:
        """
        if msg.name == 'procimage':
            bytestr = req['image']
            nparr = np.fromstring(bytestr, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            global model, end
            if end:
                output = model.predict(np.array([img]))
                return output.tostring()
            else:
                return self.send(data=req)
        else:
            raise schema.AvroException('unexpected message:', msg.getname())

    def send(self, data):
        global request_address
        client = ipc.HTTPTransceiver(request_address[0], request_address[1])
        requestor = ipc.Requestor(PROTOCOL, client)

        output = requestor.request('procimage', data)
        output = np.fromstring(output, dtype=np.float32)
        output = output.reshape(1, 256)

        client.close()


class ImageHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        """
        this method will be called once the server gets the request
        :return:
        """
        self.responder = ImageResponder()
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
    if cmd.model != 'spatial' and cmd.model != 'temporal' and cmd.model != 'fc' and cmd.model != 'maxpool':
        print 'Warning!!'
        print 'Choose model type from: spatial, temporal, fc, maxpool'

    global end, address, port, model, request_address

    if cmd.model == 'spatial':
        model = load_spatial()
    elif cmd.model == 'temporal':
        model = load_temporal()
    elif cmd.model == 'fc':
        model = load_fc(cmd.dim)

    address = cmd.server_address
    port = cmd.server_port
    end = cmd.end

    request_address = (cmd.request_address, cmd.request_port)

    server = ThreadedHTTPServer((address, port), ImageHandler)
    server.allow_reuse_address = True
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', action='store', required=True, metavar='\b',
                        help='Choose a model type: spatial, temporal, fc, maxpool')
    parser.add_argument('--server_address', action='store', default='0.0.0.0', metavar='\b',
                        help='Set server ip address binds')
    parser.add_argument('--server_port', action='store', default=12345, type=int, metavar='\b',
                        help='Set server port number')
    parser.add_argument('--request_address', action='store', default='0.0.0.0', metavar='\b',
                        help='Set request ip address binds')
    parser.add_argument('--request_port', action='store', default=12345, type=int, metavar='\b',
                        help='Set request server port number')
    parser.add_argument('-e', action='store_true', default=False,
                        help='Set node as last element')
    parser.add_argument('-d', '--dim', metavar='\b', action='store', default=7680,
                        help='Choose fc layer input dimension')
    cmd = parser.parse_args()
    main(cmd)
