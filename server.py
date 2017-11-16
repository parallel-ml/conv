from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn

import avro.ipc as ipc
import avro.protocol as protocol
import avro.schema as schema
import matplotlib

matplotlib.use('Agg')
import time

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
            return 'GET'
        else:
            raise schema.AvroException('unexpected message:', msg.getname())


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


server_addr = ('0.0.0.0', 12345)

if __name__ == '__main__':
    server = ThreadedHTTPServer(server_addr, ImageHandler)
    server.allow_reuse_address = True
    server.serve_forever()
