import time
import array
import io
import json
import struct
import avro.datafile
import avro.schema
import avro.ipc as ipc
import avro.protocol as protocol
# import http.server
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from SocketServer import ThreadingMixIn



PROTOCOL = protocol.parse(open('image.avpr').read())

class ImageResponder(ipc.Responder):
    def __init__(self):
        ipc.Responder.__init__(self, PROTOCOL)

    def Invoke(self, msg, req):
        if msg.name == 'process':
            image = req['image']

            response = []
            response.append("1")
            return response

        else:
            print("Schema does not match")


class ImageHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.responder = ImageResponder()
        call_request_reader = ipc.FramedReader(self.rfile)
        call_request = call_request_reader.Read()
        resp_body = self.responder.Respond(call_request)
        self.send_response(200)
        self.send_header('Content=Type', 'avro/binary')
        self.end_headers()
        resp_writer = ipc.FramedWriter(self.wfile)
        resp_writer.Write(resp_body)

server_addr = ('0.0.0.0', 8000)

if __name__ == '__main__':
    server = HTTPServer(server_addr, ImageHandler)
    server.allow_reuse_address = True
    server.serve_forever()
