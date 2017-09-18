from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn

import avro.ipc as ipc
import avro.protocol as protocol
import avro.schema as schema
import keras
import matplotlib

matplotlib.use('Agg')
import numpy as np
import cv2
from skimage import img_as_float

NN = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None,
                                          input_shape=(224, 224, 3), pooling=None, classes=1000)

PROTOCOL = protocol.parse(open('image.avpr').read())


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
        print req
        if msg.name == 'procimage':
            bytestr = req['image']
            return self.process(bytestr)
        else:
            raise schema.AvroException('unexpected message:', msg.getname())

    def process(self, bytestr):
        nparr = np.fromstring(bytestr, np.uint8)
        image = img_as_float(cv2.imdecode(nparr, cv2.IMREAD_COLOR))
        resized_image = cv2.resize(image, (224, 224))
        test_x = np.array([resized_image])
        test_y = NN.predict(test_x)
        return test_y


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
