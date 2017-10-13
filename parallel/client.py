import avro.ipc as ipc
import avro.protocol as protocol
import cv2
import numpy as np

# from picamera import PiCamera
# from picamera.array import PiRGBArray

PROTOCOL = protocol.parse(open('resource/image.avpr').read())

server_addr = ('127.0.0.1', 12345)

image = cv2.imread('data/tiger.jpg')
resized_image = cv2.resize(image, (16, 12))


def send_request(bytestr, time=None):
    client = ipc.HTTPTransceiver(server_addr[0], server_addr[1])
    requestor = ipc.Requestor(PROTOCOL, client)

    data = dict()
    data['image'] = bytestr

    output = requestor.request('procimage', data)
    output = np.fromstring(output, dtype=np.float32)
    output = output.reshape(1, 256)

    client.close()


def main():
    bytestr = cv2.imencode('.jpg', resized_image)[1].tostring()
    send_request(bytestr)


if __name__ == '__main__':
    main()
