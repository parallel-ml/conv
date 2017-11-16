import avro.ipc as ipc
import avro.protocol as protocol
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import threading
import time

PROTOCOL = protocol.parse(open('image.avpr').read())

server_addr = ('192.168.1.2', 12345)


def send_request(size):
    client = ipc.HTTPTransceiver(server_addr[0], server_addr[1])
    requestor = ipc.Requestor(PROTOCOL, client)

    data = dict()
    data['image'] = '1' * (1000 * size)

    requestor.request('procimage', data)

    client.close()


def main():
    for i in range(1,10):
        start = time.time()
        send_request(i)
        print i, ",", time.time() - start

if __name__ == '__main__':
    main()
