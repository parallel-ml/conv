import avro.ipc as ipc
import avro.protocol as protocol
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import threading
import time

PROTOCOL = protocol.parse(open('resource/image.avpr').read())

server_addr = ('192.168.1.2', 12345)

image = cv2.imread('data/tiger.jpg')
resized_image = cv2.resize(image, (224, 224))


def video():
    camera = PiCamera()
    rawCapture = PiRGBArray(camera)

    width = 224
    height = 224
    camera.resolution = (width, height)
    camera.framerate = 24

    while True:
        camera.capture(rawCapture, format='rgb')
        # opencv won't display the rgb format correctly but it's just for transimtting the image
        im = rawCapture.array
        resized_image = cv2.resize(im, (224, 224))
        bytestr = cv2.imencode('.jpg', resized_image)[1].tostring()
        (threading.Thread(target=send_request, args=(bytestr,))).start()

        rawCapture.truncate(0)  # this is important


def send_request(bytestr, time=None):
    client = ipc.HTTPTransceiver(server_addr[0], server_addr[1])
    requestor = ipc.Requestor(PROTOCOL, client)

    data = dict()
    data['image'] = bytestr
    data['time'] = time

    print 'image label', requestor.request('procimage', data)

    client.close()


def main():
    bytestr = cv2.imencode('.jpg', resized_image)[1].tostring()
    start = time.time()
    for _ in range(10):
        send_request(bytestr, start)
    print 'client stuff?'


if __name__ == '__main__':
    main()
