import avro.ipc as ipc
import avro.protocol as protocol
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import threading

PROTOCOL = protocol.parse(open('image.avpr').read())

server_addr = ('128.61.18.28', 12345)


def main():
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


def send_request(bytestr):
    client = ipc.HTTPTransceiver(server_addr[0], server_addr[1])
    requestor = ipc.Requestor(PROTOCOL, client)

    data = dict()
    data['image'] = bytestr

    print 'image label', requestor.request('procimage', data)

    client.close()


if __name__ == '__main__':
    main()
