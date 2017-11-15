import time
import multiprocessing as mp
import picamera
from picamera.array import PiRGBArray
import numpy as np
import scipy.misc
import cv2
import json
import avro.schema
import avro.io
import avro.ipc as ipc
import avro.protocol as protocol
import http.client
import sys

PROTOCOL = protocol.Parse(open('image.avpr').read())

def ping(size):
    server_addr = ('128.61.79.61', 8000)
    client = ipc.HTTPTransceiver(server_addr[0], server_addr[1])
    requestor = ipc.Requestor(PROTOCOL, client)

    # fill in the Message record and send it
    message = dict()
    message['image'] = bytes(size*1000)

    data = requestor.Request('process', message)
    client.Close()


def main():
    for i in range(0,100):
        start = time.time()
        ping(i)
        print i, " : ", time.time() - start

main()
