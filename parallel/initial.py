from collections import deque
from multiprocessing import Queue
from threading import Thread
import time

import avro.ipc as ipc
import avro.protocol as protocol
import cv2
import numpy as np
import yaml

PROTOCOL = protocol.parse(open('resource/image.avpr').read())


class Initializer:
    instance = None

    def __init__(self):
        self.spatial_q = Queue()
        self.temporal_q = Queue()
        self.flows = deque()

    @classmethod
    def create_init(cls):
        if cls.instance is None:
            cls.instance = Initializer()
        return cls.instance


def send_request(bytestr, mode='spatial'):
    init = Initializer.create_init()
    queue = init.spatial_q if mode == 'spatial' else init.temporal_q

    addr = queue.get()
    client = ipc.HTTPTransceiver(addr, 12345)
    requestor = ipc.Requestor(PROTOCOL, client)

    data = dict()
    data['input'] = bytestr
    data['name'] = mode

    requestor.request('forward', data)

    client.close()
    queue.put(addr)


def master():
    init = Initializer.create_init()
    frame0 = None
    while True:
        ret, frame = 'unknown', np.random.rand(12, 16, 3) * 255
        frame = frame.astype(dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame0 is not None:
            init.flow.appendleft(cv2.calcOpticalFlowFarneback(frame0, frame, None, 0.5, 3, 4, 3, 5, 1.1, 0))
            if init.flow.count() == 6:
                Thread(target=send_request, args=(frame.tobytes(), 'spatial')).start()
                optical_flow = np.concatenate(init.flow, axis=2)
                Thread(target=send_request, args=(optical_flow.tobytes(), 'temporal')).start()
                init.flow.pop()

        frame0 = frame
        time.sleep(1)


def main():
    init = Initializer.create_init()
    with open('resource/ip') as file:
        address = yaml.safe_load(file)
        for addr in address['spatial']:
            if addr == '#':
                break
            init.spatial_q.put(addr)
        for addr in address['temporal']:
            if addr == '#':
                break
            init.temporal_q.put(addr)
    master()


if __name__ == '__main__':
    main()
