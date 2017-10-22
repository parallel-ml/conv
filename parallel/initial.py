from multiprocessing import Queue
from threading import Thread

import avro.ipc as ipc
import avro.protocol as protocol
import cv2
import numpy as np
import yaml

PROTOCOL = protocol.parse(open('resource/image.avpr').read())

global spatial_q, temporal_q


def send_request(bytestr, mode='spatial'):
    global spatial_q, temporal_q

    queue = None
    if mode == 'spatial':
        queue = spatial_q
    else:
        queue = temporal_q

    addr = queue.get()
    client = ipc.HTTPTransceiver(addr, 12345)
    requestor = ipc.Requestor(PROTOCOL, client)

    data = dict()
    data['input'] = bytestr
    data['name'] = mode

    output = requestor.request('forward', data)
    if output is not None and len(output) > 1:
        output = np.fromstring(output, dtype=np.float32)
        output = output.reshape(1, 51)
        print output

    client.close()
    queue.put(addr)


def master():
    frame_count = 4
    frame_width = 16
    frame_height = 12

    image = None

    frame0 = None
    flows = np.zeros((frame_height, frame_width, (frame_count - 1) * 2), dtype='float32')

    index = 0
    while index < 4:
        ret, frame = 'unknown', np.random.rand(12, 16, 3) * 255
        frame = frame.astype(dtype=np.uint8)

        image = frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame0 is not None:
            flows[..., (index - 1) * 2:(index - 1) * 2 + 2] = cv2.calcOpticalFlowFarneback(frame0, frame, None, 0.5, 3,
                                                                                           4, 3, 5, 1.1, 0)

        frame0 = frame
        index += 1

    Thread(target=send_request, args=(image.tobytes(), 'spatial')).start()
    Thread(target=send_request, args=(flows.tobytes(), 'temporal')).start()


def main():
    global spatial_q, temporal_q
    with open('resource/ip') as file:
        address = yaml.safe_load(file)
        spatial_q, temporal_q = Queue(), Queue()
        for addr in address['spatial']:
            spatial_q.put(addr)
        for addr in address['temporal']:
            temporal_q.put(addr)
    master()


if __name__ == '__main__':
    main()
