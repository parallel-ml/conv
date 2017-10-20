import json
from multiprocessing import Pool, Queue

import avro.ipc as ipc
import avro.protocol as protocol
import cv2
import numpy as np

PROTOCOL = protocol.parse(open('resource/image.avpr').read())


def send_request(bytestr, addr_queue, mode):
    addr = addr_queue.get()
    client = ipc.HTTPTransceiver(addr[0], addr[1])
    requestor = ipc.Requestor(PROTOCOL, client)

    data = dict()
    data['input'] = bytestr
    data['name'] = 'fc'

    output = requestor.request('forward', data)
    if output is not None and len(output) > 1:
        output = np.fromstring(output, dtype=np.float32)
        output = output.reshape(1, 51)

    client.close()
    addr_queue.put(addr)


def master(ip):
    pool = Pool(5)

    frame_count = 4
    frame_width = 16
    frame_height = 12

    frame0 = None
    flows = np.zeros((frame_height, frame_width, (frame_count - 1) * 2), dtype='float32')

    index = 0
    while index < 4:
        ret, frame = 'unknown', np.random.rand(12, 16, 3) * 255
        frame = frame.astype(dtype=np.uint8)

        pool.apply_async(send_request, (frame.tobytes(), ip['spatial'], 'spatial'))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not frame0:
            flows[..., (index - 1) * 2:(index - 1) * 2 + 2] = cv2.calcOpticalFlowFarneback(frame0, frame, None, 0.5, 3,
                                                                                           4, 3, 5, 1.1, 0)

        frame0 = frame
        index += 1

    pool.apply_async(send_request, (frame.tobytes(), ip['temporal'], 'temporal'))


def main():
    ip = dict()
    with open('resource/ip') as file:
        address = json.load(file)
        spatial_q, temporal_q = Queue(), Queue()
        for ip in address['spatial']:
            spatial_q.put(ip)
        for ip in address['temporal']:
            temporal_q.put(ip)
        ip['spatial'] = spatial_q
        ip['temporal'] = temporal_q
    master(ip)


if __name__ == '__main__':
    main()
