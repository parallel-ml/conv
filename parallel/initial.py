import avro.ipc as ipc
import avro.protocol as protocol
import numpy as np

PROTOCOL = protocol.parse(open('resource/image.avpr').read())

server_addr = ('127.0.0.1', 12345)


def send_request(bytestr):
    client = ipc.HTTPTransceiver(server_addr[0], server_addr[1])
    requestor = ipc.Requestor(PROTOCOL, client)

    data = dict()
    data['input'] = bytestr
    data['name'] = 'fc'

    output = requestor.request('forward', data)
    if output is not None and len(output) > 1:
        output = np.fromstring(output, dtype=np.float32)
        output = output.reshape(1, 51)

    print output

    client.close()


def main():
    # data = np.random.rand(12, 16, 3).astype(dtype=np.uint8)
    # data *= 255
    data = np.random.rand(512).astype(dtype=np.float32)
    bytestr = data.tostring()
    send_request(bytestr)


if __name__ == '__main__':
    main()