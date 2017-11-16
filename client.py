import time
import json
import avro.schema
import avro.io
import avro.ipc as ipc
import avro.protocol as protocol
import httplib
import sys

PROTOCOL = protocol.parse(open('image.avpr').read())

def ping(size):
    server_addr = ('198.162.1.2', 8000)
    client = ipc.HTTPTransceiver(server_addr[0], server_addr[1])
    requestor = ipc.Requestor(PROTOCOL, client)

    # fill in the Message record and send it
    message = dict()
    message['image'] ='1' * (size*1000)

    data = requestor.Request('process', message)
    client.Close()


def main():
    print "Start testing"
    for i in range(1,100):
        start = time.time()
        ping(i)
        print i, " : ", time.time() - start

main()
