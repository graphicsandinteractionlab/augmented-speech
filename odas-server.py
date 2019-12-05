#!/usr/bin/env python3

import subprocess
import os
import json

from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient

odas_dir = '/home/hartmut/Development/SDKs/odas'
odaslive_path = odas_dir + '/bin/odaslive'
odaslive_config = odas_dir + '/config/odaslive/pseye.cfg'

odaslive_cmd = [odaslive_path, '-c', odaslive_config]

dispatcher = Dispatcher()
# server = BlockingOSCUDPServer(("127.0.0.1", 8050), dispatcher)
client = SimpleUDPClient("212.201.64.123", 8050)


# processes the frames
def process_frame(buffer):
    buffer_dict = json.loads(buffer)
#    client.send_message("/oscControl/slider1", buffer_dict['src'][0]['x'])
#    client.send_message("/oscControl/slider2", buffer_dict['src'][0]['y'])
#    client.send_message("/oscControl/slider3", buffer_dict['src'][0]['z'])
#    client.send_message("/oscControl/slider4", buffer_dict['src'][0]['id'])
#    print(buffer_dict['src'][0]['x'],' ')
    print(buffer)
#    print(buffer_dict['src'][0]['id'],' ') 
#    print(buffer_dict['src'][0]['y'],' ')
#    print(buffer_dict['src'][0]['z'])
    print()
    pass

def run():
    print('Ready ... ')
    buffer = ""
    # we pipe everything to the wrapper
    p = subprocess.Popen(odaslive_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(p.stdout.readline, b''):
        s = str(line,'utf-8')
        # a frame can be identified by a closing curly bracket
        if s.find('}\n') == 0:
            buffer += s
            process_frame(buffer)
            buffer = ""
        else:
            buffer += s
    p.stdout.close()
    p.wait()
    print('Stop!')

run()
