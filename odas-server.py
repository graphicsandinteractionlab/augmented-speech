#!/usr/bin/env python3

import subprocess
import os
import json

from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import osc_bundle_builder
from pythonosc import osc_message_builder

from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient

odas_dir = '/home/hartmut/Development/SDKs/odas'
odaslive_path = odas_dir + '/bin/odaslive'
odaslive_config = odas_dir + '/config/odaslive/pseye.cfg'

odaslive_cmd = [odaslive_path, '-c', odaslive_config]

dispatcher = Dispatcher()
client = SimpleUDPClient("212.201.64.123", 8050)


# processes the frames
def process_frame(buffer):
    # get dict of json buffer
    buffer_dict = json.loads(buffer)
    # parse src
    for v in buffer_dict['src']:
        message_load = []
        message_load.append(buffer_dict['timeStamp'])
        message_load.append(v['id'])
        message_load.append(v['x'])
        message_load.append(v['y'])
        message_load.append(v['z'])
        message_load.append(v['activity'])
        message_load.append(v['tag'])
        
        client.send_message('/source', message_load)
        

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
