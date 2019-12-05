#!/usr/bin/env python3

import subprocess
import os
import argparse
import json

from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import osc_bundle_builder
from pythonosc import osc_message_builder

from pythonosc.udp_client import SimpleUDPClient

from deepspeech import Model

odas_dir = os.getenv('HOME') + '/Code/SDKs/odas'
odaslive_path = odas_dir + '/bin/odaslive'
odaslive_config = odas_dir + '/config/odaslive/pseye.cfg'

odaslive_cmd = [odaslive_path, '-c', odaslive_config]

ds_features = { 'n_features' : 26, 'n_context' : 9, 'beam_width' : 500, 'lm_alpha' : 0.75, 'lm_beta' : 1.85 }
ds_model_path = os.getcwd() + ' models/deepspeech-0.6.0-models/output_graph.pbmm'
ds_lm_path = os.getcwd() + ' models/deepspeech-0.6.0-models/lm.binary'
ds_trie_path = os.getcwd() + ' models/deepspeech-0.6.0-models/trie'
ds_alphabet = ''

# setting up deepspeech
def setup_deepspeech():
    ds_model = Model(ds_model_path, ds_features['beam_width'])
    ds_model.enableDecoderWithLM(ds_lm_path,ds_trie_path)
    pass


# processes the frames
def process_frame(buffer,client):
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
        

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1", help="The ip to listen on")
    parser.add_argument("--port", type=int, default=8080, help="The port to listen on")
    args = parser.parse_args()

    client = SimpleUDPClient(args.ip, args.port)

    print('Ready ... ')

    buffer = ""
    # we pipe everything to the wrapper
    p = subprocess.Popen(odaslive_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(p.stdout.readline, b''):
        s = str(line,'utf-8')
        # a frame can be identified by a closing curly bracket
        if s.find('}\n') == 0:
            buffer += s
            process_frame(buffer,client)
            buffer = ""
        else:
            buffer += s
    p.stdout.close()
    p.wait()
    print('Stop!')

