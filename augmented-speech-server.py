#!/usr/bin/env python3

# local stuff
import subprocess
import os
import sys
import argparse
import json
import numpy as np
import yaml

# profiling stuff
from timeit import default_timer as timer

# python-osc
from pythonosc import osc_message_builder
from pythonosc.udp_client import SimpleUDPClient

from deepspeech import Model

odas_dir = os.getenv('HOME') + '/Code/SDKs/odas'
odaslive_path = odas_dir + '/bin/odaslive'
odaslive_config = os.getcwd() + '/config/odas_pseye.cfg'

odaslive_cmd = [odaslive_path, '-c', odaslive_config]

# input based on deepspeech examples 
ds_features = { 'n_features' : 26, 'n_context' : 9, 'beam_width' : 500, 'lm_alpha' : 0.75, 'lm_beta' : 1.85 }
ds_model_path = os.getcwd() + '/models/deepspeech-0.6.0-models/output_graph.pbmm'
ds_lm_path = os.getcwd() + '/models/deepspeech-0.6.0-models/lm.binary'
ds_trie_path = os.getcwd() + '/models/deepspeech-0.6.0-models/trie'

class ODAS2DS:
    def __init__(self):
        # 16bit with 16000Hz sampled
        self.channels = 4
        self.frame_size = 2 * self.channels
        pass

    def run(self):
        with open(os.getcwd() + '/test_separated.raw', 'rb') as self.file:
            frame = self.file.read(self.frame_size)
        pass

class SpeechBlock:
    def __init__(self,startTimeStamp,speechId,channel):
        self.start = startTimeStamp
        self.end = self.start
        self.id = speechId
        self.channel = channel
        pass

    def identifier(self):
        return self.id

    def setEndTimeStamp(self,timeStamp):
        self.end = timeStamp

    def __str__(self):
        return "id:{0} start:{1} end:{2}".format(self.id,self.start,self.end)

class AugmentedSpeech:
    """
    """
    def __init__(self,runVerbose = False):
        self.ds_model = None
        self.osc_client = None
        self.verbose = runVerbose
        self.currentBlocks = []
        pass

    # setting up OSC subsystem
    def init_osc(self,host,port):
        self.osc_client = SimpleUDPClient(host, port)
        pass


    # Setting up deepspeech
    def init_deepspeech(self):
        self.ds_model = Model(ds_model_path, ds_features['beam_width'])
        self.ds_model.enableDecoderWithLM(ds_lm_path,ds_trie_path,ds_features['lm_alpha'],ds_features['lm_beta'])
        pass

    
    def purge(self):
        for blk in self.currentBlocks:
            print(blk)


    def update_tracker(self,id,channel,timeStamp):
        for blk in self.currentBlocks:
            if blk.identifier() == id:
                # update block
                blk.setEndTimeStamp(timeStamp)
                return

        # not found and updated - add a new block
        self.currentBlocks.append(SpeechBlock(timeStamp,id,channel))

        # now send off stuff to deepspeech
        self.purge()

    # processes a frame of the ODAS tracker
    def __process_odas_frame(self,buffer):
        # get dict of json buffer
        buffer_dict = json.loads(buffer)

        buffer_src_list = buffer_dict['src']; 

        # parse src
        for i in range(len(buffer_src_list)):

            # this is the i'th channel
            v = buffer_src_list[i]

            timeStamp = buffer_dict['timeStamp']

            # filter out inactive sources
            if v['activity'] < 0.5:
                continue

            # update the tracker
            self.update_tracker(v['id'],i,timeStamp)

            # trap ahead - added channel
            pay_load = []
            pay_load.append(timeStamp)
            pay_load.append(i)       # this is the channel - needed to parse the right channel out of the raw file
            pay_load.append(v['id'])
            pay_load.append(v['x'])
            pay_load.append(v['y'])
            pay_load.append(v['z'])
            pay_load.append(v['activity'])
            pay_load.append(v['tag'])

            self.osc_client.send_message('/source', pay_load)


    def run(self):
        print('ready ... ')
        buffer = ""
        # we pipe everything to the wrapper
        p = subprocess.Popen(odaslive_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(p.stdout.readline, b''):
            s = str(line,'utf-8')
            # a frame can be identified by a closing curly bracket
            if s.find('}\n') == 0:
                buffer += s
                self.__process_odas_frame(buffer)
                buffer = ""
            else:
                buffer += s
        p.stdout.close()
        # pass back return code
        return p.wait()

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1", help="The ip to listen on")
    parser.add_argument("--port", type=int, default=8080, help="The port to listen on")
    parser.add_argument("--verbose", type=bool, default=False, help="Run in verbose mode")   
    args = parser.parse_args()
 
    augs = AugmentedSpeech(args.verbose)
    augs.init_deepspeech()
    augs.init_osc(args.ip, args.port)

    # o2d = ODAS2DS()
    # o2d.run()

    sys.exit(augs.run())



if __name__ == "__main__":
    main()