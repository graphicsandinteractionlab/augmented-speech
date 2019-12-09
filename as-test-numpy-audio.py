#!/usr/bin/env python3

import numpy as np

class RawParser: 
    def __init__(self):
        pass

    def run(self):
        pass


def main():
    pass


if __name__ == "__main__":
    x = np.fromfile('data/test_sep_000.raw',dtype=np.int16)


    # file is 2138112
    print('size ',x.size) # size in samples (16bit)


    # hop size - Number of samples acquired on each channel at each frame
    hop_size = 128
    channels = 4
    bytes_per_sample = 2 

    n_frames = 2088 # 

    total_bytes = hop_size*channels*n_frames
    bytes_per_channel = int(total_bytes / channels)

    print('total ', total_bytes)
    print('per channel ',bytes_per_channel)


    # file has 2088 frames
    x = np.reshape(x,(bytes_per_channel,channels))

    c0 = x[:,0] # channel one

    print(c0)
    
    print(x)
    main()