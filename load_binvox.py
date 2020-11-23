import binvox_rw
import numpy as np
import sys


if __name__=="__main__":
    filename = sys.argv[1]
    with open(filename, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    print((model.data).shape)
