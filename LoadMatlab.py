import scipy.io as sio
import sys
from os.path import basename, dirname

if __name__=="__main__":

    object_path = sys.argv[1]
    object_name = basename(dirname(object_path))

    point_cloud = sio.loadmat(object_path)['voxel']
    
    save_path = sys.argv[2]+ object_name +".npy"
    
    np.save(save_path, point_cloud)