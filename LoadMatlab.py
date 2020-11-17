import scipy.io as sio
import sys


if __name__=="__main__":
    object_path = sys.argv[0]

    point_cloud = sio.loadmat(object_path, struct_as_record=True)

    print(point_cloud)