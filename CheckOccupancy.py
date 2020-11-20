import bpy
import sys
import mathutils
import numpy as np

def is_inside(p, obj, max_dist = 1.84467e+19):
    found, point, normal, face = obj.closest_point_on_mesh(p, max_dist)
    p2 = point-p
    v = p2.dot(normal)
    # print(found, v)
    return not(v<0.0)


def main():
    
    print("Hello world")
    bpy.ops.import_scene.obj(filepath="/home/sk002/Downloads/pix3d/model/bed/IKEA_BEDDINGE/model.obj")

    
    a = False
    v1=None
    while True:
        v1=np.random.uniform(low=-1, high=1, size=(3,))

        # point = mathutils.Vector((v1[0], -v1[1] , v1[2]))
        point = mathutils.Vector((0,0,0))
        obj = bpy.context.selected_objects[0]
        
        a= is_inside(point, obj)
        if a==False:
            break
    
    print(a, v1)
    return a

if __name__=="__main__":
    main()