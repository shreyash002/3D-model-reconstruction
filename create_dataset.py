import sys
import glob
import os
import shutil

if __name__=="__main__":
    limit = 400
    source_directory = sys.argv[1]
    if source_directory[-1]!="/":
        source_directory+="/"

    destination_directory = sys.argv[2]
    if destination_directory[-1]!="/":
        destination_directory+="/"

    class_list = ["02691156", "02958343", "03001627", "04090263", "03636649", "04401088", "04530566", "02828884"]
    counter = 0
    
    for folder in glob.glob(source_directory+"*/"):
        if os.path.dirname(folder)[-8:] not in class_list:
            continue
        
        num_folder = 0
        if not os.path.isdir(folder):
             continue
        for subdir in glob.glob(folder+"*/"):
            if not os.path.isdir(subdir):
                continue
            savepath = destination_directory+str(counter)+"/"
            # os.mkdir(savepath)
            # os.mkdir(savepath+"images")
            shutil.copytree(subdir+"img_choy2016", savepath+"images/")
            shutil.copy(subdir+"model.binvox", savepath)
            counter += 1
            num_folder += 1
            if num_folder == limit:
                break
