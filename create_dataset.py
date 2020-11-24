import sys
import glob
import os
import shutil

if __name__=="__main__":
    source_directory = sys.argv[1]
    if source_directory[-1]!="/":
        source_directory+="/"

    destination_directory = sys.argv[2]
    if destination_directory[-1]!="/":
        destination_directory+="/"

    counter = 0
    for folder in glob.glob(source_directory+"*/"):
        if not os.path.isdir(folder):
             continue
        for subdir in glob.glob(folder+"*/"):
            if not os.path.isdir(subdir):
                continue
            print(subdir)
            savepath = destination_directory+str(counter)+"/"
            os.mkdir(savepath)
            # os.mkdir(savepath+"images")
            shutil.copytree(subdir+"img_choy2016", savepath+"images/")
            shutil.copy(subdir+"model.binvox", savepath)
            counter+=1