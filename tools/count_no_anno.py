import os
import torch
import numpy as np
import random
import linecache

src_file = "train_frame_list.txt"

#fp =open( src_file, "r")
count =0
list = random.sample(range(1,1879502),1000)
for i in list:
    line = linecache.getline(src_file,i).strip()
    lst = ["anno/train/",line.split("/")[1],"_",line.split("/")[-1].split(",")[0],".txt"]
    dst_path = "".join(lst)
    if os.path.isfile(dst_path):
        count = count+ 1
print("count =",count)
print("frames without annotation:",count/10,"%")
#print(list)

#fp.close()