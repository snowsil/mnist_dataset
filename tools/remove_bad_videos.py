import os
import torch
import numpy as np
import random
import linecache
import shutil
"""

"""
src_file = "/media/HD1/yuyue/dataset/train_frame_list.txt"

i = 1
j = 1
line = linecache.getline(src_file,i).strip()
a = 0
b = 0 
#print(line)
while line:
    num = int(line.split(",")[-1])
    #print(num)
    count = 0
    b= b+1
    for j in range(1,num):
        #print(line)
        lst = ["anno/train/",line.split("/")[1],"_",str(j),".txt"]
        dst_path = "".join(lst)
        #print("dst_path =",dst_path)
        if os.path.isfile(dst_path):
            count = count+ 1
        i = i + 1
    
    if count/num <0.5:
        a =a+1
        #print("anno/test/",line.split("/")[1])
        #print("picture with anno =",count/num)
        lstt = ["train_data/",line.split("/")[1]]
        bad_video ="".join(lstt)
        if os.path.exists(bad_video):
            shutil.move(bad_video,"/media/HD1/yuyue/dataset/bad_videos")

    line = linecache.getline(src_file,i).strip()
#print("remainings:",a/b)

"""
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
"""
#fp.close()