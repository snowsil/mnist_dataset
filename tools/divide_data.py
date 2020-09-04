import random
import linecache
import os
import numpy as np
list1 = random.sample(range(0,11563), 1500)
print(list1)
src_file = "all_video.txt"
dst_file = "test_video_list.txt"
fp = open(dst_file , "w")
for i in list1:
	line = linecache.getline(src_file , i)
	fp.write(line)

fp.close()
