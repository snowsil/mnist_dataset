import shutil
import os
root_path = "/media/HD1/yuyue/"
fp = open("test_video_list.txt","r")
for line in fp:
	list0 = [root_path,line.strip()]
	src_path = "".join(list0)
	shutil.move(src_path,"/media/HD1/yuyue/dataset/test_data")
fp.close()
