import os
src_file = "train_video_list.txt"
dst_file = "train_data_list.txt"
fp =open(dst_file, "w")
for line in open(src_file):
    path0 = line.strip()
    count = 0
    for file in os.listdir(path0):
        count = count+1
    list1 = [path0,",",str(count),"\n"]
    str0 = "".join(list1)
    fp.write(str0)
fp.close()
