import os
src_path = "train_data_list.txt"
fp =open(src_path,"r")
lines = fp.readlines()
for line in lines:
    line = line.strip()
    length = int(line.split(",")[-1])
    path0 = line.split(",")[0]
    list0 = ["annotation/",path0.split("_")[0],"/",path0.split("/")[-1],".txt"]
    read_path = "".join(list0)
    fr = open(read_path, "r")
    annotation = fr.readlines()
    t = 0
    for anno in annotation:
        anno = anno.strip()
        i = anno.split(" ")[0]
        x = anno.split(" ")[1]
        y = anno.split(" ")[2]
        w = anno.split(" ")[3]
        h = anno.split(" ")[4]
        score = anno.split(" ")[5]
        list1 = ["anno/",path0.split("_")[0],"/",path0.split("/")[-1],"_",str(i),".txt"]
        dst_path = "".join(list1)
        fw = open(dst_path,"w")
        if t == i:
            listx = [x," ",y," ",w," ",h," ",score,"\n"]
            str0 = "".join(listx)
            fw.write(str0)
        else:
            t = i
            fw.close()
            list1 = ["anno/",path0.split("_")[0],"/",path0.split("/")[-1],"_",str(i),".txt"]
            dst_path = "".join(list1)
            fw = open(dst_path,"w")
            listx = [x," ",y," ",w," ",h," ",score,"\n"]
            str0 = "".join(listx)
            fw.write(str0)
        fw.close()

        
    
    fr.close()


fp.close()
"""
从列表里读注释，把一个视频一个文件变成一个帧一个图
"""