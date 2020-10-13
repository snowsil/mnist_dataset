import cv2
import os
import numpy as np
import linecache
import pathlib
"""
root_p = 'Dataset/'
root_s = '/media/HD1/yuyue/dataset/'
set_file = 'img_list.txt'
ids = []
#print(root_p)
with open(set_file) as f:
    for line in f:
        #print(line)
        line = line.strip()
        list1 = [root_p,line]
        list2=[root_s,line.split("/")[-1].split(".")[0],"/"]
        video_path = "".join(list1)
        save_path = "".join(list2)
        isExists=os.path.exists(save_path)
        if not isExists:
            os.makedirs(save_path) 
        print(video_path)
        #print(save_path)
        ########################
        video = cv2.VideoCapture(video_path)
        success,frame = video.read()
        #print(success)
        i = 0
        while(success):
            list0 = [save_path,str(i),".jpg"]
            out_path = "".join(list0)
            #print(out_path)
            cv2.imwrite(out_path, frame) 
            success, frame = video.read()
            i = i + 1
"""

####################################################################

#现在ids[]里面是所有的视频

frame_path = '/media/HD1/yuyue/dataset/val_data/v_BasketballDunk_g03_c04'
annotation_file = '/media/HD1/yuyue/dataset/annotation/val/v_BasketballDunk_g03_c04.txt'
save_file = 'result01.avi'
#video = cv2.VideoCapture(video_path)
#print("Loaded video ...")
"""
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)"""
fps = 30
fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
fw = open(annotation_file, 'r')
outVideo = cv2.VideoWriter(save_file, fourcc, fps, (320, 240))
"""success,frame = video.read()
print(success)"""
line = fw.readline()
print(line)
if line:
    i, xmin, ymin, w, h, score = line.split(' ')
    i=int(i)
    xmin=int(xmin)
    ymin=int(ymin)
    w = int(w)
    h = int(h)
    score = float(score)
else:
    i = -1
"""
print(i)
print(xmin) 
print(ymin)
print(score)
"""
"""
while(success):"""
for j in range(0,81):
    str_a = "/%03d"+".jpg"
    frame = cv2.imread(frame_path + str_a % j)
    print(frame_path + str_a % j)
    while(i==j):
        cv2.rectangle(frame, (xmin, ymin), (w+xmin, h+ymin), (0, 0, 255), 2)
        text = "{:.4f}".format(score)
        cx = xmin
        cy = ymin + 12
        cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        line = fw.readline()
        if not line:
            break
        i, xmin, ymin, w, h, score = line.split(' ')
        i=int(i)
        xmin=int(xmin)
        ymin=int(ymin)
        w = int(w)
        h = int(h)
        score = float(score)
    j= j+1
    outVideo.write(frame)
    #print("save frame {:d}".format(j))

fw.close()

