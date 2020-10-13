import os
src_file = "val_data.txt"
dst_file = "val_frame_list.txt"
fp =open(dst_file, "w")

for line in open(src_file):
    line = line.strip()
    #print(line)
    lista = [line.split("/")[0],"/",line.split("/")[1]]
    path0 = "".join(lista)
    #print(path0)
    count=0
    number = line.split("/")[-1].split(".")[0]
    num =100*int(number[0])+10*int(number[1])+int(number[2])
    for file in os.listdir(path0):
        count = count+1
    lst = [line.split("/")[0],"/",line.split("/")[1],"/",str(num),",",str(count),"\n"]
    str0 = "".join(lst)
    fp.write(str0)
    
fp.close()
