import torch

pthfile = r'/home/yuyue/mega.pytorch/inference/predictions.pth'
net = torch.load(pthfile, map_location='cpu')


with open('predictions.txt', 'a') as file0:
    for items in net:
        print(items.get_field("scores"), file=file0)
