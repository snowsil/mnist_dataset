import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cv2
import numpy as np
#from skimage import io
#from skimage import transform
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
"""
参考内容：
网络：https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
主体代码：https://zhuanlan.zhihu.com/p/95701775

通过网络的各个层计算输入输出尺寸，
对卷积层relu层以及全连接层有了认识
transforms 作为一个预处理函数
多个类别计算损失值的方法，softmax
（1）三步实现：softmax+log+nll_loss
（2）两步实现：log_softmax+nll_loss
（3）一步实现：crossEntropyLoss
"""

BATCH_SIZE = 512 # 大概需要2G的显存,512
EPOCHS = 20 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

#######
"""
datasets.MNIST('data', train = False, transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1037,), (0.3081,))
]))"""
########
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1037,), (0.3081,))
])
class MY_MNIST(Dataset):
    training_file = 'training.pt'
    test_file = 'test.pt'
    def __init__(self, root,  transform=None):
        self.transform = transform
        self.data, self.targets = torch.load(root)  #读的是tensor
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = np.asarray(img)
        #cv2.imwrite("img.jpg",img)
        
        img = cv2.Canny(np.asarray(img), 80, 80)
        img = Image.fromarray(img, mode='L')  #tensor变成数组，数组到img
        
        if self.transform is not None:
            img = self.transform(img)  #tensor
        #img =transforms.ToTensor()(img)

        return img,target

    def __len__(self):
        return len(self.data)
#print(MY_MNIST)       
train = MY_MNIST(root='./data/MNIST/processed/training.pt', transform= transform)
test = MY_MNIST(root='./data/MNIST/processed/test.pt', transform= transform)
# 下载训练集
train_loader = torch.utils.data.DataLoader( train ,batch_size = BATCH_SIZE, shuffle = True)

# 测试集
test_loader = torch.utils.data.DataLoader( test ,batch_size = BATCH_SIZE, shuffle = True)
#加一个预处理
class ConvNet(nn.Module):
    """def __init__(self):
        super().__init__()
        #1*1*28*28
        self.conv1 = nn.Conv2d(1, 10, 5) 
        self.conv2 = nn.Conv2d(10, 20, 3) 
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        in_size = x.size(0)
        #print("0.x.shape=",x.shape)
        out= self.conv1(x) # 1* 10 * 24 *24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2) # 1* 10 * 12 * 12
        #print("1.x.shape=",out.shape)
        out = self.conv2(out) # 1* 20 * 10 * 10
        out = F.relu(out)
        out = out.view(in_size, -1) # 1 * 2000
        #print("2.x.shape=",out.shape)
        out = self.fc1(out) # 1 * 500
        out = F.relu(out)
        out = self.fc2(out) # 1 * 10
        out = F.log_softmax(out, dim = 1)
        return out
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #print("0.x.shape=",x.shape)
        #1*28*28
        #32*32
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  #6 * 15 * 15
        #print("1.x.shape=",x.shape)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) #16 * 6 * 6 
        #print("2.x.shape=",x.shape)
        x = x.view(-1, self.num_flat_features(x))
        #print("3.x.shape=",x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = F.log_softmax(x, dim = 1)
        """
        求多分类交叉熵损失有三种途径可以实现，分别是：
        （1）三步实现：softmax+log+nll_loss
        （2）两步实现：log_softmax+nll_loss
        （3）一步实现：crossEntropyLoss
        """
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())
print(model)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #print(data,target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target) 
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss =0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction = 'sum') # 将一批的损失相加
            test_loss += F.cross_entropy(output, target, reduction = 'sum') # 将一批的损失相加
            pred = output.max(1, keepdim = True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100.* correct / len(test_loader.dataset)
            ))
for epoch in range(1, EPOCHS + 1):
    train(model,  DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)