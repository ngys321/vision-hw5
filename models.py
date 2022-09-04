import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import pdb
import time
import torchvision.models as torchmodels

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        if not os.path.exists('logs'):
            os.makedirs('logs')
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S_log.txt')
        self.logFile = open('logs/' + st, 'w')

    def log(self, str):
        print(str)
        self.logFile.write(str + '\n')

    def criterion(self):
        return nn.CrossEntropyLoss()

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, weight_decay = 0.1)

    def adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.lr  # TODO: Implement decreasing learning rate's rules
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
       


class LazyNet(BaseModel):
    def __init__(self):
        super(LazyNet, self).__init__()
        # TODO: Define model here
        self.fc1 = nn.Linear(3*32*32, 10)

    def forward(self, x):
        # TODO: Implement forward pass for LazyNet
        x = torch.flatten(x, 1) ##################################### IMPORTANT!
        x = self.fc1(x)
        return x
        

class BoringNet(BaseModel):
    def __init__(self):
        super(BoringNet, self).__init__()
        # TODO: Define model here
        self.fc1 = nn.Linear(3*32*32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        # TODO: Implement forward pass for BoringNet
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class CoolNet(BaseModel):
    def __init__(self):
        super(CoolNet, self).__init__()
        # TODO: Define model here
        self.conv1 = nn.Conv2d(3, 6, 3, padding = 1) # Conv2d(in_img_channels, out_img_channels, kernel_size)
        self.conv2 = nn.Conv2d(6, 6, 3, padding = 1)
        self.fc1 = nn.Linear(6 * 8 * 8, 10)

    def forward(self, x):
        # TODO: Implement forward pass for CoolNet
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) # max_pool2d(input, kernel_size)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1) # flatten(input, start_dim_index), flatten all dimensions except the batch dimension
                                # 즉 flatten(input, 1) 이면, input 을 이루는 tensor에서 dim 이 여러축이 존재하는데, 이 여러 dim 축을 나타내는 index 는 0부터 시작함.
                                # 근데 start_dim_index = 1 이면 어떤 의미냐면, flatten 를 적용할때, dim 축의 index 가 0인 첫번째 축(보통 batch 를 나타내는 dim)을 제외하고,
                                # 나머지 dim 들 모두를 flatten 한다는 것임.
                                # 참고 : 세번째 인자도 있음. end_dim 임. 이 인자는 아무값도 안주면, default 값으로 -1 값을 가짐. 즉 맨 마지막 dim 축까지 flatten 하게 됨.
        x = self.fc1(x)
        return x

# for test
# net = LazyNet()
# print(net)

# params = list(net.parameters())
# print(params)
# print(len(params))
# print(params[1].size())
