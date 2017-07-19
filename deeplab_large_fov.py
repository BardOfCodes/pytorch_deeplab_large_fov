import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(3,64,3,padding = 1)
        self.conv1_2 = nn.Conv2d(64,64,3,padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.conv2_1 = nn.Conv2d(64,128,3,padding = 1)
        self.conv2_2 = nn.Conv2d(128,128,3,padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.conv3_1 = nn.Conv2d(128,256,3,padding = 1)
        self.conv3_2 = nn.Conv2d(256,256,3,padding = 1)
        self.conv3_3 = nn.Conv2d(256,256,3,padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.conv4_1 = nn.Conv2d(256,512,3,padding = 1)
        self.conv4_2 = nn.Conv2d(512,512,3,padding = 1)
        self.conv4_3 = nn.Conv2d(512,512,3,padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size = 3, stride = 1,padding=1)
        self.conv5_1 = nn.Conv2d(512,512,3,padding = 2,dilation = 2)
        self.conv5_2 = nn.Conv2d(512,512,3,padding = 2,dilation = 2)
        self.conv5_3 = nn.Conv2d(512,512,3,padding = 2,dilation = 2)
        self.pool5 = nn.MaxPool2d(kernel_size = 3, stride = 1,padding=1)
        self.pool5a = nn.AvgPool2d(kernel_size = 3, stride = 1,padding=1)
        self.fc6 = nn.Conv2d(512,1024,3,padding = 12,dilation = 12)
        self.drop6 = nn.Dropout2d(p=0.5)
        self.fc7 = nn.Conv2d(1024,1024,1)
        self.drop7 = nn.Dropout2d(p=0.5)
        self.fc8_voc12 = nn.Conv2d(1024,21,1)
        self.log_softmax = nn.LogSoftmax()
        self.fc8_interp_test = nn.UpsamplingBilinear2d(size=(513,513))
        
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool1(F.relu(self.conv1_2(x)))
        x = F.relu(self.conv2_1(x))
        x = self.pool2(F.relu(self.conv2_2(x)))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(F.relu(self.conv3_3(x)))
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.pool4(F.relu(self.conv4_3(x)))
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.pool5a(self.pool5(F.relu(self.conv5_3(x))))
        x = self.drop6(F.relu(self.fc6(x)))
        x = F.relu(self.fc7(x))
        x= self.fc8_voc12(self.drop7(x))
        x = self.log_softmax(x)
        
        return x
    
    def forward_test(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool1(F.relu(self.conv1_2(x)))
        x = F.relu(self.conv2_1(x))
        x = self.pool2(F.relu(self.conv2_2(x)))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(F.relu(self.conv3_3(x)))
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.pool4(F.relu(self.conv4_3(x)))
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.pool5a(self.pool5(F.relu(self.conv5_3(x))))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8_voc12(x)
        x = self.log_softmax(x)
        x = self.fc8_interp_test(x)
        
        return x
