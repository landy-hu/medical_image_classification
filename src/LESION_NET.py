from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import numpy as np
class Lesion_net(nn.Module):
    def __init__(self):
        super(Lesion_net, self).__init__()
        self.activate = nn.LeakyReLU(0.01)
        self.activate2= nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, dilation=2, padding=4,bias=False)
        self.norm1 = nn.BatchNorm2d(num_features = 32)
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding=1,bias=False)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, dilation=2, padding=4,bias=False)
        self.norm2 = nn.BatchNorm2d(num_features=64)
        self.conv21 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1, bias=False)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, dilation=2, padding=4,bias=False)
        self.norm3 = nn.BatchNorm2d(num_features=128)
        self.conv31 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,padding=1, bias=False)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, dilation=2, padding=4,bias=False)
        self.norm4 = nn.BatchNorm2d(num_features=256)
        self.conv41 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,padding=1, bias=False)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, dilation=2, padding=4,bias=False)
        self.norm5 = nn.BatchNorm2d(num_features=512)
        self.conv51 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,padding=1, bias=False)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=2, dilation=2, padding=4,bias=False)
        self.norm6 = nn.BatchNorm2d(num_features=1024)
        # self.conv61 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, bias=False)

        self.dconv6 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1,output_padding=1,bias=False)
        self.norm7 = nn.BatchNorm2d(num_features=512)

        self.dconv5 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=3, stride=2, padding=1,output_padding=1,bias=False)
        self.norm8 = nn.BatchNorm2d(num_features=256)

        self.dconv4 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1,output_padding=1,bias=False)
        self.norm9 = nn.BatchNorm2d(num_features=128)

        self.dconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, stride=2, padding=1,output_padding=1,bias=False)
        self.norm10 = nn.BatchNorm2d(num_features=64)

        self.dconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1,output_padding=1,bias=False)
        self.norm11 = nn.BatchNorm2d(num_features=32)

        self.dconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,output_padding=1,bias=False)
        self.norm12 = nn.BatchNorm2d(num_features=32)

        self.conv0 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, bias=False)
        #self.norm0 = nn.BatchNorm2d(num_features=1)
        #self.func = nn.Sigmoid()

    def _init_weights(self):
        class_name = self.Lesion_net.__class__.__name__
        if class_name.find('conv') != -1:
            self.Lesion_net.weight.data.normal_(0, 0.02)
        if class_name.find('norm') != -1:
            self.Lesion_net.weight.data.normal_(1, 0.02)

    def forward(self, x):
        out1 = self.activate(self.norm1(self.conv1(x)))
        out2 = self.activate(self.norm2(self.conv2(out1)))
        out3 = self.activate(self.norm3(self.conv3(out2)))
        out4 = self.activate(self.norm4(self.conv4(out3)))
        out5 = self.activate(self.norm5(self.conv5(out4)))
        out6 = self.activate(self.norm6(self.conv6(out5)))
        out7 = self.activate(self.norm7(self.dconv6(out6)))
        # print(out1.size(),self.conv21(out2).size(),self.conv31(out3).size(),self.conv51(out5).size(),self.conv61(out6).size(),out7.size())
        input = torch.cat((self.activate(self.norm5(self.conv51(out5)))+out5,out7),1)
        out8 = self.activate(self.norm8(self.dconv5(input)))
        input = torch.cat((self.activate(self.norm4(self.conv41(out4)))+out4,out8),1)
        out9 = self.activate(self.norm9(self.dconv4(input)))
        input = torch.cat((self.activate(self.norm3(self.conv31(out3)))+out3,out9),1)
        out10 = self.activate(self.norm10(self.dconv3(input)))
        input = torch.cat((self.activate(self.norm2(self.conv21(out2)))+out2,out10),1)
        out11 = self.activate(self.norm11(self.dconv2(input)))
        input = torch.cat((self.activate(self.norm1(self.conv11(out1)))+out1,out11),1)
        out12 = self.norm12(self.dconv1(input))
        return self.activate2(self.conv0(out12))

if __name__ == '__main__':
    input = Variable(torch.randn(10, 3, 1024, 1024))
    les = Lesion_net()
    A = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3,padding=1)
    output = les(input)
    print(output.size())