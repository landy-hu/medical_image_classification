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
        self.norm1 = nn.BatchNorm2d(num_features=32)
        self.norm2 = nn.BatchNorm2d(num_features=64)
        self.conv11 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1,padding=1)
        # self.norm11 = nn.BatchNorm2d(num_features=32)
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1)


        self.conv21 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding=1)
        self.conv22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv23 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # self.norm2 = nn.BatchNorm2d(num_features=64)

        self.conv31 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding=1)
        self.conv32 = nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.norm3 = nn.BatchNorm2d(num_features=128)

        self.conv41 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding=1)
        self.conv42 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv43 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.norm4 = nn.BatchNorm2d(num_features=256)

        self.conv51 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding=1)
        self.conv52 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv53 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1,output_padding=1)
        # self.norm5 = nn.BatchNorm2d(num_features=512)


        # self.dconv = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1,output_padding=1)
        self.conv61 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv62 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv63 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        # self.norm7 = nn.BatchNorm2d(num_features=512)

        # self.dconv71 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1,output_padding=1)
        self.conv71 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv72 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv73 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1,output_padding=1)


        self.conv81 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv82 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv83 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        # self.norm9 = nn.BatchNorm2d(num_features=128)

        self.conv91 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv92 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv93 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv94 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)


        # self.func = nn.Sigmoid()

    def _init_weights(self):
        class_name = self.Lesion_net.__class__.__name__
        if class_name.find('conv') != -1:
            self.Lesion_net.weight.data.normal_(0, 0.02)
        if class_name.find('norm') != -1:
            self.Lesion_net.weight.data.normal_(1, 0.02)

    def forward(self, x):
        out_1 = self.conv12(self.conv11(x))

        out1 = self.conv21(out_1)
        out2 = self.conv22(out1)
        input = out1+out2
        out_2 = self.conv23(input)

        out1 = self.conv31(out_2)
        out2 = self.conv32(out1)
        input = out1+out2
        out_3 = self.conv33(input)

        out1 = self.conv41(out_3)
        out2 = self.conv42(out1)
        input = out1+out2
        out_4 = self.conv43(input)

        out1 = self.conv51(out_4)
        out2 = self.conv52(out1)
        input = out1+out2
        out_5 = self.conv53(input)


        input = torch.cat((out_4,out_5),1)
        out1 = self.conv61(input)
        out2 = self.conv62(out1)
        input = out1+out2
        out_6 = self.conv63(input)

        input = torch.cat((out_3,out_6),1)
        out1 = self.conv71(input)
        out2 = self.conv72(out1)
        input = out1+out2
        out_7 = self.conv73(input)

        input = torch.cat((out_2, out_7), 1)
        out1 = self.conv81(input)
        out2 = self.conv82(out1)
        input = out1+out2
        out_8 = self.conv83(input)

        input = torch.cat((out_1, out_8), 1)
        out1 = self.conv91(input)
        out2 = self.conv92(out1)
        input = out1+out2
        out_9 = self.conv93(input)
        return self.conv94(out_9)

if __name__ == '__main__':

    input = Variable(torch.randn(4, 3, 720, 720))
    les = Lesion_net()
    conv11 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
    out = les(input)
    print(out.size())