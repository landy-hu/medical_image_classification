# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = 'lan hu'
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
import torch.nn
import os
from torch.autograd import Variable
from PIL import Image
from src.tools import *
import torchvision.transforms as transforms
from src.LESION_NET import Lesion_net
from src.loader import get_train_data
import torch.cuda
import numpy as np
def loss(x,mask,f):
    # temp0 = mask.data.cpu().numpy()
    temp = (mask-x).data.cpu().numpy()
    idx1 = np.where(temp>170)
    idx0 = np.where(temp<-80)
    return (idx1[0].shape[0])/(idx0[0].shape[0]+0.001)*f((mask/255)*x,mask)+f((1-mask/255)*x,mask)

class options():
    def __init__(self):
        self.lam = 0.1
        self.tau =1
        self.mu =1
        self.batch_size = 10# training batch size
        self.num_epochs = 500  # umber of epochs to train for
        self.learning_rate = 0.0001
        self.dataPath  = '/home/mpl/medical_image_classification/ISBI_DATASET/Lesion_Segmentation/'
        self.layers = 10
        self.vols = 100
        self.num_feature = 1024
        self.f = nn.MSELoss()

def train(Lesnet, train_dataset, val_dataset, options, epoch):
    lossData = 0
    lesnet_optimizer = torch.optim.Adam(Lesnet.parameters(), lr=options.learning_rate,betas=(0.5,0.999))
    for i, data in enumerate(train_dataset):
        # sizes = data.size()
        if data.size(0) < options.batch_size:
            lossData_val = 0
            for j, val_data in enumerate(val_dataset):
                if val_data.size(0) < options.batch_size:
                    print('val---Epoch [{}/{}/{}], loss: {:.6f}'.format(i, epoch, options.num_epochs, lossData_val / j))
                    show(xx_val)
                    show(mask)
                    continue
                input = torch.zeros(options.batch_size, 3, options.num_feature, options.num_feature)
                mask = torch.zeros(options.batch_size, 1, options.num_feature, options.num_feature)

                mask[:, :, :, :] = val_data[:, 0]
                input[:, :, :, :] = val_data[:, 1:]
                input = Variable(input).float().cuda()
                mask = Variable(mask).float().cuda()
                xx_val = Lesnet(input)
                loss_lesnet_val = loss(xx_val, mask, options.f)
                lossData_val += loss_lesnet_val.data[0]
            print('train---Epoch [{}/{}/{}], loss: {:.6f}'.format(i, epoch, options.num_epochs, lossData/i))
            # show(xx)
            continue
        else:
            input = torch.zeros(options.batch_size, 3, options.num_feature, options.num_feature)
            mask = torch.zeros(options.batch_size, 1, options.num_feature, options.num_feature)
            # print('hello')
            mask[:, :, :, :] = data[:, 0]
            input[:,:,:,:]=data[:,1:]
            input = Variable(input).float().cuda()
            mask = Variable(mask).float().cuda()
            lesnet_optimizer.zero_grad()
            xx = Lesnet(input)
            loss_lesnet = loss(xx, mask, options.f)
            loss_lesnet.backward()
            lesnet_optimizer.step()
            lossData += loss_lesnet.data[0]

if __name__ == '__main__':
    options = options()
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    train_dataset = DataLoader( get_train_data(options.dataPath,'trainInput','trainGroundTruth'), options.batch_size, shuffle=True)
    val_dataset = DataLoader( get_train_data(options.dataPath,'valiInput','valiGroundTruth'), options.batch_size, shuffle=True)
    Lesnet = torch.nn.DataParallel(Lesion_net()).cuda()
    # Lesnet.load_state_dict(torch.load('/home/mpl/medical_image_classification/checkpoints/Stack1_Lesnet60.pth.tar'))
    # print(Lesnet)
    for epoch in range(1, options.num_epochs):
        train(Lesnet, train_dataset, val_dataset, options, epoch)
        if (epoch+1) % 5 == 0:
            save_checkpoint(Lesnet.state_dict(), filename='Stack_new_Lesnet{}.pth.tar'.format(epoch + 1), dir="checkpoints")
