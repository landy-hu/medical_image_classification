# -*- coding: utf-8 -*-
__author__ = 'lan hu'
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
import torch.nn
import os
from PIL import Image
from src.tools import *
import torchvision.transforms as transforms
from src.LESION_NET import Lesion_net
from src.loader import get_train_data
import torch.cuda
import numpy as np
import pylab as pl
def loss(x,xx):
    loss = F.mse_loss(x,xx)
    # loss +=options.lam*l1_loss(s)
    return loss

class options():
    def __init__(self):
        self.lam = 0.1
        self.tau =1
        self.mu =1
        self.batch_size = 4# training batch size
        self.num_epochs = 500  # umber of epochs to train for
        self.learning_rate = 0.0001
        self.dataPath  = '/home/mpl/medical_image_classification/ISBI_DATASET/Lesion_Segmentation/'
        self.layers = 10
        self.vols = 100
        self.num_feature = 1024

def accu(Lesnet, val_dataset, SE_dataset,MA_dataset,EX_dataset,HE_dataset,options):
    lossData = 0
    lossData_val = 0
    acc = np.zeros((18))
    for thres in range(0,18):
        tp=0
        fp=0
        tn=0
        fn=0
        for j, val_data in enumerate(val_dataset):
            # print(val_data.size())
            if val_data.size(0) < options.batch_size:
                continue
            input = torch.zeros(options.batch_size, 3, options.num_feature, options.num_feature)
            mask = torch.zeros(options.batch_size, 1, options.num_feature, options.num_feature)
            mask[:, :, :, :] = val_data[:, 0]
            input[:, :, :, :] = val_data[:, 1:]
            input = Variable(input).float().cuda()
            mask = Variable(mask).float().cuda()
            xx_val = Lesnet(input)
            xx_val = xx_val.data.cpu().numpy()
            xx_val[np.where(xx_val <thres*10)] = 0
            xx_val[np.where(xx_val >=thres*10)]=1

            mask = mask.data.cpu().numpy()
            mask[np.where(mask==255)]=1
            idx1 = np.where(mask==1)
            idx0 = np.where(mask<1)
            # xx_val = Variable(torch.from_numpy(xx_val)).float().cuda()
            # loss_lesnet_val = loss(xx_val, mask)
            # mask_test = np.zeros((10,1,1024,1024))
            error = mask-xx_val
            idx = np.where(error[idx1] == 0)
            tp += idx[0].shape[0]
            idx = np.where(error[idx1] == 1)
            fn += idx[0].shape[0]

            idx = np.where(error[idx0] == 0)
            tn += idx[0].shape[0]
            idx = np.where(error[idx0] == -1)
            fp += idx[0].shape[0]
        p = (tp)/(tp+fp)
        r = (tp)/(tp+fn)
        acc[thres] = (2*p*r)/(p+r)
        print(acc[thres])
    # print('SE---------------------------------------------')
    # acc1 = np.zeros((18))
    # for thres in range(0, 18):
    #     tp = 0
    #     fp = 0
    #     tn = 0
    #     fn = 0
    #     for j, val_data in enumerate(SE_dataset):
    #         # print(val_data.size())
    #         if val_data.size(0) < options.batch_size:
    #             continue
    #         input = torch.zeros(options.batch_size, 3, options.num_feature, options.num_feature)
    #         mask = torch.zeros(options.batch_size, 1, options.num_feature, options.num_feature)
    #         mask[:, :, :, :] = val_data[:, 0]
    #         input[:, :, :, :] = val_data[:, 1:]
    #         input = Variable(input).float().cuda()
    #         mask = Variable(mask).float().cuda()
    #         xx_val = Lesnet(input)
    #         xx_val = xx_val.data.cpu().numpy()
    #         xx_val[np.where(xx_val < thres * 10)] = 0
    #         xx_val[np.where(xx_val >= thres * 10)] = 1
    #
    #         mask = mask.data.cpu().numpy()
    #         mask[np.where(mask == 255)] = 1
    #         idx1 = np.where(mask == 1)
    #         idx0 = np.where(mask < 1)
    #         # xx_val = Variable(torch.from_numpy(xx_val)).float().cuda()
    #         # loss_lesnet_val = loss(xx_val, mask)
    #         # mask_test = np.zeros((10,1,1024,1024))
    #         error = mask - xx_val
    #         idx = np.where(error[idx1] == 0)
    #         tp += idx[0].shape[0]
    #         idx = np.where(error[idx1] == 1)
    #         fn += idx[0].shape[0]
    #
    #         idx = np.where(error[idx0] == 0)
    #         tn += idx[0].shape[0]
    #         idx = np.where(error[idx0] == -1)
    #         fp += idx[0].shape[0]
    #     p = (tp) / (tp + fp)
    #     r = (tp) / (tp + fn)
    #     acc1[thres] = (2 * p * r) / (p + r)
    #     print(acc1[thres])
    # print('MA------------------------------')
    # acc2 = np.zeros((18))
    # for thres in range(0, 18):
    #     tp = 0
    #     fp = 0
    #     tn = 0
    #     fn = 0
    #     for j, val_data in enumerate(MA_dataset):
    #         # print(val_data.size())
    #         if val_data.size(0) < options.batch_size:
    #             continue
    #         input = torch.zeros(options.batch_size, 3, options.num_feature, options.num_feature)
    #         mask = torch.zeros(options.batch_size, 1, options.num_feature, options.num_feature)
    #         mask[:, :, :, :] = val_data[:, 0]
    #         input[:, :, :, :] = val_data[:, 1:]
    #         input = Variable(input).float().cuda()
    #         mask = Variable(mask).float().cuda()
    #         xx_val = Lesnet(input)
    #         xx_val = xx_val.data.cpu().numpy()
    #         xx_val[np.where(xx_val < thres * 10)] = 0
    #         xx_val[np.where(xx_val >= thres * 10)] = 1
    #
    #         mask = mask.data.cpu().numpy()
    #         mask[np.where(mask == 255)] = 1
    #         idx1 = np.where(mask == 1)
    #         idx0 = np.where(mask < 1)
    #         # xx_val = Variable(torch.from_numpy(xx_val)).float().cuda()
    #         # loss_lesnet_val = loss(xx_val, mask)
    #         # mask_test = np.zeros((10,1,1024,1024))
    #         error = mask - xx_val
    #         idx = np.where(error[idx1] == 0)
    #         tp += idx[0].shape[0]
    #         idx = np.where(error[idx1] == 1)
    #         fn += idx[0].shape[0]
    #
    #         idx = np.where(error[idx0] == 0)
    #         tn += idx[0].shape[0]
    #         idx = np.where(error[idx0] == -1)
    #         fp += idx[0].shape[0]
    #     p = (tp) / (tp + fp)
    #     r = (tp) / (tp + fn)
    #     acc2[thres] = (2 * p * r) / (p + r)
    #     print(acc2[thres])
    # print('HE-------------------------------')
    # acc3 = np.zeros((18))
    # for thres in range(0, 18):
    #     tp = 0
    #     fp = 0
    #     tn = 0
    #     fn = 0
    #     for j, val_data in enumerate(HE_dataset):
    #         # print(val_data.size())
    #         if val_data.size(0) < options.batch_size:
    #             continue
    #         input = torch.zeros(options.batch_size, 3, options.num_feature, options.num_feature)
    #         mask = torch.zeros(options.batch_size, 1, options.num_feature, options.num_feature)
    #         mask[:, :, :, :] = val_data[:, 0]
    #         input[:, :, :, :] = val_data[:, 1:]
    #         input = Variable(input).float().cuda()
    #         mask = Variable(mask).float().cuda()
    #         xx_val = Lesnet(input)
    #         xx_val = xx_val.data.cpu().numpy()
    #         xx_val[np.where(xx_val < thres * 10)] = 0
    #         xx_val[np.where(xx_val >= thres * 10)] = 1
    #
    #         mask = mask.data.cpu().numpy()
    #         mask[np.where(mask == 255)] = 1
    #         idx1 = np.where(mask == 1)
    #         idx0 = np.where(mask < 1)
    #         # xx_val = Variable(torch.from_numpy(xx_val)).float().cuda()
    #         # loss_lesnet_val = loss(xx_val, mask)
    #         # mask_test = np.zeros((10,1,1024,1024))
    #         error = mask - xx_val
    #         idx = np.where(error[idx1] == 0)
    #         tp += idx[0].shape[0]
    #         idx = np.where(error[idx1] == 1)
    #         fn += idx[0].shape[0]
    #
    #         idx = np.where(error[idx0] == 0)
    #         tn += idx[0].shape[0]
    #         idx = np.where(error[idx0] == -1)
    #         fp += idx[0].shape[0]
    #     p = (tp) / (tp + fp)
    #     r = (tp) / (tp + fn)
    #     acc3[thres] = (2 * p * r) / (p + r)
    #     print(acc3[thres])
    # print('EX---------------------')
    # acc4 = np.zeros((18))
    # for thres in range(0, 18):
    #     tp = 0
    #     fp = 0
    #     tn = 0
    #     fn = 0
    #     for j, val_data in enumerate(EX_dataset):
    #         # print(val_data.size())
    #         if val_data.size(0) < options.batch_size:
    #             continue
    #         input = torch.zeros(options.batch_size, 3, options.num_feature, options.num_feature)
    #         mask = torch.zeros(options.batch_size, 1, options.num_feature, options.num_feature)
    #         mask[:, :, :, :] = val_data[:, 0]
    #         input[:, :, :, :] = val_data[:, 1:]
    #         input = Variable(input).float().cuda()
    #         mask = Variable(mask).float().cuda()
    #         xx_val = Lesnet(input)
    #         xx_val = xx_val.data.cpu().numpy()
    #         xx_val[np.where(xx_val < thres * 10)] = 0
    #         xx_val[np.where(xx_val >= thres * 10)] = 1
    #
    #         mask = mask.data.cpu().numpy()
    #         mask[np.where(mask == 255)] = 1
    #         idx1 = np.where(mask == 1)
    #         idx0 = np.where(mask < 1)
    #         # xx_val = Variable(torch.from_numpy(xx_val)).float().cuda()
    #         # loss_lesnet_val = loss(xx_val, mask)
    #         # mask_test = np.zeros((10,1,1024,1024))
    #         error = mask - xx_val
    #         idx = np.where(error[idx1] == 0)
    #         tp += idx[0].shape[0]
    #         idx = np.where(error[idx1] == 1)
    #         fn += idx[0].shape[0]
    #
    #         idx = np.where(error[idx0] == 0)
    #         tn += idx[0].shape[0]
    #         idx = np.where(error[idx0] == -1)
    #         fp += idx[0].shape[0]
    #     p = (tp) / (tp + fp)
    #     r = (tp) / (tp + fn)
    #     acc4[thres] = (2 * p * r) / (p + r)
    #     print(acc4[thres])
        # record[thres, 2] = acc
        # print(acc)
            # error1 = np.abs(mask-mask_test)
            # num = np.where(error==0)
            # num1 = np.where(error1==0)
            # print(num[0].shape[0]/(options.batch_size*1024*1024))
            # print(num1[0].shape[0] / (options.batch_size * 1024 * 1024))
        # lossData_val += loss_lesnet_val.data[0]
    # print('loss: {:.6f}'.format(lossData_val / j))
    # show(xx_val)
    # show(input)

    pl.plot(record[:,0],record[:,1])
    pl.show()
    # show(mask)


if __name__ == '__main__':
    options = options()
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    val_dataset = DataLoader( get_train_data(options.dataPath,'valiInput','valiGroundTruth'), options.batch_size, shuffle=True)
    SE_dataset = DataLoader(get_train_data(options.dataPath, 'valiInput', 'SE_au'), options.batch_size, shuffle=True)
    MA_dataset = DataLoader(get_train_data(options.dataPath, 'valiInput', 'MA_au'), options.batch_size, shuffle=True)
    EX_dataset = DataLoader(get_train_data(options.dataPath, 'valiInput', 'EX_au'), options.batch_size, shuffle=True)
    HE_dataset =DataLoader( get_train_data(options.dataPath,'valiInput','HE_au'), options.batch_size, shuffle=True)
    Lesnet = torch.nn.DataParallel(Lesion_net()).cuda()
    Lesnet.load_state_dict(torch.load('/home/mpl/medical_image_classification/checkpoints/Stack1_Lesnet35.pth.tar'))

    accu(Lesnet, val_dataset,SE_dataset,MA_dataset,EX_dataset,HE_dataset, options)

