from __future__ import print_function
from os.path import exists, join
from torchvision.transforms import RandomCrop,Compose, CenterCrop, ToTensor, Scale
import torch.utils.data as data
from libtiff import TIFF
from os import listdir
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from skimage import io
from src.tools import *
from scipy.ndimage import filters
from torch.autograd import Variable
#import torchvision.transforms.functional as TF
# def input_transform(crop_size):
#     return Compose([RandomCrop(crop_size),ToTensor])
def get_train_data(path,mode1,mode2):
    return DatasetFromFolder(path, mode1, mode2)
def get_test_data(path):
    return testDataFromFolder(path)

class testDataFromFolder(data.Dataset):
    def __init__(self, path):
        super(testDataFromFolder, self).__init__()
        self.path = path
        self.image1_filenames = [x for x in listdir(path)]
    def __getitem__(self, index):
        mask = np.array(Image.open(self.image1_filenames[index])).transpose((2, 0, 1))
        return mask
    def __len__(self):
        return len(self.image1_filenames)
class DatasetFromFolder(data.Dataset):
    def __init__(self, path, mode1, mode2):

        super(DatasetFromFolder, self).__init__()
        self.path = path
        self.mode1 =mode1
        self.res = 3400
        self.dirMask =[]
        self.dirOriginal=[]
        self.mode2 = mode2
        self.trans = RandomCrop(1024)
        dir2 = join(self.path,self.mode2)
        for x in listdir(dir2):
            # print(x[-3:])
            if x[-3:] == 'tif':
                self.dirMask.append(join(self.path, self.mode2, x))
                temp = x[:8]+x[-9:-4]+'.jpg'
                self.dirOriginal.append(join(self.path, self.mode1, temp))

    def __getitem__(self, index):
        mask = Image.open(self.dirMask[index]).resize((1024,1024))
        image = Image.open(self.dirOriginal[index]).resize((1024,1024))
        mask = np.array(mask)
        image = np.array(image)
        data = np.zeros((4, 1024,1024))
        data[0, :, :] = mask
        data[1:, :, :] = image.transpose((2, 0, 1))

        return data
        # return data
    def __len__(self):
        return len(self.dirMask)

if __name__=='__main__':
    path ='/home/mpl/medical_image_classification/ISBI_DATASET/Lesion_Segmentation/'
    train_dataset = DataLoader(get_train_data(path, 'valiInput', 'valiGroundTruth'),
                               10, shuffle=True)
    # trainData = DataLoader(get_train_data(path), 20, shuffle=True)
    for i, data in enumerate(train_dataset):
        input = torch.zeros(10, 3, 1024, 1024)
        mask = torch.zeros(10, 1, 1024, 1024)
        # input = torch.zeros(10, 3, 3400, 3400)
        # mask = torch.zeros(10, 1, 3400, 3400)
        print(data.size())
        mask[:, :, :, :] = data[:, 0,:,:]
        input[:, :, :, :] = data[:, 1:,:,:]
        input = Variable(input).float().cuda()
        mask = Variable(mask).float().cuda()
        # show(xx_val, 0)
        # show(input, 1)
        # show(mask, 0)
        print(data.size())

