from __future__ import print_function
from os.path import exists, join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale
import torch.utils.data as data
from libtiff import TIFF
from os import listdir
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from skimage import io
from src.tools import *
from scipy.ndimage import filters
def get_train_data(path,mode1,mode2):
    return DatasetFromFolder(path, mode1, mode2)
class DatasetFromFolder(data.Dataset):
    def __init__(self, path, mode1, mode2):
        self.path=path
        super(DatasetFromFolder, self).__init__()
        self.mode1 =mode1
        self.dirMask =[]
        self.dirOriginal=[]
        self.mode2 = mode2
        dir2 = join(self.path,self.mode2)
        for x in listdir(dir2):
            # print(x[-3:])
            if x[-3:] == 'tif':
                self.dirMask.append(join(self.path, self.mode2, x))
                temp = x[:8]+x[-8:-4]+'.jpg'
                # print(temp)
                # print(temp)
                self.dirOriginal.append(join(self.path, self.mode1, temp))

    def __getitem__(self, index):
        img = np.array(Image.open(self.dirMask[index]).resize((1024,1024))).astype(np.float32)
        img[np.where(img>0)]=255
        data1 = filters.gaussian_filter(img,3)
        # print(data1)
        # print(data1[np.where(data1>0)])
        # print(self.dirOriginal[index])
        data2 = np.array(Image.open(self.dirOriginal[index]).resize((1024,1024))).astype(np.float32)
        data = np.zeros((4,1024,1024))
        data[0,:,:] = data1
        data[1:,:,:]=data2.transpose((2,0,1))
        return data
    def __len__(self):
        return len(self.dirMask)

if __name__=='__main__':
    path ='/home/mpl/medical_image_classification/ISBI_DATASET/Lesion_Segmentation'
    trainData = DataLoader(get_train_data(path), 20, shuffle=True)
    # for i, data in enumerate(trainData):
        # print(data.size())