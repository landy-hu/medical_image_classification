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
from os.path import exists, join
from src.tools import *
from scipy.ndimage import filters
from torch.autograd import Variable
from src.LESION_NET import Lesion_net
def mask_test(path,mode1,mode2):
    # get_test_data(path)
    image1_filenames = [x for x in listdir(join(path,mode1))]
    image1_filenames1 = [x for x in listdir(join(path, 'valiGroundTruth'))]
    Lesnet = torch.nn.DataParallel(Lesion_net()).cuda()
    Lesnet.load_state_dict(torch.load('/home/mpl/medical_image_classification/checkpoints/Stack2_Lesnet15.pth.tar'))
    Lesnet.eval()
    for index in range(0,len(image1_filenames)):
        # print(image1_filenames[index])
        dir = join(path,mode1,image1_filenames[index])
        img = np.array(Image.open(dir)).transpose((2,0,1))
        dir1 = join(path, 'valiGroundTruth', image1_filenames1[index])
        img1 = np.array(Image.open(dir1))
        print(img.shape)
        # np.transpose(img,(2,0,1))
        # print(img.shape)
        mask = np.zeros((3400,3400))
        for j in range(0,500):
            top = np.random.randint(0, 2196)
            left = np.random.randint(0, 2196)
            temp = torch.from_numpy(img[:,top:top + 1024, left:left + 1024])
            # temp = torch.from_numpy(img)
            input = torch.zeros(1,3,1024,1024)
            input[0,:,:,:]=temp[:,:,:]
            # print(input.size())
            input = Variable(input).float().cuda()
            # show(input, 1)
            output= Lesnet(input)
            # show(output,0)
            temp = output.data.cpu().numpy().reshape((1024,1024))
            temp[np.where(temp<0)]=0
            temp[np.where(temp >1)] = 1
            # print(np.where(temp>0.3))
            mask[top:top + 1024, left:left + 1024] = temp[:,:]
            # mask = temp*255
        im = Image.fromarray(mask)
        show(Variable(torch.from_numpy(mask)).float().cuda(),0)
        name = str(index)+'.tif'
        im.save(join(path,mode2,name))

if __name__=='__main__':
    path ='/home/mpl/medical_image_classification/ISBI_DATASET/Lesion_Segmentation'
    mask_test(path, 'valiInput', 'mask_inter')
