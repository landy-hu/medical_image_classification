import  torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil
import numpy as np
import random
import PIL.Image as img
from skimage import io

def save_checkpoint(state, filename='checkpoint.pth.tar', dir=None):
    torch.save(state, os.path.join(dir, filename))

    shutil.copyfile(os.path.join(dir, filename),
                    os.path.join(dir, 'latest.pth.tar'))
    return
def show(input):
    input = input.data.cpu().numpy()
    input = input[0, 0]
    # input[input>0]=255
    # print(input.shape)
    # image = np.array(input[np.where(input > 0.5))
    images = img.fromarray(input)
    images.show()
    # io.imshow(input)
    # images.pause(1)
    return