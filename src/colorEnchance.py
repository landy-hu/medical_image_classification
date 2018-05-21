import csv
import cv2
import glob
import os
import numpy as np
import concurrent.futures
import socket
import shutil
scale = 500
pool_size = 20
image_folder = '/home/mpl/medical_image_classification/ISBI_DATASET/Lesion_Segmentation/Apparent_Retinopathy/'
save_folder = '/home/mpl/medical_image_classification/ISBI_DATASET/Lesion_Segmentation/originalEnColor_stack_unet/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
else:
    shutil.rmtree(save_folder)
    os.makedirs(save_folder)
def scaleRadius(img, scale):
    x = img[img.shape[0]//2, :, :].sum(1)   #
    r = (x>x.mean()/10).sum()/2
    if r == 0.0:
        s = 1.0
    else:
        s = scale*1.0/r
    return cv2.resize(img, (0,0), fx=s, fy=s)

def scale_enhance_rotate(img_name):
    img_path = os.path.join(image_folder, img_name)
    try:
        a = cv2.imread(img_path)
        # a = scaleRadius(a, scale)
        a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)
        # b = np.zeros(a.shape)
        # cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(scale*0.95),(1,1,1),-1,8,0)
        # a = a*b + 128 * (1 - b)

        path_all = os.path.join(save_folder, img_name)
        cv2.imwrite(path_all, a)
        print(path_all)

    except FileNotFoundError(img_path):
        print(img_path)

if __name__ == "__main__":

    for image_name in os.listdir(image_folder):
        scale_enhance_rotate(image_name)