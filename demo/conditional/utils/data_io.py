import os
import math
import numpy as np
from PIL import Image
import scipy.misc

from six.moves import xrange
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def LoadDataSet(pathlist, im_size, mode):
    # imgList = [f for path in paths for f in os.listdir(path)]
    imgList = pathlist
    imgList.sort()
    images = np.zeros((len(imgList), im_size[0], im_size[1], im_size[2])).astype(np.float32)
    # print('Loading dataset: {}'.format(path))
    print("Loading dataset...")
    for i in tqdm(xrange(len(imgList))):
            if im_size[2]==3:
                image = Image.open(imgList[i]).convert('RGB')
                image = image.resize(im_size[0:2],Image.LANCZOS)
                #LANCZOS,ANTIALIAS,BILINEAR,BICUBIC
            elif im_size[2]==1:
                image = Image.open(imgList[i]).convert('LA')
                #image = image.resize(im_size[0:2])
            if mode==1:
                image = np.array(image)[:,:,0:im_size[2]].astype(np.float32)/255
            elif mode==2:
                image = np.array(image)[:,:,0:im_size[2]].astype(np.float32)/255
                #max_val = image.max();min_val = image.min();image = (image - min_val) / (max_val - min_val)*2-1
                image=(image-0.5)/0.5
            images[i] = image.reshape(im_size)
    print('Data loaded, shape: {}'.format(images.shape))
    return images


def LoadDataSet_condition(pathlist, im_size, class_num, mode):
    # imgList = [f for path in paths for f in os.listdir(path)]
    imgList = pathlist
    imgList.sort()
    images = np.zeros((len(imgList), im_size[0], im_size[1], im_size[2])).astype(np.float32)
    labels = np.zeros((len(imgList), class_num)).astype(np.float32)
    # print('Loading dataset: {}'.format(path))
    print("Loading dataset...")
    for i in tqdm(xrange(len(imgList))):
        path, category = imgList[i]
        labels[i][category] = 1.0
        if im_size[2]==3:
            image = Image.open(path).convert('RGB')
            image = image.resize(im_size[0:2],Image.LANCZOS)
            #LANCZOS,ANTIALIAS,BILINEAR,BICUBIC
        elif im_size[2]==1:
            image = Image.open(path).convert('LA')
            #image = image.resize(im_size[0:2])
        if mode==1:
            image = np.array(image)[:,:,0:im_size[2]].astype(np.float32)/255
        elif mode==2:
            image = np.array(image)[:,:,0:im_size[2]].astype(np.float32)/255
            #max_val = image.max();min_val = image.min();image = (image - min_val) / (max_val - min_val)*2-1
            image=(image-0.5)/0.5
        images[i] = image.reshape(im_size)
        break
    print('Data loaded, shape: {}'.format(images.shape))
    print('Category loaded, shape: {}'.format(labels.shape))
    return images, labels












def LoadDataSetN(path, im_size, N,mode):
    #imgList = [f for f in os.listdir(path) if any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS)]
    imgList = [f for f in os.listdir(path)]
    imgList.sort()
    images = np.zeros((N, im_size[0], im_size[1], im_size[2]),dtype=np.float16)
    print('Loading dataset: {}'.format(path))
    for i in xrange(N):
            if im_size[2]==3:
                image = Image.open(os.path.join(path, imgList[i])).convert('RGB')
                image = image.resize(im_size[0:2],Image.LANCZOS)
                #LANCZOS,ANTIALIAS,BILINEAR,BICUBIC
            elif im_size[2]==1:
                image = Image.open(os.path.join(path, imgList[i])).convert('LA')
                #image = image.resize(im_size[0:2])
            if mode==1:
                image = np.array(image)[:,:,0:im_size[2]].astype(np.float32)/255
            elif mode==2:
                image = np.array(image)[:,:,0:im_size[2]].astype(np.float32)/255
                #max_val = image.max();min_val = image.min();image = (image - min_val) / (max_val - min_val)*2-1
                image=(image-0.5)/0.5
            images[i] = image.reshape(im_size)
    print('Data loaded, shape: {}'.format(images.shape))
    return images

def LoadDataSetlocal(path,ni, im_size, mode):
    #imgList = [f for f in os.listdir(path) if any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS)]
    imgList = [f for f in os.listdir(path)]
    imgList.sort()
    #nim=len(imgList)
    npatch=100
    images = np.zeros((npatch, im_size[0], im_size[1], im_size[2])).astype(np.float32)
    print('Loading dataset: {}'.format(path))
    for i in xrange(npatch):
            if im_size[2]==3:
                image = Image.open(os.path.join(path, imgList[npatch*ni+i])).convert('RGB')
                image = image.resize(im_size[0:2],Image.LANCZOS)
                #LANCZOS,ANTIALIAS,BILINEAR,BICUBIC
            elif im_size[2]==1:
                image = Image.open(os.path.join(path, imgList[i])).convert('LA')
                #image = image.resize(im_size[0:2])
            if mode==1:
                image = np.array(image)[:,:,0:im_size[2]].astype(np.float32)/255
            elif mode==2:
                image = np.array(image)[:,:,0:im_size[2]].astype(np.float32)/255
                #max_val = image.max();min_val = image.min();image = (image - min_val) / (max_val - min_val)*2-1
                image=(image-0.5)/0.5
            images[i] = image.reshape(im_size)
    print('Data loaded, shape: {}'.format(images.shape))
    return images


