from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib import cm
import scipy.misc
def imnormalzo(image,channel):
#    for ch in range(channel):
#        image[:,:,ch]= (image[:,:,ch]-(image[:,:,ch]).min())/(
#                (image[:,:,ch]).max()-(image[:,:,ch]).min()) 
    if channel==1:
        immin=(image[:,:]).min()
        immax=(image[:,:]).max()
        image=(image-immin)/(immax-immin+1e-8)
    else:
        immin=(image[:,:,:]).min()
        immax=(image[:,:,:]).max()
        image=(image-immin)/(immax-immin+1e-8)
    return image

def plot(samples,Nh,Nc,channel,IMG_HEIGHT, IMG_WIDTH):
    fig = plt.figure(figsize=(10, 10))
    #fig = plt.figure()
    gs = gridspec.GridSpec(Nh, Nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples[0:Nh*Nc,:,:,:]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channel==1:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH)
        else:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH,channel)
        #image=imnormalzo(image,channel)
        image=np.clip(image,-1,1)
        image=image*0.5+0.5
        if channel==1:
            plt.imshow(image,cmap=plt.cm.gray)
        else:
            plt.imshow(image)
    return fig 
def mysaveimg(filename,mat,channel,IMG_HEIGHT, IMG_WIDTH):
    image=mat.reshape(IMG_HEIGHT, IMG_WIDTH,channel)
    image=imnormalzo(image,channel)
    #image = image.resize(im_size[0:2],Image.LANCZOS)    
    scipy.misc.imsave(filename, image)

def mysaveimgori(filename,mat,channel,IMG_HEIGHT, IMG_WIDTH):
    image=mat.reshape(IMG_HEIGHT, IMG_WIDTH,channel)  
    scipy.misc.imsave(filename, image)
    
def plotnorm(samples,Nh,Nc,channel,IMG_HEIGHT, IMG_WIDTH):
    fig = plt.figure(figsize=(10, 10))
    #fig = plt.figure()
    gs = gridspec.GridSpec(Nh, Nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples[0:Nh*Nc,:,:,:]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channel==1:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH)
        else:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH,channel)
        image=imnormalzo(image,channel)
        #image=np.clip(image,-1,1)
        #image=image*0.5+0.5
        if channel==1:
            plt.imshow(image,cmap=plt.cm.gray)
        else:
            plt.imshow(image)
    return fig 

def plotnorm1(samples,Nh,Nc,channel,IMG_HEIGHT, IMG_WIDTH):
    fig = plt.figure(figsize=(20, 10))
    #fig = plt.figure()
    gs = gridspec.GridSpec(Nh, Nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples[0:Nh*Nc,:,:,:]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channel==1:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH)
        else:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH,channel)
        image=imnormalzo(image,channel)
        #image=np.clip(image,-1,1)
        #image=image*0.5+0.5
        if channel==1:
            plt.imshow(image,cmap=plt.cm.gray)
        else:
            plt.imshow(image)
    return fig 

def plot1(samples,Nh,Nc,channel,IMG_HEIGHT, IMG_WIDTH):
    fig = plt.figure(figsize=(20, 10))
    #fig = plt.figure()
    gs = gridspec.GridSpec(Nh, Nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples[0:Nh*Nc,:,:,:]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channel==1:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH)
        else:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH,channel)
        #image=imnormalzo(image,channel)
        image=np.clip(image,-1,1)
        image=image*0.5+0.5
        if channel==1:
            plt.imshow(image,cmap=plt.cm.gray)
        else:
            plt.imshow(image)
    return fig 


def plotonesave(samples,channel,IMG_HEIGHT, IMG_WIDTH,savepath,name):
    #fig = plt.figure(figsize=(20, 20))
    fig = plt.figure()

    #for i, sample in enumerate(samples):
    if 1:
        sample=samples
        plt.axis('off')
        ax = plt.subplot(1,1,1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if channel==1:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH)
        else:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH,channel)
        #
        #image=(image+1)/2
        image=imnormalzo(image,channel)
        plt.imshow(image)
        
        plt.savefig(os.path.join(savepath, name), bbox_inches='tight')
    plt.close(fig)   

def plotfilter1(samples,Nh,Nc,channel,IMG_HEIGHT, IMG_WIDTH):
    fig = plt.figure(figsize=(20, 20))
    #fig = plt.figure()
    gs = gridspec.GridSpec(Nh, Nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples[0:Nh*Nc,:,:]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channel==1:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH)
            plt.imshow(image,cmap=plt.cm.gray)
        else:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH,channel)
            image=imnormalzo(image,channel)
            plt.imshow(image)
    return fig 

def plotfilter(samples,Nh,Nc,channel,IMG_HEIGHT, IMG_WIDTH):
    fig = plt.figure(figsize=(20, 10))
    #fig = plt.figure()
    gs = gridspec.GridSpec(Nh, Nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples[0:Nh*Nc,:,:]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channel==1:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH)
            plt.imshow(image,cmap=plt.cm.gray)
        else:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH,channel)
            image=imnormalzo(image,channel)
            plt.imshow(image)
    return fig 