import tensorflow as tf
import numpy as np
from scipy import signal
import os
import PIL.Image as pim
import matplotlib.pyplot as plt
#import cv2
import warnings
#from skimage import exposure

warnings.filterwarnings('ignore')



def rescalen1p1(I):
    minI = I.min()
    I = I - minI
    maxI= I.max()
    I = I/maxI*2
    meanI=np.mean(I)
    I=I-meanI
#    cv2.imshow('image',I)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    plt.hist(I[:,:,0],bins=256, normed=1,edgecolor='None',facecolor='blue')
    return I


    
def plothist(Y):
    shape=np.shape(Y)
    nb=shape[0]
    nh=shape[1]
    nw=shape[2]
    nc=shape[3]
    fig = plt.figure()
    if nc==1:
        Yf=np.reshape(Y,(nb*nh*nw*nc))
        plt.hist(Yf,bins=256, normed=1,edgecolor='None',facecolor='blue')
        #plt.show()
    elif nc==3:
        Yfr=np.reshape(Y[:,:,:,0],(nb*nh*nw))
        plt.hist(Yfr, bins=256, normed=1,facecolor='r',edgecolor='r',alpha = 0.3,hold=1)
        Yfg=np.reshape(Y[:,:,:,1],(nb*nh*nw))
        plt.hist(Yfg, bins=256, normed=1,facecolor='g',edgecolor='g',alpha = 0.3,hold=1)
        Yfb=np.reshape(Y[:,:,:,2],(nb*nh*nw))
        plt.hist(Yfb, bins=256, normed=1,facecolor='b',edgecolor='b',alpha = 0.3,hold=1)
        #plt.show()
    return fig

