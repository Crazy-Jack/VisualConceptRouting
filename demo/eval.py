#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:27:08 2019

@author: andy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 08:38:42 2019

@author: andy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 13:43:19 2019

@author: andy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:22:53 2019
save
@author: andy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:42:39 2019
sample
@author: andy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 03:50:52 2019
evalute
@author: andy
"""

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib.pyplot as plt
from utils.myplot import plot
from utils.myplot import plotnorm
from utils.myplot import plot1
from utils.myplot import plotnorm1
#from utils.myplot import plotfilter1
from utils.operation2 import block_down
from utils.operation2 import resblock_down
from utils.operation2 import block_up_no
from utils.operation2 import resblock_up_no
from utils.operation2 import block_up_cm_no
from utils.operation2 import block_up_mask_no
from utils.operation2 import resblock_up_cm_no
from utils.operation2 import resblock_up_mask_no
from utils.operation2 import conv2d
#from utils.operation2 import deconv2d
from utils.operation2 import lrelu
from utils.operation2 import dense
from utils.operation2 import minibatch_stddev_layer
from utils.operation2 import sparsech
from utils.operation2 import sparseop
from utils.operation2 import sparseopattributeq
from utils.operation2 import sparsechone
from utils.operation2 import sparsesingleopxylocat
from utils.operation2 import sparseopidx
from utils.operation2 import sparsesingleopattributeq
#from utils.data_io import LoadDataSetlocal
from skimage.transform import resize
#from utils.data_io import LoadDataSet
#from utils.imgpross import plothist
import time
from utils.myplot import mysaveimg
from utils.myplot import mysaveimgori
from tqdm import tqdm

dir = os.path.dirname(os.path.realpath(__file__))
#datapath='/home/andy/tensorcode/sparsegeneratornew/Dataset/bedroom10k128/'
#datapath='/home/andy/Downloads/lsun-master/Bedroom100K'
datapath='../data_unzip/bedroom_train_lmdb/lsun_bed100k/imgs'
#datapath='./Dataset/stlimgall'
special='SparsegeneratorAEadv10k64ResnonormandzCelebA'
dataname='celebA10k64'
figpath=dir+'/Results/'+dataname+'/'+'figuresevalep1100i0samplexylocation'+special+'/'
logpath=dir+'/Results/'+dataname+'/'
parampath=dir+'/Results/'+dataname+'/'+'params'+special+'/'
modelpath=dir+'/Results/'+dataname+'/'+'params'+special+'/'+'model.ckpt-255'
logfile=logpath+'log'+dataname+special

Nall=10000
BATCH_SIZE=100
IMG_HEIGHT = 64 # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 64 # CHANGE HERE, the image width to be resized to
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale
Nbatch= int(np.floor(Nall/BATCH_SIZE))
Nall=int(Nbatch*BATCH_SIZE)
Nepoch=2000
#train_set = LoadDataSet(datapath, im_size=[IMG_HEIGHT,IMG_WIDTH,CHANNELS],mode=2)
#train_set = LoadDataSetlocal(datapath,0,im_size=[IMG_HEIGHT,IMG_WIDTH,CHANNELS],mode=2)
tf.reset_default_graph()
z_dim = 100

lr=tf.placeholder(dtype=tf.float32,shape=())
lr_d=tf.placeholder(dtype=tf.float32,shape=())
z= tf.placeholder(shape=[BATCH_SIZE, z_dim], dtype=tf.float32, name='z')
z_s= tf.placeholder(shape=[BATCH_SIZE, z_dim], dtype=tf.float32, name='zs')
Y = tf.placeholder(tf.float32, shape = [None, IMG_HEIGHT,IMG_WIDTH,CHANNELS])
ip=tf.placeholder(dtype=tf.int32,shape=())
jp=tf.placeholder(dtype=tf.int32,shape=())
xp=tf.placeholder(dtype=tf.int32,shape=())
yp=tf.placeholder(dtype=tf.int32,shape=())
chb=64
channelnd = [chb*8,chb*8,chb*4,chb*2,chb,CHANNELS]
channelne = [CHANNELS,chb,chb*2,chb*4,chb*8,chb*8]
channelndes = [CHANNELS,chb,chb*2,chb*4,chb*8,chb*8]
kersized=[3,3,3,3,5]
kersizee=[5,3,3,3,3]
kersizedes=[5,3,3,3,3]
fmd=[4,8,16,32,64,64]
fme=[64,64,32,16,8,4]
fmdes=[64,64,32,16,8,4]
strided=[2,2,2,2,1]
stridee=[1,2,2,2,2]
stridedes=[1,2,2,2,2]
nzatte = 200 #global
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

def Q(x,reuse=False):
    with tf.variable_scope('infer', reuse=reuse):
        feamap0 =lrelu(conv2d(x, channelne[1],k=kersizee[0], s=1,name="conv2d_0"))
        feamap1 = block_down(feamap0,channelne[1],k=kersizee[1],s=1,name='block1') # 32x32
        feamap2 = resblock_down(feamap1,channelne[2],k=kersizee[2],s=1,name='block2') # 16x16
        feamap3 = block_down(feamap2,channelne[3],k=kersizee[3],s=1,name='block3') # 8x8
        feamap4 = block_down(feamap3,channelne[4],k=kersizee[4],s=1,name='block4') # 4x4
        #feamap5 = lrelu(conv2d(feamap4, channelne[5],k=kersizee[4], s=1,name="conv2d_5")) # 4x4
        #feamap5 = lrelu(conv2d(feamap5, channelne[5],k=4, s=1,padding='VALID',name="conv2d_6")) # 1x1
        feamap5 =tf.reshape(feamap4 ,[BATCH_SIZE,channelne[5]*fme[5]*fme[5]])
        za = tf.nn.tanh(dense(feamap5, z_dim,gain=1,name='fully_f1'))
        return za

def P(z,reuse=False):
    with tf.variable_scope('gen', reuse=reuse):
        #output_shape_da_conv5 = tf.stack([BATCH_SIZE, fmd[5], fmd[5], channelnd[5]])
        ch=fmd[0]*fmd[0]*channelnd[0]
        hf = dense(z, ch,gain=np.sqrt(2)/4,name='fully_f')
        hc=tf.reshape(hf,[-1, fmd[0], fmd[0], channelnd[0]])
        hc=tf.nn.relu(hc)
        hc=tf.nn.relu(conv2d(hc,nzatte,k=kersized[0], s=1,name="conv2d_0"))  # 4x4
        zattg=tf.reshape(hc,[BATCH_SIZE, fmd[0]*fmd[0],nzatte])
        zattg=sparsech(zattg,int(nzatte*0.1))
        fmd1=tf.zeros((BATCH_SIZE, fmd[0], fmd[0], fmd[0]*fmd[0]))
        fmsparsel1=sparseopattributeq(fmd1,zattg)
        fmsparsel1=tf.reshape(fmsparsel1,[BATCH_SIZE, fmd[0], fmd[0], nzatte*fmd[0]*fmd[0]])
        fmd2=block_up_no(fmsparsel1,channelnd[1],k=kersized[0], s=1,name='block1') #8x8
        fmsparsel2=sparseop(fmd2,int(1/4*fmd[1]*fmd[1]),1)
        fmd3=block_up_no(fmsparsel2,channelnd[2],k=kersized[1], s=1,name='block2')#16x16
        fmsparsel3=sparseop(fmd3,int(1/4*fmd[2]*fmd[2]),1)
        fmd4=resblock_up_no(fmsparsel3,channelnd[3],k=kersized[2], s=1,name='block3')#32x32
        fmsparsel4=sparseop(fmd4,int(1/3*fmd[3]*fmd[3]),1)
        fmd5=block_up_no(fmsparsel4,channelnd[4],k=kersized[3], s=1,name='block4')#64x64
        gx = tf.nn.tanh(conv2d(fmd5,channelnd[5],k=kersized[4], s=1,name="conv2d_5"))#64x64
        return gx

def parsingraph(z,reuse=True):
    with tf.variable_scope('gen', reuse=reuse):
        ch=fmd[0]*fmd[0]*channelnd[0]
        hf = dense(z, ch,gain=np.sqrt(2)/4,name='fully_f')
        hc=tf.reshape(hf,[-1, fmd[0], fmd[0], channelnd[0]])
        hc=tf.nn.relu(hc)
        hc=tf.nn.relu(conv2d(hc,nzatte,k=kersized[0], s=1,name="conv2d_0"))   # 4x4
        zattg=tf.reshape(hc,[BATCH_SIZE, fmd[0]*fmd[0],nzatte])
        zattg=sparsech(zattg,int(nzatte*0.1))
        fmd1=tf.zeros((BATCH_SIZE, fmd[0], fmd[0], fmd[0]*fmd[0]))
        fmsparsel1=sparseopattributeq(fmd1,zattg)
        fmsparsel1=tf.reshape(fmsparsel1,[BATCH_SIZE, fmd[0], fmd[0], nzatte*fmd[0]*fmd[0]])

        fmd2,relumaskl21,relumaskl22=block_up_cm_no(fmsparsel1,channelnd[1],k=kersized[0], s=1,name='block1') #8x8
        maskl2=sparseopidx(fmd2,int(1/4*fmd[1]*fmd[1]),1)
        fmsparsel2=sparseop(fmd2,int(1/4*fmd[1]*fmd[1]),1)
        fmd3,relumaskl31,relumaskl32=block_up_cm_no(fmsparsel2,channelnd[2],k=kersized[1], s=1,name='block2')#16x16
        maskl3=sparseopidx(fmd3,int(1/4*fmd[2]*fmd[2]),1)
        fmsparsel3=sparseop(fmd3,int(1/4*fmd[2]*fmd[2]),1)
        fmd4,relumaskl41,relumaskl42=resblock_up_cm_no(fmsparsel3,channelnd[3],k=kersized[2], s=1,name='block3')#32x32
        maskl4=sparseopidx(fmd4,int(1/3*fmd[3]*fmd[3]),1)
        fmsparsel4=sparseop(fmd4,int(1/3*fmd[3]*fmd[3]),1)
        fmd5,relumaskl51,relumaskl52=block_up_cm_no(fmsparsel4,channelnd[4],k=kersized[3], s=1,name='block4')#64x64
        return  relumaskl21,relumaskl22,relumaskl31,relumaskl32,relumaskl41,relumaskl42,relumaskl51,relumaskl52, maskl2,maskl3,maskl4


def BasisofMultiplyGlobal(z,relumaskl21,relumaskl22,relumaskl31,relumaskl32,relumaskl41,relumaskl42,relumaskl51,relumaskl52, maskl2,maskl3,maskl4,reuse=True):
  with tf.variable_scope('gen', reuse=reuse):
     ch=fmd[0]*fmd[0]*channelnd[0]
     hf = dense(z, ch,gain=np.sqrt(2)/4,name='fully_f')
     hc=tf.reshape(hf,[-1, fmd[0], fmd[0], channelnd[0]])
     hc=tf.nn.relu(hc)
     hc=tf.nn.relu(conv2d(hc,nzatte,k=kersized[0], s=1,name="conv2d_0"))   # 4x4
     zattg=tf.reshape(hc,[BATCH_SIZE, fmd[0]*fmd[0],nzatte])
     fmd1=tf.zeros((BATCH_SIZE, fmd[0], fmd[0], fmd[0]*fmd[0]))
     ######## for layer part
     basislayerglobal=[]
     #for i in range(fmd[0]):
     #   for j in range(fmd[0]):
     for k in range(20):
             zattg1=sparsechone(zattg,int(nzatte*0.1),k)
             fmsparsel1=sparseopattributeq(fmd1,zattg1)
             fmsparsel1=tf.reshape(fmsparsel1,[BATCH_SIZE, fmd[0], fmd[0], nzatte*fmd[0]*fmd[0]])

             fmd2=block_up_mask_no(fmsparsel1,channelnd[1],relumaskl21,relumaskl22,k=kersized[0], s=1,name='block1') #8x8
             fmsparsel2=fmd2*maskl2
             fmd3=block_up_mask_no(fmsparsel2,channelnd[2],relumaskl31,relumaskl32,k=kersized[1], s=1,name='block2')#16x16
             fmsparsel3=fmd3*maskl3
             fmd4=resblock_up_mask_no(fmsparsel3,channelnd[3],relumaskl41,relumaskl42,k=kersized[2], s=1,name='block3')#32x32
             fmsparsel4=fmd4*maskl4
             fmd5=block_up_mask_no(fmsparsel4,channelnd[4],relumaskl51,relumaskl52,k=kersized[3], s=1,name='block4')#64x64
             gx = tf.nn.tanh(conv2d(fmd5,channelnd[5],k=kersized[4], s=1,name="conv2d_5"))
             #gx = conv2d(fmd5,channelnd[5],k=kersized[4], s=1,name="conv2d_5")
             gx=tf.reshape(gx,[BATCH_SIZE, fmd[5], fmd[5], channelnd[5],1])
             basislayerglobal.append(gx)
     basislayerglobal=tf.concat(basislayerglobal,4)
     return basislayerglobal


def BasisofMultiplylpart(z,i,j,relumaskl21,relumaskl22,relumaskl31,relumaskl32,relumaskl41,relumaskl42,relumaskl51,relumaskl52, maskl2,maskl3,maskl4,reuse=True):
  with tf.variable_scope('gen', reuse=reuse):
     ch=fmd[0]*fmd[0]*channelnd[0]
     hf = dense(z, ch,gain=np.sqrt(2)/4,name='fully_f')
     hc=tf.reshape(hf,[-1, fmd[0], fmd[0], channelnd[0]])
     hc=tf.nn.relu(hc)
     hc=tf.nn.relu(conv2d(hc,nzatte,k=kersized[0], s=1,name="conv2d_0"))   # 4x4
     zattg=tf.reshape(hc,[BATCH_SIZE, fmd[0]*fmd[0],nzatte])
     fmd1=tf.zeros((BATCH_SIZE, fmd[0], fmd[0], fmd[0]*fmd[0]))
     ######## for layer part
     basislayerpart=[]
     #for i in range(fmd[0]):
     #   for j in range(fmd[0]):
     for k in range(20):
             zattg1=sparsechone(zattg,int(nzatte*0.1),k)
             fmsparsel1=sparsesingleopattributeq(fmd1,1,1,i,j,zattg1)
             fmsparsel1=tf.reshape(fmsparsel1,[BATCH_SIZE, fmd[0], fmd[0], nzatte*fmd[0]*fmd[0]])

             fmd2=block_up_mask_no(fmsparsel1,channelnd[1],relumaskl21,relumaskl22,k=kersized[0], s=1,name='block1') #8x8
             fmsparsel2=fmd2*maskl2
             fmd3=block_up_mask_no(fmsparsel2,channelnd[2],relumaskl31,relumaskl32,k=kersized[1], s=1,name='block2')#16x16
             fmsparsel3=fmd3*maskl3
             fmd4=resblock_up_mask_no(fmsparsel3,channelnd[3],relumaskl41,relumaskl42,k=kersized[2], s=1,name='block3')#32x32
             fmsparsel4=fmd4*maskl4
             fmd5=block_up_mask_no(fmsparsel4,channelnd[4],relumaskl51,relumaskl52,k=kersized[3], s=1,name='block4')#64x64
             gx = tf.nn.tanh(conv2d(fmd5,channelnd[5],k=kersized[4], s=1,name="conv2d_5"))
             #gx = conv2d(fmd5,channelnd[5],k=kersized[4], s=1,name="conv2d_5")
             gx=tf.reshape(gx,[BATCH_SIZE, fmd[5], fmd[5], channelnd[5],1])
             basislayerpart.append(gx)
     basislayerpart=tf.concat(basislayerpart,4)
     return basislayerpart

def BasisofMultiplyl2(z,i,x,y,relumaskl21,relumaskl22,relumaskl31,relumaskl32,relumaskl41,relumaskl42,relumaskl51,relumaskl52, maskl2,maskl3,maskl4,reuse=True):
  with tf.variable_scope('gen', reuse=reuse):
     ch=fmd[0]*fmd[0]*channelnd[0]
     hf = dense(z, ch,gain=np.sqrt(2)/4,name='fully_f')
     hc=tf.reshape(hf,[-1, fmd[0], fmd[0], channelnd[0]])
     hc=tf.nn.relu(hc)
     hc=tf.nn.relu(conv2d(hc,nzatte,k=kersized[0], s=1,name="conv2d_0"))   # 4x4
     zattg=tf.reshape(hc,[BATCH_SIZE, fmd[0]*fmd[0],nzatte])
     zattg=sparsech(zattg,int(nzatte*0.1))
     fmd1=tf.zeros((BATCH_SIZE, fmd[0], fmd[0], fmd[0]*fmd[0]))
     fmsparsel1=sparseopattributeq(fmd1,zattg)
     fmsparsel1=tf.reshape(fmsparsel1,[BATCH_SIZE, fmd[0], fmd[0], nzatte*fmd[0]*fmd[0]])
     fmd2=block_up_no(fmsparsel1,channelnd[1],k=kersized[0], s=1,name='block1') #8x8
     ######## for layer l2
     basislayerl2=[]
     #for i in range(25):
     fmsparsel2,numbasisloc=sparsesingleopxylocat(fmd2,int(1/4*fmd[1]*fmd[1]),1,i,x,y)
     fmd3=block_up_mask_no(fmsparsel2,channelnd[2],relumaskl31,relumaskl32,k=kersized[1], s=1,name='block2')#16x16
     fmsparsel3=fmd3*maskl3
     fmd4=resblock_up_mask_no(fmsparsel3,channelnd[3],relumaskl41,relumaskl42,k=kersized[2], s=1,name='block3')#32x32
     fmsparsel4=fmd4*maskl4
     fmd5=block_up_mask_no(fmsparsel4,channelnd[4],relumaskl51,relumaskl52,k=kersized[3], s=1,name='block4')#64x64
     gx = tf.nn.tanh(conv2d(fmd5,channelnd[5],k=kersized[4], s=1,name="conv2d_5"))
     #gx = conv2d(fmd5,channelnd[5],k=kersized[4], s=1,name="conv2d_5")
     gx=tf.reshape(gx,[BATCH_SIZE, fmd[5], fmd[5], channelnd[5],1])
     basislayerl2.append(gx)
     basislayerl2=tf.concat(basislayerl2,4)
     return basislayerl2,numbasisloc

def BasisofMultiplyl3(z,i,x,y,relumaskl21,relumaskl22,relumaskl31,relumaskl32,relumaskl41,relumaskl42,relumaskl51,relumaskl52, maskl2,maskl3,maskl4,reuse=True):
  with tf.variable_scope('gen', reuse=reuse):
     ch=fmd[0]*fmd[0]*channelnd[0]
     hf = dense(z, ch,gain=np.sqrt(2)/4,name='fully_f')
     hc=tf.reshape(hf,[-1, fmd[0], fmd[0], channelnd[0]])
     hc=tf.nn.relu(hc)
     hc=tf.nn.relu(conv2d(hc,nzatte,k=kersized[0], s=1,name="conv2d_0"))   # 4x4
     zattg=tf.reshape(hc,[BATCH_SIZE, fmd[0]*fmd[0],nzatte])
     zattg=sparsech(zattg,int(nzatte*0.1))
     fmd1=tf.zeros((BATCH_SIZE, fmd[0], fmd[0], fmd[0]*fmd[0]))
     fmsparsel1=sparseopattributeq(fmd1,zattg)
     fmsparsel1=tf.reshape(fmsparsel1,[BATCH_SIZE, fmd[0], fmd[0], nzatte*fmd[0]*fmd[0]])
     fmd2=block_up_no(fmsparsel1,channelnd[1],k=kersized[0], s=1,name='block1') #8x8
     fmsparsel2=fmd2*maskl2
     fmd3=block_up_mask_no(fmsparsel2,channelnd[2],relumaskl31,relumaskl32,k=kersized[1], s=1,name='block2')#16x16
     ######## for layer l3
     basislayerl3=[]
     #for i in range(25):
     fmsparsel3,numbasisloc=sparsesingleopxylocat(fmd3,int(1/4*fmd[2]*fmd[2]),1,i,x,y)
     fmd4=resblock_up_mask_no(fmsparsel3,channelnd[3],relumaskl41,relumaskl42,k=kersized[2], s=1,name='block3')#32x32
     fmsparsel4=fmd4*maskl4
     fmd5=block_up_mask_no(fmsparsel4,channelnd[4],relumaskl51,relumaskl52,k=kersized[3], s=1,name='block4')#64x64
     gx = tf.nn.tanh(conv2d(fmd5,channelnd[5],k=kersized[4], s=1,name="conv2d_5"))
     #gx = conv2d(fmd5,channelnd[5],k=kersized[4], s=1,name="conv2d_5")
     gx=tf.reshape(gx,[BATCH_SIZE, fmd[5], fmd[5], channelnd[5],1])
     basislayerl3.append(gx)
     basislayerl3=tf.concat(basislayerl3,4)
     return basislayerl3,numbasisloc

def BasisofMultiplyl4(z,i,x,y,relumaskl21,relumaskl22,relumaskl31,relumaskl32,relumaskl41,relumaskl42,relumaskl51,relumaskl52, maskl2,maskl3,maskl4,reuse=True):
  with tf.variable_scope('gen', reuse=reuse):
     ch=fmd[0]*fmd[0]*channelnd[0]
     hf = dense(z, ch,gain=np.sqrt(2)/4,name='fully_f')
     hc=tf.reshape(hf,[-1, fmd[0], fmd[0], channelnd[0]])
     hc=tf.nn.relu(hc)
     hc=tf.nn.relu(conv2d(hc,nzatte,k=kersized[0], s=1,name="conv2d_0"))   # 4x4
     zattg=tf.reshape(hc,[BATCH_SIZE, fmd[0]*fmd[0],nzatte])
     zattg=sparsech(zattg,int(nzatte*0.1))
     fmd1=tf.zeros((BATCH_SIZE, fmd[0], fmd[0], fmd[0]*fmd[0]))
     fmsparsel1=sparseopattributeq(fmd1,zattg)
     fmsparsel1=tf.reshape(fmsparsel1,[BATCH_SIZE, fmd[0], fmd[0], nzatte*fmd[0]*fmd[0]])
     fmd2=block_up_no(fmsparsel1,channelnd[1],k=kersized[0], s=1,name='block1') #8x8
     fmsparsel2=fmd2*maskl2
     fmd3=block_up_mask_no(fmsparsel2,channelnd[2],relumaskl31,relumaskl32,k=kersized[1], s=1,name='block2')#16x16
     fmsparsel3=fmd3*maskl3
     fmd4=resblock_up_mask_no(fmsparsel3,channelnd[3],relumaskl41,relumaskl42,k=kersized[2], s=1,name='block3')#32x32
     ######## for layer l4
     basislayerl4=[]
     #for i in range(25):
     fmsparsel4,numbasisloc=sparsesingleopxylocat(fmd4,int(1/3*fmd[3]*fmd[3]),1,i,x,y)
     fmd5=block_up_mask_no(fmsparsel4,channelnd[4],relumaskl51,relumaskl52,k=kersized[3], s=1,name='block4')#64x64
     gx = tf.nn.tanh(conv2d(fmd5,channelnd[5],k=kersized[4], s=1,name="conv2d_5"))
     #gx = conv2d(fmd5,channelnd[5],k=kersized[4], s=1,name="conv2d_5")
     gx=tf.reshape(gx,[BATCH_SIZE, fmd[5], fmd[5], channelnd[5],1])
     basislayerl4.append(gx)
     basislayerl4=tf.concat(basislayerl4,4)
     return basislayerl4,numbasisloc

def descriptor(x,reuse=False):
    with tf.variable_scope('des', reuse=reuse):
        feamap0 =lrelu(conv2d(x, channelndes[1],k=kersizedes[0], s=1,name="conv2d_0"))
        feamap1 = block_down(feamap0,channelndes[1],k=kersizedes[1],s=1,name='block1') # 32x32
        feamap2 = resblock_down(feamap1,channelndes[2],k=kersizedes[2],s=1,name='block2') # 16x16
        feamap3 = block_down(feamap2,channelndes[3],k=kersizedes[3],s=1,name='block3') # 8x8
        feamap4 = block_down(feamap3,channelndes[4],k=kersizedes[4],s=1,name='block4') # 4x4
        feamap4 = minibatch_stddev_layer(feamap4,10)
        feamap5 = lrelu(conv2d(feamap4, channelndes[5],k=kersizedes[4], s=1,name="conv2d_5")) # 4x4
        #feamap5 = lrelu(conv2d(feamap5, channelndes[5],k=4, s=1,padding='VALID',name="conv2d_6")) # 1x1
        feamap5 =tf.reshape(feamap5 ,[BATCH_SIZE,channelndes[5]*fmdes[5]*fmdes[5]])
        out = dense(feamap5, 1,gain=1,name='fully_f')
        return out


"""
Train EBM
"""
real_score=descriptor(Y)
energy_real=-tf.reduce_mean(real_score)
#Gys = P(tf.reshape(z_s,[-1,1,1,z_dim]))
Gys = P(z_s)
syn_score=descriptor(tf.stop_gradient(Gys),reuse=True)
energy_syn=-tf.reduce_mean(syn_score)
lossd=tf.subtract(energy_real,energy_syn)
#lossds=discriminator_loss(real_score,syn_score)
#####################gp#####################
lgp=10
eps = tf.random_uniform(shape=[tf.shape(Y)[0], 1, 1, 1], minval=0., maxval=1.)
Y_hat = eps*Y + (1.-eps)* tf.stop_gradient(Gys)
inter_score = descriptor(Y_hat, reuse=True)
inter_grad = tf.gradients(inter_score, [Y_hat])[0]
#inter_grad_norm = tf.norm(tf.layers.flatten(inter_grad), axis=1)
inter_grad_norm  = tf.sqrt(tf.reduce_sum(tf.square(inter_grad), reduction_indices=[1, 2, 3]))
GP = lgp * tf.reduce_mean(tf.square(inter_grad_norm - 1.))
##########
leps=1e-3
DReg=leps*tf.reduce_mean(tf.square(real_score))
##########
#Theta_Dw=[var for var in tf.trainable_variables() if (var.name.startswith('des')&var.name.endswith('weight:0'))]
#DgradNorm=getGradNorm(lossd,Theta_Dw)
Theta_D=[var for var in tf.trainable_variables() if var.name.startswith('des')]
update_opsD = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='des')
with tf.control_dependencies(update_opsD):
    optimizerD=tf.train.AdamOptimizer(lr_d, beta1=0.0,beta2=0.99).minimize(lossd+GP+DReg,var_list=Theta_D)

"""
Train I and G
"""
beta=0.1
za = Q(Y)
Gyr = P(za,reuse=True)
recons_loss = (tf.nn.l2_loss(Gyr - Y))/(BATCH_SIZE)*2

fgz=descriptor(Gys,reuse=True)
lossg=-tf.reduce_mean(fgz)
loss2klgen=beta*recons_loss+lossg
Alpha_IG=[var for var in tf.trainable_variables() if (var.name.startswith('infer')|var.name.startswith('gen'))]

optimizerIG=tf.train.AdamOptimizer(lr, beta1=0.0,beta2=0.99).minimize(loss2klgen, var_list=Alpha_IG)

##############################
relumaskl21,relumaskl22,relumaskl31,relumaskl32,relumaskl41,relumaskl42,relumaskl51,relumaskl52, maskl2,maskl3,maskl4 = parsingraph(z_s)
basislayerglobal = BasisofMultiplyGlobal(z_s,relumaskl21,relumaskl22,relumaskl31,relumaskl32,relumaskl41,relumaskl42,relumaskl51,relumaskl52, maskl2,maskl3,maskl4)
basislayerpart = BasisofMultiplylpart(z_s,ip,jp,relumaskl21,relumaskl22,relumaskl31,relumaskl32,relumaskl41,relumaskl42,relumaskl51,relumaskl52, maskl2,maskl3,maskl4)
basislayerl2,numbasisloc2=BasisofMultiplyl2(z_s,ip,xp,yp,relumaskl21,relumaskl22,relumaskl31,relumaskl32,relumaskl41,relumaskl42,relumaskl51,relumaskl52, maskl2,maskl3,maskl4)
basislayerl3,numbasisloc3=BasisofMultiplyl3(z_s,ip,xp,yp,relumaskl21,relumaskl22,relumaskl31,relumaskl32,relumaskl41,relumaskl42,relumaskl51,relumaskl52, maskl2,maskl3,maskl4)
basislayerl4,numbasisloc4=BasisofMultiplyl4(z_s,ip,xp,yp,relumaskl21,relumaskl22,relumaskl31,relumaskl32,relumaskl41,relumaskl42,relumaskl51,relumaskl52, maskl2,maskl3,maskl4)
##############################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    if not os.path.exists(figpath):
            os.makedirs(figpath)
    if not os.path.exists(parampath):
            os.makedirs(parampath)
    saver = tf.train.Saver()
    saver.restore(sess, modelpath)

    #tf.get_default_graph().finalize()
    lr_dp=7e-4
    lr_p=7e-4
    f=open(logfile,"a")
    for iepoch in range(1):

        for i in range(0,1):
            #Ypori=train_set[i * BATCH_SIZE:min(len(train_set), (i + 1) * BATCH_SIZE)]
            #Yp=localCN_v2(Ypori)
            #Yp=Ypori
            z_sp = np.random.uniform(-1,1,(BATCH_SIZE, z_dim))

            if 0:
                Gyspaf = sess.run(Gys,feed_dict={z_s:z_sp})
                fig = plot(Gyspaf,int(np.sqrt(BATCH_SIZE)),int(np.sqrt(BATCH_SIZE)),CHANNELS,IMG_HEIGHT, IMG_WIDTH)
                ttt='batch{}samplingn.png'.format(str(i).zfill(3))
                plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
                plt.close(fig)
            if 0:
                ttt1='sample-img'
                figpath1=figpath+ttt1+'/'
                if not os.path.exists(figpath1):
                    os.makedirs(figpath1)
                for inb in range(BATCH_SIZE):
                    ttt2='page{}inb{}.png'.format(str(i).zfill(3),str(inb).zfill(3))
                    mysaveimgori(os.path.join(figpath1, ttt2), Gyspaf[inb],CHANNELS,fmd[5],fmd[5])

############################################################
    if 1:
        #idp=53
        idp=46
        nbasis=20
        basislayerglobalp=sess.run(basislayerglobal,feed_dict={z_s: z_sp})

        #fig = plot(np.transpose(np.tanh(basislayerglobalp[idp]),(3,0,1,2)),int(4),5,CHANNELS,fmd[5],fmd[5])
        fig = plotnorm1(np.transpose(np.tanh(basislayerglobalp[idp]),(3,0,1,2)),int(4),5,CHANNELS,fmd[5],fmd[5])
        ttt='eachdimreconsbasis1-globalid{}.png'.format(str(idp))
        plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
        plt.close(fig)
        ttt1='globalbasis1-id{}-vglobal'.format(str(idp).zfill(3))
        figpath1=figpath+ttt1+'/'
        if not os.path.exists(figpath1):
            os.makedirs(figpath1)
        for inb in range(nbasis):
            ttt2='inb{}.png'.format(str(inb).zfill(3))
            mysaveimg(os.path.join(figpath1, ttt2),np.tanh(basislayerglobalp[idp,:,:,:,inb]),CHANNELS,fmd[5],fmd[5])
    if 1:
            idp=46
            #idp=73
            basislayerpartp=[]
            nbasis=80
            for ipp in range(4):
                 for jpp in range(4):
                     basislayerpartp1 = sess.run(basislayerpart,feed_dict={z_s: z_sp,ip:ipp,jp:jpp})
                     basislayerpartp.append(basislayerpartp1)
            basislayerpartp=np.concatenate(basislayerpartp,4)
            for pic in range(4):
                 basislayerpartlocal= basislayerpartp[idp,:,:,:,pic*8*10:pic*8*10+8*10]
                 basislayerpartlocal=np.transpose(basislayerpartlocal,(3,0,1,2))
                 #fig = plot1(np.tanh(basislayerpartlocal),int(8),10,CHANNELS,fmd[5],fmd[5])
                 fig = plotnorm1(np.tanh(basislayerpartlocal),int(8),10,CHANNELS,fmd[5],fmd[5])
                 ttt='eachdimreconsbasis1-partlayerid{}pid{}.png'.format(str(idp),str(pic))
                 plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
                 plt.close(fig)
                 ttt1='eachcropbasis1-id{}-vtoplayer'.format(str(idp).zfill(3))
                 figpath1=figpath+ttt1+'/'
                 if not os.path.exists(figpath1):
                    os.makedirs(figpath1)
                 for inb in range(nbasis):
                     ttt2='page{}inb{}.png'.format(str(pic).zfill(3),str(inb).zfill(3))
                     mysaveimg(os.path.join(figpath1, ttt2), basislayerpartlocal[inb],CHANNELS,fmd[5],fmd[5])


            #########
    if 1:
            basislayerl2p=[]
            for ipp in range(64):
                basislayerl2p1 = sess.run(basislayerl2,feed_dict={z_s: z_sp,ip:ipp})
                basislayerl2p.append(basislayerl2p1)
            basislayerl2p=np.concatenate(basislayerl2p,4)
            basislayerl2local = basislayerl2p[idp,:,:,:,:]
            basislayerl2local= np.transpose(basislayerl2local,(3,0,1,2))
            nbasis_mean=64
            #fig = plot(np.tanh(basislayerl2local[:,:,:,:]),int(np.sqrt(nbasis_mean)),int(np.sqrt(nbasis_mean)),CHANNELS,fmd[5],fmd[5])
            fig = plotnorm(basislayerl2local[:,:,:,:],int(np.sqrt(nbasis_mean)),int(np.sqrt(nbasis_mean)),CHANNELS,fmd[5],fmd[5])

            ttt='eachdimreconsbasis1-v2layerid{}.png'.format(str(idp).zfill(3))
            plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
            plt.close(fig)
            ############

            basislayerl3p=[]
            for ipp in range(64):
                basislayerl3p1 = sess.run(basislayerl3,feed_dict={z_s: z_sp,ip:ipp})
                basislayerl3p.append(basislayerl3p1)
            basislayerl3p=np.concatenate(basislayerl3p,4)
            basislayerl3local = basislayerl3p[idp,:,:,:,:]
            basislayerl3local= np.transpose(basislayerl3local,(3,0,1,2))
            nbasis_mean=64
            #fig = plot(np.tanh(basislayerl3local[:,:,:,:]),int(np.sqrt(nbasis_mean)),int(np.sqrt(nbasis_mean)),CHANNELS,fmd[5],fmd[5])
            fig = plotnorm(basislayerl3local[:,:,:,:],int(np.sqrt(nbasis_mean)),int(np.sqrt(nbasis_mean)),CHANNELS,fmd[5],fmd[5])

            ttt='eachdimreconsbasis1-v3layerid{}.png'.format(str(idp).zfill(3))
            plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
            plt.close(fig)
            ##############
            basislayerl4p=[]
            for ipp in range(64):
                basislayerl4p1 = sess.run(basislayerl4,feed_dict={z_s: z_sp,ip:ipp})
                basislayerl4p.append(basislayerl4p1)
            basislayerl4p=np.concatenate(basislayerl4p,4)
            basislayerl4local = basislayerl4p[idp,:,:,:,:]
            basislayerl4local= np.transpose(basislayerl4local,(3,0,1,2))
            nbasis_mean=64
            #fig = plot(np.tanh(basislayerl4local[:,:,:,:]),int(np.sqrt(nbasis_mean)),int(np.sqrt(nbasis_mean)),CHANNELS,fmd[5],fmd[5])
            fig = plotnorm(basislayerl4local[:,:,:,:],int(np.sqrt(nbasis_mean)),int(np.sqrt(nbasis_mean)),CHANNELS,fmd[5],fmd[5])
            ttt='eachdimreconsbasis1-v4layerid{}.png'.format(str(idp).zfill(3))
            plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
            plt.close(fig)
###########################################################
    if 1:
            #crop subbasis
        idp=73
        nbasis=400
        for ipage in range(0,4):
            basislayerl4p=[]
            for ipp in range(ipage*nbasis,(ipage+1)*nbasis):
                basislayerl4p1 = sess.run(basislayerl4,feed_dict={z_s: z_sp,ip:ipp})
                basislayerl4p.append(basislayerl4p1)
            basislayerl4p=np.concatenate(basislayerl4p,4)
            basislayerl4local = basislayerl4p[idp,:,:,:,:]
            basislayerl4local= np.transpose(basislayerl4local,(3,0,1,2))
            basislayerl4localcrop=[]
            xleng=np.zeros((1,nbasis));yleng=np.zeros((1,nbasis));
            for ipp in range(nbasis):
                singlebasis=basislayerl4local[ipp]
                singlebasism=np.mean(singlebasis,2)
                a,b= np.where( np.abs(singlebasism) > 1e-4 )
                xmin=a.min();xmax=a.max();ymin=b.min();ymax=b.max()
                xleng[0,ipp]=xmax-xmin;yleng[0,ipp]=ymax-ymin;
            xl=np.median(xleng);yl=np.median(yleng);
            for ipp in range(nbasis):
                singlebasis=basislayerl4local[ipp]
                singlebasism=np.mean(singlebasis,2)
                a,b= np.where( np.abs(singlebasism) > 1e-4 )
                xmin=a.min();xmax=a.max();ymin=b.min();ymax=b.max()
                singlebasiscrop=singlebasis[xmin:xmax+1,ymin:ymax+1,:]
                if (xmax-xmin!=xl)|(ymax-ymin!=yl):
                    singlebasiscrop= resize(singlebasiscrop, (int(xl+1), int(yl+1)),anti_aliasing=True)
                singlebasiscrop=np.reshape(singlebasiscrop,[1,int(xl+1),int(yl+1),CHANNELS])
                basislayerl4localcrop.append(singlebasiscrop)
            basislayerl4localcrop1=np.concatenate(basislayerl4localcrop,0)
            nbasis_mean=nbasis
            #fig = plotnorm(np.tanh(basislayerl4localcrop1[:,:,:,:]),int(np.sqrt(nbasis_mean)),int(np.sqrt(nbasis_mean)),CHANNELS,int(xl+1),int(yl+1))
            fig = plotnorm(basislayerl4localcrop1[:,:,:,:],int(np.sqrt(nbasis_mean)),int(np.sqrt(nbasis_mean)),CHANNELS,int(xl+1),int(yl+1))
            ttt='eachdimcropbasis1-v4layerid{}page{}.png'.format(str(idp).zfill(3),str(ipage).zfill(3))
            plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
            plt.close(fig)
            ttt1='eachcropbasis1-id{}-v4layer'.format(str(idp).zfill(3))
            figpath1=figpath+ttt1+'/'
            if not os.path.exists(figpath1):
                    os.makedirs(figpath1)
            for inb in range(nbasis):
                ttt2='page{}inb{}.png'.format(str(ipage).zfill(3),str(inb).zfill(3))
                mysaveimg(os.path.join(figpath1, ttt2), basislayerl4localcrop1[inb],CHANNELS,int(xl+1),int(yl+1))
    if 1:
            #crop subbasis
        idp=73
        nbasis=400
        for ipage in range(0,5):
            basislayerl3p=[]
            for ipp in range(ipage*nbasis,(ipage+1)*nbasis):
                basislayerl3p1 = sess.run(basislayerl3,feed_dict={z_s: z_sp,ip:ipp})
                basislayerl3p.append(basislayerl3p1)
            basislayerl3p=np.concatenate(basislayerl3p,4)
            basislayerl3local = basislayerl3p[idp,:,:,:,:]
            basislayerl3local= np.transpose(basislayerl3local,(3,0,1,2))
            basislayerl3localcrop=[]
            xleng=np.zeros((1,nbasis));yleng=np.zeros((1,nbasis));
            for ipp in range(nbasis):
                singlebasis=basislayerl3local[ipp]
                singlebasism=np.mean(singlebasis,2)
                a,b= np.where( np.abs(singlebasism) > 1e-4 )
                xmin=a.min();xmax=a.max();ymin=b.min();ymax=b.max()
                xleng[0,ipp]=xmax-xmin;yleng[0,ipp]=ymax-ymin;
            xl=np.median(xleng);yl=np.median(yleng);
            for ipp in range(nbasis):
                singlebasis=basislayerl3local[ipp]
                singlebasism=np.mean(singlebasis,2)
                a,b= np.where( np.abs(singlebasism) > 1e-4 )
                xmin=a.min();xmax=a.max();ymin=b.min();ymax=b.max()
                singlebasiscrop=singlebasis[xmin:xmax+1,ymin:ymax+1,:]
                if (xmax-xmin!=xl)|(ymax-ymin!=yl):
                    singlebasiscrop= resize(singlebasiscrop, (int(xl+1), int(yl+1)),anti_aliasing=True)
                singlebasiscrop=np.reshape(singlebasiscrop,[1,int(xl+1),int(yl+1),CHANNELS])
                basislayerl3localcrop.append(singlebasiscrop)
            basislayerl3localcrop1=np.concatenate(basislayerl3localcrop,0)
            nbasis_mean=nbasis
            #fig = plot(np.tanh(basislayerl3localcrop1[:,:,:,:]),int(np.sqrt(nbasis_mean)),int(np.sqrt(nbasis_mean)),CHANNELS,int(xl+1),int(yl+1))
            fig = plotnorm(basislayerl3localcrop1[:,:,:,:],int(np.sqrt(nbasis_mean)),int(np.sqrt(nbasis_mean)),CHANNELS,int(xl+1),int(yl+1))
            ttt='eachdimcropbasis1-v3layerid{}page{}.png'.format(str(idp).zfill(3),str(ipage).zfill(3))
            plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
            plt.close(fig)
            ttt1='eachcropbasis1-id{}-v3layer'.format(str(idp).zfill(3))
            figpath1=figpath+ttt1+'/'
            if not os.path.exists(figpath1):
                    os.makedirs(figpath1)
            for inb in range(nbasis):
                ttt2='page{}inb{}.png'.format(str(ipage).zfill(3),str(inb).zfill(3))
                mysaveimg(os.path.join(figpath1, ttt2), basislayerl3localcrop1[inb],CHANNELS,int(xl+1),int(yl+1))
    if 1:
            #crop subbasis
        idp=53

        for xpp in range(fmd[1]):
            for ypp in range(fmd[1]):
                numbasisloc2p = sess.run(numbasisloc2,feed_dict={z_s: z_sp,ip:0,xp:xpp,yp:ypp})
                nbasis=int(numbasisloc2p[idp])
                basislayerl2p=[]
                for ipp in range(nbasis):
                    basislayerl2p1 = sess.run(basislayerl2,feed_dict={z_s: z_sp,ip:ipp,xp:xpp,yp:ypp})
                    basislayerl2p.append(basislayerl2p1)
                basislayerl2p=np.concatenate(basislayerl2p,4)
                basislayerl2local = basislayerl2p[idp,:,:,:,:]
                basislayerl2local= np.transpose(basislayerl2local,(3,0,1,2))
                basislayerl2localcrop=[]
                xleng=np.zeros((1,nbasis));yleng=np.zeros((1,nbasis));
                for ipp in range(nbasis):
                    singlebasis=basislayerl2local[ipp]
                    singlebasism=np.mean(singlebasis,2)
                    a,b= np.where( np.abs(singlebasism) > 1e-4 )
                    if (a.size == 0) | (b.size==0):
                        continue
                    xmin=a.min();xmax=a.max();ymin=b.min();ymax=b.max()
                    xleng[0,ipp]=xmax-xmin;yleng[0,ipp]=ymax-ymin;
                xl=np.median(xleng);yl=np.median(yleng);
                for ipp in range(nbasis):
                    singlebasis=basislayerl2local[ipp]
                    singlebasism=np.mean(singlebasis,2)
                    a,b= np.where( np.abs(singlebasism) > 1e-4 )
                    if (a.size == 0) | (b.size==0):
                        continue
                    xmin=a.min();xmax=a.max();ymin=b.min();ymax=b.max()
                    singlebasiscrop=singlebasis[xmin:xmax+1,ymin:ymax+1,:]
                    if (xmax-xmin!=xl)|(ymax-ymin!=yl):
                        singlebasiscrop= resize(singlebasiscrop, (int(xl+1), int(yl+1)),anti_aliasing=True)
                    singlebasiscrop=np.reshape(singlebasiscrop,[1,int(xl+1),int(yl+1),CHANNELS])
                    basislayerl2localcrop.append(singlebasiscrop)
                basislayerl2localcrop1=np.concatenate(basislayerl2localcrop,0)
                nbasis_mean=nbasis
                if 0:
                    fig = plotnorm(basislayerl2localcrop1[:,:,:,:],int(np.sqrt(nbasis_mean)),int(np.sqrt(nbasis_mean)),CHANNELS,int(xl+1),int(yl+1))
                    ttt='eachdimcropbasis1-v2layerid{}locx{}y{}.png'.format(str(idp).zfill(3),str(xpp).zfill(3),str(ypp).zfill(3))
                    plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
                    plt.close(fig)
                if 1:
                    ttt1='eachcropbasis1-id{}locx{}y{}-v2layer'.format(str(idp).zfill(3),str(xpp).zfill(3),str(ypp).zfill(3))
                    figpath1=figpath+ttt1+'/'
                    if not os.path.exists(figpath1):
                        os.makedirs(figpath1)
                    for inb in range(np.minimum(nbasis,basislayerl2localcrop1.shape[0])):
                        ttt2='inb{}.png'.format(str(inb).zfill(3))
                        mysaveimg(os.path.join(figpath1, ttt2), basislayerl2localcrop1[inb],CHANNELS,int(xl+1),int(yl+1))
    if 1:
            #crop subbasis
        idp=53

        for xpp in range(fmd[2]):
            for ypp in range(fmd[2]):
                numbasisloc3p = sess.run(numbasisloc3,feed_dict={z_s: z_sp,ip:0,xp:xpp,yp:ypp})
                nbasis=int(numbasisloc3p[idp])
                basislayerl3p=[]
                for ipp in range(nbasis):
                    basislayerl3p1 = sess.run(basislayerl3,feed_dict={z_s: z_sp,ip:ipp,xp:xpp,yp:ypp})
                    basislayerl3p.append(basislayerl3p1)
                basislayerl3p=np.concatenate(basislayerl3p,4)
                basislayerl3local = basislayerl3p[idp,:,:,:,:]
                basislayerl3local= np.transpose(basislayerl3local,(3,0,1,2))
                basislayerl3localcrop=[]
                xleng=np.zeros((1,nbasis));yleng=np.zeros((1,nbasis));
                for ipp in range(nbasis):
                    singlebasis=basislayerl3local[ipp]
                    singlebasism=np.mean(singlebasis,2)
                    a,b= np.where( np.abs(singlebasism) > 1e-4 )
                    if (a.size == 0) | (b.size==0):
                        continue
                    xmin=a.min();xmax=a.max();ymin=b.min();ymax=b.max()
                    xleng[0,ipp]=xmax-xmin;yleng[0,ipp]=ymax-ymin;
                xl=np.median(xleng);yl=np.median(yleng);
                for ipp in range(nbasis):
                    singlebasis=basislayerl3local[ipp]
                    singlebasism=np.mean(singlebasis,2)
                    a,b= np.where( np.abs(singlebasism) > 1e-4 )
                    if (a.size == 0) | (b.size==0):
                        continue
                    xmin=a.min();xmax=a.max();ymin=b.min();ymax=b.max()
                    singlebasiscrop=singlebasis[xmin:xmax+1,ymin:ymax+1,:]
                    if (xmax-xmin!=xl)|(ymax-ymin!=yl):
                        singlebasiscrop= resize(singlebasiscrop, (int(xl+1), int(yl+1)),anti_aliasing=True)
                    singlebasiscrop=np.reshape(singlebasiscrop,[1,int(xl+1),int(yl+1),CHANNELS])
                    basislayerl3localcrop.append(singlebasiscrop)
                basislayerl3localcrop1=np.concatenate(basislayerl3localcrop,0)
                nbasis_mean=nbasis
                if 1:
                    fig = plotnorm(basislayerl3localcrop1[:,:,:,:],int(np.sqrt(nbasis_mean)),int(np.sqrt(nbasis_mean)),CHANNELS,int(xl+1),int(yl+1))
                    ttt='eachdimcropbasis1-v3layerid{}locx{}y{}.png'.format(str(idp).zfill(3),str(xpp).zfill(3),str(ypp).zfill(3))
                    plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
                    plt.close(fig)
                if 1:
                    ttt1='eachcropbasis1-id{}locx{}y{}-v3layer'.format(str(idp).zfill(3),str(xpp).zfill(3),str(ypp).zfill(3))
                    figpath1=figpath+ttt1+'/'
                    if not os.path.exists(figpath1):
                        os.makedirs(figpath1)
                    for inb in range(np.minimum(nbasis,basislayerl3localcrop1.shape[0])):
                        ttt2='inb{}.png'.format(str(inb).zfill(3))
                        mysaveimg(os.path.join(figpath1, ttt2), basislayerl3localcrop1[inb],CHANNELS,int(xl+1),int(yl+1))

    if 1:
            #crop subbasis
        idp=53

        for xpp in tqdm(range(fmd[3]), total=fmd[3]):
            for ypp in range(fmd[3]):
                numbasisloc4p = sess.run(numbasisloc4,feed_dict={z_s: z_sp,ip:0,xp:xpp,yp:ypp})
                nbasis=int(numbasisloc4p[idp])
                basislayerl4p=[]
                for ipp in range(nbasis):
                    basislayerl4p1 = sess.run(basislayerl4,feed_dict={z_s: z_sp,ip:ipp,xp:xpp,yp:ypp})
                    basislayerl4p.append(basislayerl4p1)
                basislayerl4p=np.concatenate(basislayerl4p,4)
                basislayerl4local = basislayerl4p[idp,:,:,:,:]
                basislayerl4local= np.transpose(basislayerl4local,(3,0,1,2))
                basislayerl4localcrop=[]
                xleng=np.zeros((1,nbasis));yleng=np.zeros((1,nbasis));
                for ipp in range(nbasis):
                    singlebasis=basislayerl4local[ipp]
                    singlebasism=np.mean(singlebasis,2)
                    a,b= np.where( np.abs(singlebasism) > 1e-4 )
                    if (a.size == 0) | (b.size==0):
                        continue
                    xmin=a.min();xmax=a.max();ymin=b.min();ymax=b.max()
                    xleng[0,ipp]=xmax-xmin;yleng[0,ipp]=ymax-ymin;
                xl=np.median(xleng);yl=np.median(yleng);
                for ipp in range(nbasis):
                    singlebasis=basislayerl4local[ipp]
                    singlebasism=np.mean(singlebasis,2)
                    a,b= np.where( np.abs(singlebasism) > 1e-4 )
                    if (a.size == 0) | (b.size==0):
                        continue
                    xmin=a.min();xmax=a.max();ymin=b.min();ymax=b.max()
                    singlebasiscrop=singlebasis[xmin:xmax+1,ymin:ymax+1,:]
                    if (xmax-xmin!=xl)|(ymax-ymin!=yl):
                        singlebasiscrop= resize(singlebasiscrop, (int(xl+1), int(yl+1)),anti_aliasing=True)
                    singlebasiscrop=np.reshape(singlebasiscrop,[1,int(xl+1),int(yl+1),CHANNELS])
                    basislayerl4localcrop.append(singlebasiscrop)
                basislayerl4localcrop1=np.concatenate(basislayerl4localcrop,0)
                nbasis_mean=nbasis
                if 1:
                    fig = plotnorm(basislayerl4localcrop1[:,:,:,:],int(np.sqrt(nbasis_mean)),int(np.sqrt(nbasis_mean)),CHANNELS,int(xl+1),int(yl+1))
                    ttt='eachdimcropbasis1-v4layerid{}locx{}y{}.png'.format(str(idp).zfill(3),str(xpp).zfill(3),str(ypp).zfill(3))
                    plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
                    plt.close(fig)
                if 1:
                    ttt1='eachcropbasis1-id{}locx{}y{}-v4layer'.format(str(idp).zfill(3),str(xpp).zfill(3),str(ypp).zfill(3))
                    figpath1=figpath+ttt1+'/'
                    if not os.path.exists(figpath1):
                        os.makedirs(figpath1)
                    for inb in range(np.minimum(nbasis,basislayerl4localcrop1.shape[0])):
                        ttt2='inb{}.png'.format(str(inb).zfill(3))
                        mysaveimg(os.path.join(figpath1, ttt2), basislayerl4localcrop1[inb],CHANNELS,int(xl+1),int(yl+1))


