#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 01:04:07 2019

@author: andy
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:26:57 2019

@author: andy
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib.pyplot as plt
from utils.myplot import plot
from utils.myplot import plotfilter1
from utils.operation2 import block_down
from utils.operation2 import resblock_down
from utils.operation2 import block_up_no
from utils.operation2 import resblock_up_no
from utils.operation2 import conv2d
#from utils.operation2 import deconv2d
from utils.operation2 import lrelu
from utils.operation2 import dense
from utils.operation2 import minibatch_stddev_layer
from utils.operation2 import sparsech
from utils.operation2 import sparseop
from utils.operation2 import sparseopattributeq
from utils.data_io import LoadDataSet
from utils.imgpross import plothist
import time
from tqdm import tqdm

dir = '/user_data/tianqinl'
#datapath='/home/andy/Downloads/lsun-master/Bedroom100K'
#datapath='/home/andy/Documents/car/resized'
#datapath='./Dataset/stlimgall'
special='SparsegeneratorAEadv10k64ResnonormandzCelebA'
dataname='deepfashion_only'
figpath=dir+'/Results/'+dataname+'/'+'figures'+special+'/'
logpath=dir+'/Results/'+dataname+'/'
parampath=dir+'/Results/'+dataname+'/'+'params'+special+'/'
modelpath=dir+'/Results/'+dataname+'/'+'params'+special+'/'+'model.ckpt'
logfile=logpath+'log'+dataname+special

Nall=10000
BATCH_SIZE=50
IMG_HEIGHT = 64 # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 64 # CHANGE HERE, the image width to be resized to
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale
Nbatch= int(np.floor(Nall/BATCH_SIZE))
Nall=int(Nbatch*BATCH_SIZE)
Nepoch=2000

deepfashion_data_folder = "/user_data/tianqinl/"
#deepfashion_index_file = "deepfashion50k.txt"
deepfashion_index_file = "listoffiles.txt"
deepfashion_index_file = pd.read_csv(os.path.join(deepfashion_data_folder, deepfashion_index_file), header=None)
deepfashion_pathlist = [os.path.join(deepfashion_data_folder, i) for i in list(deepfashion_index_file.iloc[:,0])]
datapathlist = deepfashion_pathlist
print(datapathlist)
train_set = LoadDataSet(datapathlist, im_size=[IMG_HEIGHT,IMG_WIDTH,CHANNELS],mode=2)
tf.reset_default_graph()
z_dim = 100

lr=tf.placeholder(dtype=tf.float32,shape=())
lr_d=tf.placeholder(dtype=tf.float32,shape=())
z= tf.placeholder(shape=[BATCH_SIZE, z_dim], dtype=tf.float32, name='z')
z_s= tf.placeholder(shape=[BATCH_SIZE, z_dim], dtype=tf.float32, name='zs')
Y = tf.placeholder(tf.float32, shape = [None, IMG_HEIGHT,IMG_WIDTH,CHANNELS])

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
        fmd1=tf.zeros((BATCH_SIZE, fmd[0], fmd[0], fmd[0]*fmd[0]))  #?????????????????????????? ZEROS??
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
za = Q(Y) # encoder
Gyr = P(za,reuse=True) # generate
recons_loss = (tf.nn.l2_loss(Gyr - Y))/(BATCH_SIZE)*2

fgz=descriptor(Gys,reuse=True)
lossg=-tf.reduce_mean(fgz)
loss2klgen=beta*recons_loss+lossg
Alpha_IG=[var for var in tf.trainable_variables() if (var.name.startswith('infer')|var.name.startswith('gen'))]

optimizerIG=tf.train.AdamOptimizer(lr, beta1=0.0,beta2=0.99).minimize(loss2klgen, var_list=Alpha_IG)
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
    if 1:
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        adam_vars = [g for g in g_list if 'beta1_power' in g.name]
        adam_vars += [g for g in g_list if 'Adam' in g.name]
        adam_vars += [g for g in g_list if 'beta2_power' in g.name]
        var_list += adam_vars
        print('var_list:',var_list)
    saver = tf.train.Saver(max_to_keep=50)
    tf.get_default_graph().finalize()
    lr_dp=7e-4
    lr_p=7e-4
    energy_realall=np.zeros((Nepoch+1),np.float32)
    energy_synall=np.zeros((Nepoch+1),np.float32)
    lossd_all=np.zeros((Nepoch+1),np.float32)
    lossg1_all=np.zeros((Nepoch+1),np.float32)
    lossg2_all=np.zeros((Nepoch+1),np.float32)
    lossr_all=np.zeros((Nepoch+1),np.float32)

    f=open(logfile,"a")
    for iepoch in range(Nepoch+1):
        train_set_rp=np.random.permutation(train_set)


        Recons_loss=0.
        Des_contraloss=0.
        Energy_real=0.
        Energy_syn=0.
        Lossd=0.
        Lossg1=0.
        Lossg2=0.
        Gp=0.
        Dreg=0.
        start_time = time.time()
        for i in tqdm(range(Nbatch), total=Nbatch):
            Ypori=train_set_rp[i * BATCH_SIZE:min(len(train_set), (i + 1) * BATCH_SIZE)]
            #Yp=localCN_v2(Ypori)
            Yp=Ypori
            z_sp = np.random.uniform(-1,1,(BATCH_SIZE, z_dim))
            #opdesrun=[optimizerD,lossd,GP,DReg,energy_real,energy_syn,lossg,DgradNorm]
            opdesrun=[optimizerD,lossd,GP,DReg,energy_real,energy_syn,lossg]
            #_,lossdp,GPp,DRegp,energy_realp,energy_synp,lossgp,DgradNormp=sess.run(opdesrun,feed_dict={Y: Yp,z_s:z_sp,lr_d:lr_dp})
            _,lossdp,GPp,DRegp,energy_realp,energy_synp,lossgp1=sess.run(opdesrun,feed_dict={Y: Yp,z_s:z_sp,lr_d:lr_dp})
            #_,lossgp,GgradNormp=sess.run([optimizerG,lossg,GgradNorm],feed_dict={Y: Yp,lr:lr_p,z_s:z_sp})
            _,lossgp2,recons_lossp=sess.run([optimizerIG,lossg,recons_loss],feed_dict={Y: Yp,lr:lr_p,z_s:z_sp})


            Gypaf,Gyspaf = sess.run([Gyr,Gys],feed_dict={Y: Yp,z_s:z_sp})
            Recons_loss=Recons_loss+recons_lossp/Nbatch
            Lossd=Lossd+lossdp/Nbatch
            Gp=Gp+GPp/Nbatch
            Dreg=Dreg+DRegp/Nbatch
            Energy_real=Energy_real+energy_realp/Nbatch
            Energy_syn=Energy_syn+energy_synp/Nbatch
            Lossg1=Lossg1+lossgp1/Nbatch
            Lossg2=Lossg2+lossgp2/Nbatch

        end_time = time.time()
        print('Epoch #{:d},lossd:{:.4f},Gp:{:.4f},Dreg:{:.4f},lossg1:{:.4f},time: {:.2f}s'.format(iepoch, Lossd,Gp,Dreg,Lossg1,end_time - start_time))
        print('Energy_real: {:.4f}, Energy_syn:{:.4f},Recons_loss:{:.4f},lossg2:{:.4f}'.format(Energy_real,Energy_syn,beta*Recons_loss,Lossg2))
        print('Epoch #{:d},lossd:{:.4f},Gp:{:.4f},Dreg:{:.4f},lossg1:{:.4f},time: {:.2f}s'.format(iepoch, Lossd,Gp,Dreg,Lossg1,end_time - start_time),file=f)
        print('Energy_real: {:.4f}, Energy_syn:{:.4f},Recons_loss:{:.4f},lossg2:{:.4f}'.format(Energy_real,Energy_syn,beta*Recons_loss,Lossg2),file=f)
        energy_realall[iepoch]=Energy_real
        energy_synall[iepoch]=Energy_syn
        lossd_all[iepoch]=Lossd
        lossg1_all[iepoch]=Lossg1
        lossg2_all[iepoch]=Lossg2
        lossr_all[iepoch]=beta*Recons_loss

        if iepoch % 1 == 0:
              ###plot and save sample images################
            fig = plot(Yp,int(np.sqrt(BATCH_SIZE)),int(np.sqrt(BATCH_SIZE)),CHANNELS,IMG_HEIGHT, IMG_WIDTH)
            ttt='epoch{}original.png'.format(str(iepoch).zfill(3))
            plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
            plt.close(fig)
            fig = plot(Gypaf,int(np.sqrt(BATCH_SIZE)),int(np.sqrt(BATCH_SIZE)),CHANNELS,IMG_HEIGHT, IMG_WIDTH)
            ttt='epoch{}reconsafter.png'.format(str(iepoch).zfill(3))
            plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
            plt.close(fig)
            fig = plot(Gyspaf,int(np.sqrt(BATCH_SIZE)),int(np.sqrt(BATCH_SIZE)),CHANNELS,IMG_HEIGHT, IMG_WIDTH)
            ttt='epoch{}samplingVADafter.png'.format(str(iepoch).zfill(3))
            plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure()
            plt.plot(energy_realall[0:iepoch+1],'r-')
            plt.plot(energy_synall[0:iepoch+1],'b--')
            #plt.show()
            ttt='epoch{}enerfy.png'.format(str(iepoch).zfill(3))
            plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
            plt.close(fig)
            fig = plt.figure()
            plt.plot(lossd_all[0:iepoch+1],'r-')
            plt.plot(lossg1_all[0:iepoch+1],'b--')
            plt.plot(lossg2_all[0:iepoch+1],'g--')
            #plt.show()
            ttt='epoch{}lossdrf.png'.format(str(iepoch).zfill(3))
            plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
            plt.close(fig)
            fig = plt.figure()
            plt.plot(lossr_all[0:iepoch+1],'r-')

            #plt.show()
            ttt='epoch{}ReconstrucLoss.png'.format(str(iepoch).zfill(3))
            plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
            plt.close(fig)
        if 0:
            ###############show filters and basis##################
            fig=plothist(Yp)
            ttt='epoch{}histori.png'.format(str(iepoch).zfill(3))
            plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
            plt.close(fig)
            fig=plothist(Gyspaf)
            ttt='epoch{}histdes.png'.format(str(iepoch).zfill(3))
            plt.savefig(os.path.join(figpath, ttt), bbox_inches='tight')
            plt.close(fig)
        if iepoch % 5 == 0:
            save_path = saver.save(sess, modelpath, global_step=iepoch,write_meta_graph=False)











