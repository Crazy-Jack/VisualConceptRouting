# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 06:28:54 2019

@author: dell
"""

import tensorflow as tf
import numpy as np

def get_weight(shape, gain=np.sqrt(2), use_wscale=True, fan_in=None):
    if fan_in is None:
        fan_in = np.prod(shape[:-1])
    #print( "current", shape[:-1], fan_in)
    std = gain / np.sqrt(fan_in) # He init

    if use_wscale:
    #if 0:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
        
# Fully-connected layer.
def dense(x, fmaps, gain=np.sqrt(2), use_wscale=True,name='fc'):
  with tf.variable_scope(name):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    bias = tf.get_variable("bias", [fmaps], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, w)+ bias

def conv2d(x, fmaps,k=3, s=1, gain=np.sqrt(2), use_wscale=True, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        w = get_weight([k, k, x.shape[-1].value, fmaps], gain=gain, use_wscale=use_wscale)
        w = tf.cast(w, x.dtype)
        if padding == 'Other':
            padding = 'VALID'
            x = tf.pad(x, [[0,0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        elif padding == 'VALID':
            padding = 'VALID'

        conv = tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding=padding)
        #biases = tf.get_variable('biases', [fmaps], initializer=tf.constant_initializer(0.0))
        #conv = tf.nn.bias_add(conv, biases)      
        return conv

def deconv2d(x, fmaps,output_shape,k=3, s=2, gain=np.sqrt(2), use_wscale=True, padding='SAME', name="deconv2d"):
    with tf.variable_scope(name):
        w = get_weight([k, k, x.shape[-1].value,fmaps], gain=gain, use_wscale=use_wscale)
        w = tf.cast(w, x.dtype)
        w = tf.transpose(w,[0,1,3,2])
        conv = tf.nn.conv2d_transpose(x, filter=w, output_shape=output_shape, strides=[1, s, s, 1], padding=padding)                
        
        #biases = tf.get_variable('biases', [fmaps], initializer=tf.constant_initializer(0.0))
        #conv = tf.nn.bias_add(conv, biases)      
        return conv
    
def lrelu(x, alpha=0.2):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)
# Nearest-neighbor upscaling layer.
def upscale2d(x, factor=2):
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        return x
    
def downscale2d(x, factor=2):
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):        
        return tf.nn.pool(x, [factor, factor], "AVG", "SAME", strides=[factor, factor])

def pixl_norm(x, eps=1e-8):
    if len(x.shape) > 2:
        axis_ = 3
    else:
        axis_ = 1
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis_, keep_dims=True) + eps)
        #return x
def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=tf.GraphKeys.UPDATE_OPS,
                                        is_training=is_training, scope=scope)
# Minibatch standard deviation.
def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NHWC]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMHWC] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMHWC] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMHWC] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MHWC]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MHWC]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1], s[2],1])             # [NHW1]  Replicate over group and pixels.
        return tf.concat([x, y], axis=3)                        # [NHWC]  Append as new fmap.

def block_down(x,fmaps,k=3, s=1,name='block'):
    with tf.variable_scope(name):
        x=lrelu(conv2d(x, fmaps,k, s,name="conv2d_1"))
        x=lrelu(conv2d(x, fmaps,k, s,name="conv2d_2"))
        x = downscale2d(x)
        return x

def block_down1(x,fmaps,k=3, s=1,name='block'):
    with tf.variable_scope(name):
        x=lrelu(conv2d(x, fmaps,k, s,name="conv2d_1"))
        x = downscale2d(x)
        return x

def resblock_down(x,fmaps,k=3, s=1,name='resblock'):
    with tf.variable_scope(name):
        shortcut=downscale2d(x)
        shortcut=conv2d(shortcut, fmaps,1, s,name="conv2d_0")
       
        x=lrelu(conv2d(x, fmaps,k, s,name="conv2d_1"))
        x=lrelu(conv2d(x, fmaps,k, s,name="conv2d_2"))
        x = downscale2d(x) 
        return x+shortcut

def block_up(x,fmaps,k=3, s=1,name='block'):
    with tf.variable_scope(name):
        x = upscale2d(x)
        x = pixl_norm(tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_1")))
        x = pixl_norm(tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_2")))
        return x

def add_class_condition(x, class_info):
    """add class information for cnn maps: concate class_info tensor with x"""
    nb, nh, nw, nc = x.shape
    class_info = tf.reshape(class_info, [-1, 1, 1, class_info.shape[1]])
    class_info = tf.tile(class_info, [1, nh, nw, 1])
    class_info = tf.reshape(class_info, [-1, nh, nw, class_info.shape[1]])
    new_x = tf.concat([x, class_info], 3)
    return new_x



def block_up_no(x,fmaps,k=3, s=1,factor=2,name='block', class_info=None, layer_condition=False):
    with tf.variable_scope(name):
        x = upscale2d(x,factor)
        if layer_condition:
            x = add_class_condition(x, class_info)
        x = tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_1"))
        if layer_condition:
            x = add_class_condition(x, class_info)
        x = tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_2"))
        
        return x

def block_up1(x,fmaps,k=3, s=1,name='block'):
    with tf.variable_scope(name):
        x = upscale2d(x)
        #x = pixl_norm(tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_1")))
        x = tf.nn.relu(batch_norm(conv2d(x, fmaps,k, s,name="conv2d_1")))
        return x

def block_up0(x,fmaps,k=3, s=1,name='block'):
    with tf.variable_scope(name):
        x = upscale2d(x)
        x = conv2d(x, fmaps,k, s,name="conv2d_1")        
        return tf.nn.relu(x)

def resblock_up(x,fmaps,k=3, s=1,name='block'):
    with tf.variable_scope(name):
        shortcut = conv2d(x, fmaps,1, s,name="conv2d_0")
        shortcut = upscale2d(shortcut)
    
        x = upscale2d(x)
        x = pixl_norm(tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_1")))
        x = pixl_norm(tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_2")))
        return x+shortcut
def resblock_up_no(x,fmaps,k=3, s=1,name='block', class_info=None, layer_condition=False):
    with tf.variable_scope(name):
        shortcut = conv2d(x, fmaps,1, s,name="conv2d_0")
        shortcut = upscale2d(shortcut)
    
        x = upscale2d(x)
        if layer_condition:
            x = add_class_condition(x, class_info)
        x = tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_1"))
        if layer_condition:
            x = add_class_condition(x, class_info)
        x = tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_2"))
        return x+shortcut
def resblockone_up_no(x,fmaps,k=3, s=1,name='block'):
    with tf.variable_scope(name):
        shortcut = conv2d(x, fmaps,1, s,name="conv2d_0")
        shortcut = upscale2d(shortcut)
    
        x = upscale2d(x)
        x = tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_1"))       
        return x+shortcut
#######################################
def relumask(fm):        
    shape = tf.shape(fm)
    drop = tf.where(fm <= 0,\
                        tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))            
    return drop

def block_up_cm_no(x,fmaps,k=3, s=1,factor=2,name='block'):
    with tf.variable_scope(name):
        x = upscale2d(x,factor)
        relumask1=relumask(conv2d(x, fmaps,k, s,name="conv2d_1"))
        x = tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_1"))
        relumask2=relumask(conv2d(x, fmaps,k, s,name="conv2d_2"))
        x = tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_2"))
        return x,relumask1,relumask2

def block_up0_cm_no(x,fmaps,k=3, s=1,name='block'):
    with tf.variable_scope(name):
        x = upscale2d(x)
        relumask1=relumask(conv2d(x, fmaps,k, s,name="conv2d_1"))
        x = tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_1"))
        return x,relumask1

def block_up_mask_no(x,fmaps,relumask1,relumask2,k=3, s=1,factor=2,name='block'):
    with tf.variable_scope(name):
        x = upscale2d(x,factor)
        x = relumask1*conv2d(x, fmaps,k, s,name="conv2d_1")
        x = relumask2*conv2d(x, fmaps,k, s,name="conv2d_2")
        return x

def block_up0_mask_no(x,fmaps,relumask1,k=3, s=1,name='block'):
    with tf.variable_scope(name):
        x = upscale2d(x)
        x = relumask1*conv2d(x, fmaps,k, s,name="conv2d_1")
        return x

def resblock_up_cm_no(x,fmaps,k=3, s=1,name='block'):
    with tf.variable_scope(name):
        shortcut = conv2d(x, fmaps,1, s,name="conv2d_0")
        shortcut = upscale2d(shortcut)
    
        x = upscale2d(x)
        relumask1=relumask(conv2d(x, fmaps,k, s,name="conv2d_1"))
        x = tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_1"))
        relumask2=relumask(conv2d(x, fmaps,k, s,name="conv2d_2"))
        x = tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_2"))
        return x+shortcut,relumask1,relumask2

def resblockone_up_cm_no(x,fmaps,k=3, s=1,name='block'):
    with tf.variable_scope(name):
        shortcut = conv2d(x, fmaps,1, s,name="conv2d_0")
        shortcut = upscale2d(shortcut)
    
        x = upscale2d(x)
        relumask1=relumask(conv2d(x, fmaps,k, s,name="conv2d_1"))
        x = tf.nn.relu(conv2d(x, fmaps,k, s,name="conv2d_1"))
        relumask2=1
       
        return x+shortcut,relumask1,relumask2
def resblock_up_mask_no(x,fmaps,relumask1,relumask2,k=3, s=1,name='block'):
    with tf.variable_scope(name):
        shortcut = conv2d(x, fmaps,1, s,name="conv2d_0")
        shortcut = upscale2d(shortcut)
    
        x = upscale2d(x)
        x = relumask1*conv2d(x, fmaps,k, s,name="conv2d_1")
        x = relumask2*conv2d(x, fmaps,k, s,name="conv2d_2")
        return x+shortcut
def resblockone_up_mask_no(x,fmaps,relumask1,relumask2,k=3, s=1,name='block'):
    with tf.variable_scope(name):
        shortcut = conv2d(x, fmaps,1, s,name="conv2d_0")
        shortcut = upscale2d(shortcut)
    
        x = upscale2d(x)
        x = relumask1*conv2d(x, fmaps,k, s,name="conv2d_1")
       
        return x+shortcut    
def onehotmatrix2d(i,j,nh,nw):
    maskmatrix=tf.one_hot(i*nw+j,nh*nw,1.0,0.0)
    maskmatrix=tf.reshape(maskmatrix,[nh,nw])
    return maskmatrix
def set_value(row, y, val):    
    new_row = tf.concat([tf.reshape(row[0,:y],[1,-1]), [[val]], tf.reshape(row[0,y+1:],[1,-1])], axis=1)
    return new_row 
def sparsech(fm,kinchanel):
    shape = tf.shape(fm)
    #nb = shape[0]
    #nc = shape[2] 
    ###first along the channel direction

    th, _ = tf.nn.top_k(fm, kinchanel) # nb, nt, kinchanel
    thk = tf.slice(th,[0,0,kinchanel-1],[-1,-1,1]) # nb,nt,1    
    drop1 = tf.where(fm < thk, 
      tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
    fms=fm*drop1  
    return fms 
def sparseop(fm,kinchanel,ratio):
    shape = tf.shape(fm)
    nb = shape[0]
    nc = shape[3] 
    ###first along the spatial direction
    fm_t = tf.transpose(fm, [0, 3, 1, 2]) # nb, nc, nh, nw
    fm_r = tf.reshape(fm_t, tf.stack([nb, nc, -1])) # nb, nc, nh*nw    
    th, _ = tf.nn.top_k(fm_r, kinchanel) # nb, nc, kinchanel
    thk = tf.slice(th,[0,0,kinchanel-1],[-1,-1,1]) # nb*nc*1
    th_r = tf.reshape(thk, tf.stack([nb, 1, 1, nc])) # nb, 1, 1, nc
    drop1 = tf.where(fm < th_r, 
      tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
    fms=fm*drop1
    ####Then along the minibatch together
#    if ratio<1:
#        fms_t = tf.transpose(fms, [3,0,1,2]) # nc,nb, nh, nw
#        k = tf.cast(ratio * tf.cast(nb*kinchanel, tf.float32), tf.int32)
#        fms_r = tf.reshape(fms_t, tf.stack([nc, -1])) # nc, k 
#        th, _ = tf.nn.top_k(fms_r, k) # nc, k
#        thk = tf.slice(th,[0,k-1],[-1,1]) # nc*1
#        thk=tf.transpose(thk) #1*nc
#        th_r = tf.reshape(thk, tf.stack([1, 1, 1, nc])) # 1, 1, 1, nc
#        drop2 = tf.where(fms < th_r, 
#                        tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
#        fms=fms*drop2
    return fms
def onehotmatrix(i,j,nb,nh,nw,nc):
    maskmatrix=tf.one_hot(i*nw+j,nh*nw,1.0,0.0)
    maskmatrix=tf.reshape(maskmatrix,[nh,nw])
    maskmatrix=tf.reshape(maskmatrix,[1,nh,nw,1])
    maskmatrix=tf.tile(maskmatrix,[nb,1,1,nc])
    return maskmatrix

def allonmatrix(nb,nh,nw,nc):
    maskmatrix=tf.ones([nh,nw])
    maskmatrix=tf.reshape(maskmatrix,[nh,nw])
    maskmatrix=tf.reshape(maskmatrix,[1,nh,nw,1])
    maskmatrix=tf.tile(maskmatrix,[nb,1,1,nc])
    return maskmatrix

def onehotmatrixeq(nb,nh,nw,nc):
    maskcube=[]
    for i in range(nh):
        for j in range(nw):
            maskmatrix=tf.one_hot(i*nw+j,nh*nw,1.0,0.0)
            maskmatrix=tf.reshape(maskmatrix,[nh,nw])
            maskmatrix=tf.reshape(maskmatrix,[1,nh,nw,1])
            maskcube.append(maskmatrix)
    maskcube=tf.concat(maskcube,3)
    maskcube=tf.tile(maskcube,[nb,1,1,1])
    return maskcube
def normlizeZ(z):
     #normalization
    shape = z.get_shape().as_list()
    nb = shape[0]
    nt = shape[1]
    nv = shape[2]
    z = tf.reshape(z,[nb*nt,nv])    
    norm=tf.norm(z,axis=1) 
    norm=tf.reshape(norm,[nb*nt,1])
    z = z/(norm+1e-5)
    z = tf.reshape(z,[nb,nt,nv])
    return z   
def sparseopattributeq(fm,z):
    shape = fm.get_shape().as_list()
    nb = shape[0]
    nh = shape[1]
    nw = shape[2]
    nc = shape[3] 
    shape1=tf.shape(z)
    nv=shape1[2]
    #nt=shape1[1]
    drop1=onehotmatrixeq(nb,nh,nw,nc)
    #fms=fm*drop1  # b x h x w  x c  
    fms=drop1  # b x h x w  x c  
    #z = normlizeZ(z)   
    #########Add Attributes#################################
    fmsexpend=tf.zeros((nb,nh,nw,nv,nc))
    for i in range(nh):
        for j in range(nw):
            fmscomp=fms*onehotmatrix(i,j,nb,nh,nw,nc) # b x h x w  x c
            zcomp=z[:,i*4+j,:] #b x v
          
            fmscomp=tf.expand_dims(fmscomp,3) # b x h x w x 1 x c
            fmscomp=tf.tile(fmscomp,tf.stack([1,1,1,nv,1])) # b x h x w x v x c
          
            zcomp = tf.reshape(zcomp,[nb,1,1,nv,1])
            zcomp = tf.tile(zcomp,tf.stack([1,nh,nw,1,nc])) # b x h x w x v x c
            fmscomp = fmscomp * zcomp
            fmsexpend = fmsexpend + fmscomp
    return fmsexpend
 
def sparsesingleop(fm,kinchanel,ratio,i):
    shape = tf.shape(fm)
    nb = shape[0]
    #nh = shape[1]
    #nw = shape[2]
    nc = shape[3]
    ###first along the spatial direction
    fm_t = tf.transpose(fm, [0, 3, 1, 2]) # nb, nc, nh, nw
    fm_r = tf.reshape(fm_t, tf.stack([nb, nc, -1])) # nb, nc, nh*nw    
    th, _ = tf.nn.top_k(fm_r, kinchanel) # nb, nc, kinchanel
    thk = tf.slice(th,[0,0,kinchanel-1],[-1,-1,1]) # nb*nc*1
    th_r = tf.reshape(thk, tf.stack([nb, 1, 1, nc])) # nb, 1, 1, nc
    drop1 = tf.where(fm < th_r, 
      tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
    fms=fm*drop1
#    ####Then along the minibatch together
#    if ratio<1:
#        fms_t = tf.transpose(fms, [3,0,1,2]) # nc,nb, nh, nw
#        k = tf.cast(ratio * tf.cast(nb*kinchanel, tf.float32), tf.int32)
#        fms_r = tf.reshape(fms_t, tf.stack([nc, -1])) # nc, k 
#        th, _ = tf.nn.top_k(fms_r, k) # nc, k
#        thk = tf.slice(th,[0,k-1],[-1,1]) # nc*1
#        thk=tf.transpose(thk) #1*nc
#        th_r = tf.reshape(thk, tf.stack([1, 1, 1, nc])) # 1, 1, 1, nc
#        drop2 = tf.where(fms < th_r, 
#                        tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
#        fms=fms*drop2
     #####################
     # for computing the location easily
    fms=tf.transpose(fms, [0, 3, 1, 2]) #nb x nc x nh x nw
    shapes = tf.shape(fms)
#    # overall sparse along cxwxh 
    fms_r=tf.reshape(fms,tf.stack([nb,-1]))
    tk=tf.cast(ratio * tf.cast(nc*kinchanel, tf.float32), tf.int32)
    th, loca = tf.nn.top_k(fms_r, tk) # nb*k
#    orderinspace=tf.mod(loca[:,i],nh*nw)
#    rowindex=tf.reshape(orderinspace // nw,[nb,1])
#    colindex=tf.reshape(tf.mod(orderinspace, nw),[nb,1])
#    location=tf.reshape(tf.concat([rowindex,colindex],1),tf.stack([nb,2]))
           
##############    
     #if i==0:    # the 1st comp
    def f1(i,th,fms,nb,shapes):
        bit=i       
        th1 = tf.slice(th,[0,bit],[-1,1]) # nb*1        
        thr = tf.reshape(th1, tf.stack([nb, 1,1,1]))
        drop = tf.where(fms < thr,\
                        tf.zeros(shapes, tf.float32), tf.ones(shapes, tf.float32))
        return fms*drop
#if i>0:
    def f2(i,th,fms,nb,shapes):
        bit=i
        tht= tf.slice(th,[0,bit],[-1,1])
        thts= tf.slice(th,[0,bit-1],[-1,1])
        thtr = tf.reshape(tht, tf.stack([nb, 1,1,1]))
        thtsr = tf.reshape(thts, tf.stack([nb, 1,1,1]))
        dropm = tf.where(fms < thtr,\
                        tf.zeros(shapes, tf.float32), tf.ones(shapes, tf.float32))
        dropms = tf.where(fms < thtsr,\
                        tf.ones(shapes, tf.float32), tf.zeros(shapes, tf.float32))
        return fms*dropm*dropms
   
    fms= tf.cond(tf.greater(i, 0), lambda :f2(i,th,fms,nb,shapes), lambda :f1(i,th,fms,nb,shapes))
    fms=tf.transpose(fms,[0, 2,3,1])
    ###############
    return fms
    #return fms,location

def sparsesingleopxylocat(fm,kinchanel,ratio,i,x,y):
    shape = tf.shape(fm)
    nb = shape[0]
    nh = shape[1]
    nw = shape[2]
    nc = shape[3]
    ###first along the spatial direction
    fm_t = tf.transpose(fm, [0, 3, 1, 2]) # nb, nc, nh, nw
    fm_r = tf.reshape(fm_t, tf.stack([nb, nc, -1])) # nb, nc, nh*nw    
    th, _ = tf.nn.top_k(fm_r, kinchanel) # nb, nc, kinchanel
    thk = tf.slice(th,[0,0,kinchanel-1],[-1,-1,1]) # nb*nc*1
    th_r = tf.reshape(thk, tf.stack([nb, 1, 1, nc])) # nb, 1, 1, nc
    drop1 = tf.where(fm < th_r, 
      tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
    fms=fm*drop1
#    ####Then along the minibatch together
     #####################
    maskspatial=tf.one_hot(x*nh+y,nh*nw,1.0,0.0)
    maskspatial=tf.reshape(maskspatial,[nh,nw])
    maskspatial=tf.reshape(maskspatial,[1,nh,nw,1])
    maskspatial=tf.tile(maskspatial,[nb,1,1,nc])
    fms=fms*maskspatial
    nonzeros = tf.where(fms > 0, 
      tf.ones(shape, tf.float32), tf.zeros(shape, tf.float32))
    numbasisloc=tf.reduce_sum(nonzeros,axis=[1,2,3]) #nb
    ####################    
    shapes = tf.shape(fms)
#    # overall sparse along cxwxh 
    fms_r=tf.reshape(fms,tf.stack([nb,-1]))
    tk,_=tf.nn.top_k(numbasisloc, 1)
    th, _ = tf.nn.top_k(fms_r, tf.cast(tk[0],tf.int32)) # nb*k      
##############    
     #if i==0:    # the 1st comp
    def f1(i,th,fms,nb,shapes):
        bit=i       
        th1 = tf.slice(th,[0,bit],[-1,1]) # nb*1        
        thr = tf.reshape(th1, tf.stack([nb, 1,1,1]))
        drop = tf.where(fms < thr,\
                        tf.zeros(shapes, tf.float32), tf.ones(shapes, tf.float32))
        return fms*drop
#if i>0:
    def f2(i,th,fms,nb,shapes):
        bit=i
        tht= tf.slice(th,[0,bit],[-1,1])
        thts= tf.slice(th,[0,bit-1],[-1,1])
        thtr = tf.reshape(tht, tf.stack([nb, 1,1,1]))
        thtsr = tf.reshape(thts, tf.stack([nb, 1,1,1]))
        dropm = tf.where(fms < thtr,\
                        tf.zeros(shapes, tf.float32), tf.ones(shapes, tf.float32))
        dropms = tf.where(fms < thtsr,\
                        tf.ones(shapes, tf.float32), tf.zeros(shapes, tf.float32))
        return fms*dropm*dropms
   
    fms= tf.cond(tf.greater(i, 0), lambda :f2(i,th,fms,nb,shapes), lambda :f1(i,th,fms,nb,shapes))
    ###############
    return fms,numbasisloc

def densesingleopxylocat(fm,ratio,i,x,y):
    shape = tf.shape(fm)
    nb = shape[0]
    nh = shape[1]
    nw = shape[2]
    nc = shape[3]
    ###first along the spatial direction
   
    fms=fm
#    ####Then along the minibatch together
     #####################
    maskspatial=tf.one_hot(x*nh+y,nh*nw,1.0,0.0)
    maskspatial=tf.reshape(maskspatial,[nh,nw])
    maskspatial=tf.reshape(maskspatial,[1,nh,nw,1])
    maskspatial=tf.tile(maskspatial,[nb,1,1,nc])
    fms=fms*maskspatial
    nonzeros = tf.where(fms > 0, 
      tf.ones(shape, tf.float32), tf.zeros(shape, tf.float32))
    numbasisloc=tf.reduce_sum(nonzeros,axis=[1,2,3]) #nb
    ####################    
    shapes = tf.shape(fms)
#    # overall sparse along cxwxh 
    fms_r=tf.reshape(fms,tf.stack([nb,-1]))
    tk,_=tf.nn.top_k(numbasisloc, 1)
    th, _ = tf.nn.top_k(fms_r, tf.cast(tk[0],tf.int32)) # nb*k      
##############    
     #if i==0:    # the 1st comp
    def f1(i,th,fms,nb,shapes):
        bit=i       
        th1 = tf.slice(th,[0,bit],[-1,1]) # nb*1        
        thr = tf.reshape(th1, tf.stack([nb, 1,1,1]))
        drop = tf.where(fms < thr,\
                        tf.zeros(shapes, tf.float32), tf.ones(shapes, tf.float32))
        return fms*drop
#if i>0:
    def f2(i,th,fms,nb,shapes):
        bit=i
        tht= tf.slice(th,[0,bit],[-1,1])
        thts= tf.slice(th,[0,bit-1],[-1,1])
        thtr = tf.reshape(tht, tf.stack([nb, 1,1,1]))
        thtsr = tf.reshape(thts, tf.stack([nb, 1,1,1]))
        dropm = tf.where(fms < thtr,\
                        tf.zeros(shapes, tf.float32), tf.ones(shapes, tf.float32))
        dropms = tf.where(fms < thtsr,\
                        tf.ones(shapes, tf.float32), tf.zeros(shapes, tf.float32))
        return fms*dropm*dropms
   
    fms= tf.cond(tf.greater(i, 0), lambda :f2(i,th,fms,nb,shapes), lambda :f1(i,th,fms,nb,shapes))
    ###############
    return fms,numbasisloc

def sparsesingleop1(fm,kinchanel,ratio,i):
    shape = tf.shape(fm)
    nb = shape[0]
    #nh = shape[1]
    #nw = shape[2]
    nc = shape[3]
    ###first along the spatial direction
    fm_t = tf.transpose(fm, [0, 3, 1, 2]) # nb, nc, nh, nw
    fm_r = tf.reshape(fm_t, tf.stack([nb, nc, -1])) # nb, nc, nh*nw    
    th, _ = tf.nn.top_k(fm_r, kinchanel) # nb, nc, kinchanel
    tfmax,_=tf.nn.top_k(tf.reshape(fm, tf.stack([-1])),1)
    thk = tf.slice(th,[0,0,kinchanel-1],[-1,-1,1]) # nb*nc*1
    th_r = tf.reshape(thk, tf.stack([nb, 1, 1, nc])) # nb, 1, 1, nc
    drop1 = tf.where(fm < th_r, 
      tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
    fms=tfmax*drop1
#    ####Then along the minibatch together
#    if ratio<1:
#        fms_t = tf.transpose(fms, [3,0,1,2]) # nc,nb, nh, nw
#        k = tf.cast(ratio * tf.cast(nb*kinchanel, tf.float32), tf.int32)
#        fms_r = tf.reshape(fms_t, tf.stack([nc, -1])) # nc, k 
#        th, _ = tf.nn.top_k(fms_r, k) # nc, k
#        thk = tf.slice(th,[0,k-1],[-1,1]) # nc*1
#        thk=tf.transpose(thk) #1*nc
#        th_r = tf.reshape(thk, tf.stack([1, 1, 1, nc])) # 1, 1, 1, nc
#        drop2 = tf.where(fms < th_r, 
#                        tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
#        fms=fms*drop2
     #####################
     # for computing the location easily
    fms=tf.transpose(fms, [0, 3, 1, 2]) #nb x nc x nh x nw
    shapes = tf.shape(fms)
#    # overall sparse along cxwxh 
    fms_r=tf.reshape(fms,tf.stack([nb,-1]))
    tk=tf.cast(ratio * tf.cast(nc*kinchanel, tf.float32), tf.int32)
    th, loca = tf.nn.top_k(fms_r, tk) # nb*k
#    orderinspace=tf.mod(loca[:,i],nh*nw)
#    rowindex=tf.reshape(orderinspace // nw,[nb,1])
#    colindex=tf.reshape(tf.mod(orderinspace, nw),[nb,1])
#    location=tf.reshape(tf.concat([rowindex,colindex],1),tf.stack([nb,2]))
           
##############    
     #if i==0:    # the 1st comp
    def f1(i,th,fms,nb,shapes):
        bit=i       
        th1 = tf.slice(th,[0,bit],[-1,1]) # nb*1        
        thr = tf.reshape(th1, tf.stack([nb, 1,1,1]))
        drop = tf.where(fms < thr,\
                        tf.zeros(shapes, tf.float32), tf.ones(shapes, tf.float32))
        return fms*drop
#if i>0:
    def f2(i,th,fms,nb,shapes):
        bit=i
        tht= tf.slice(th,[0,bit],[-1,1])
        thts= tf.slice(th,[0,bit-1],[-1,1])
        thtr = tf.reshape(tht, tf.stack([nb, 1,1,1]))
        thtsr = tf.reshape(thts, tf.stack([nb, 1,1,1]))
        dropm = tf.where(fms < thtr,\
                        tf.zeros(shapes, tf.float32), tf.ones(shapes, tf.float32))
        dropms = tf.where(fms < thtsr,\
                        tf.ones(shapes, tf.float32), tf.zeros(shapes, tf.float32))
        return fms*dropm*dropms
   
    fms= tf.cond(tf.greater(i, 0), lambda :f2(i,th,fms,nb,shapes), lambda :f1(i,th,fms,nb,shapes))
    fms=tf.transpose(fms,[0, 2,3,1])
    ###############
    return fms
    #return fms,location


def sparsechidx(fm,kinchanel):
    shape = tf.shape(fm)
#    nb = shape[0]
#    nc = shape[2] 
    ###first along the channel direction      
    th, _ = tf.nn.top_k(fm, kinchanel) # nb, nt, kinchanel
    thk = tf.slice(th,[0,0,kinchanel-1],[-1,-1,1]) # nb,nt,1    
    drop1 = tf.where(fm < thk, 
      tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))   
    return drop1 
       
def sparseopidx(fm,kinchanel,ratio):
    shape = tf.shape(fm)
    nb = shape[0]
    nc = shape[3] 
    ###first along the spatial direction
    fm_t = tf.transpose(fm, [0, 3, 1, 2]) # nb, nc, nh, nw
    fm_r = tf.reshape(fm_t, tf.stack([nb, nc, -1])) # nb, nc, nh*nw    
    th, _ = tf.nn.top_k(fm_r, kinchanel) # nb, nc, kinchanel
    thk = tf.slice(th,[0,0,kinchanel-1],[-1,-1,1]) # nb*nc*1
    th_r = tf.reshape(thk, tf.stack([nb, 1, 1, nc])) # nb, 1, 1, nc
    drop1 = tf.where(fm < th_r, 
      tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))  
    return drop1
  
def sparsechone(fm,kinchanel,i):
    shape = tf.shape(fm)
    #nb = shape[0]
    #nc = shape[2] 
    ###first along the channel direction
      
    th, _ = tf.nn.top_k(fm, kinchanel) # nb, nt, kinchanel
    thk = tf.slice(th,[0,0,kinchanel-1],[-1,-1,1]) # nb,nt,1    
    drop1 = tf.where(fm < thk, 
      tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
    fms=fm*drop1
    ##############    
    if i==0:    # the 1st comp
        bit=i       
        th1 = tf.slice(th,[0,0,bit],[-1,-1,1]) # nb*nt*1        
       
        drop = tf.where(fms < th1,\
                        tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
        fms=fms*drop
    if i>0:
        bit=i
        tht= tf.slice(th,[0,0,bit],[-1,-1,1])
        thts= tf.slice(th,[0,0,bit-1],[-1,-1,1])
       
        dropm = tf.where(fms < tht,\
                        tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
        dropms = tf.where(fms < thts,\
                        tf.ones(shape, tf.float32), tf.zeros(shape, tf.float32))
        fms=fms*dropm*dropms    
    return fms 

def sparsesingleopattributeq(fm,kinchanel,ratio,i,j,z):
    shape = fm.get_shape().as_list()
    nb = shape[0]
    nh = shape[1]
    nw = shape[2]
    nc = shape[3]
    shape1=tf.shape(z)
    nv=shape1[2]
    #nt=shape1[1]
    drop1=onehotmatrixeq(nb,nh,nw,nc)
    #fms=fm*drop1
    fms=drop1
    ####Then along the minibatch together
    #normalization
    #z = normlizeZ(z)    
    #####################
     # for computing the location easily   
    ##############    
    fmsexpend=tf.zeros((nb,nh,nw,nv,nc))
    fmscomp=fms*onehotmatrix(i,j,nb,nh,nw,nc) # b x h x w  x c
    zcomp=z[:,i*4+j,:] #b x v
    fmscomp=tf.expand_dims(fmscomp,3) # b x h x w x 1 x c
    fmscomp=tf.tile(fmscomp,tf.stack([1,1,1,nv,1])) # b x h x w x v x c
    zcomp = tf.reshape(zcomp,[nb,1,1,nv,1])
    zcomp = tf.tile(zcomp,tf.stack([1,nh,nw,1,nc])) # b x h x w x v x c
    fmscomp = fmscomp * zcomp
    fmsexpend = fmsexpend + fmscomp   
    
    return fmsexpend

