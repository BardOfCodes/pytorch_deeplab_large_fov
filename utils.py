import torch
import torch.nn as nn
import numpy as np
import cv2
import random
import math
from torch.autograd import Variable

def get_parameters(model, bias=False,final=False):
    if(final):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if (m.out_channels == 21):
                    if bias:
                        yield m.bias
                    else:
                        yield m.weight
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if (not m.out_channels == 21): 
                    if bias:
                        yield m.bias
                    else:
                        yield m.weight


def adjust_learning_rate(optimizer, lr):
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr']=2*lr
    optimizer.param_groups[2]['lr']=10*lr
    optimizer.param_groups[3]['lr']=20*lr
    return optimizer


def chunker(seq, size):
    return (seq[pos:pos+size] for pos in xrange(0,len(seq), size))


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list


def resize_label_batch(label, size):
    label_resized = Variable(torch.zeros((label.shape[3],1,size,size)))
    interp = nn.UpsamplingBilinear2d(size=(size, size))
    labelVar = Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar)
    return label_resized

def flip(I,flip_p):
    if flip_p>0.5:
        return I[:,::-1,:]
    else:
        return I
def blur(img_temp,blur_p):
    if blur_p>0.5:
        return cv2.GaussianBlur(img_temp,(3,3),1)
    else:
        return img_temp

def crop(img_temp,dim,new_p=True,h_p=0,w_p=0):
    h =img_temp.shape[0]
    w = img_temp.shape[1]
    trig_h=trig_w=False
    if(h>dim):
        if(new_p):
            h_p = int(random.uniform(0,1)*(h-dim))
        img_temp = img_temp[h_p:h_p+dim,:,:]
    elif(h<dim):
        trig_h = True
    if(w>dim):
        if(new_p):
            w_p = int(random.uniform(0,1)*(w-dim))
        img_temp = img_temp[:,w_p:w_p+dim,:]
    elif(w<dim):
        trig_w = True
    if(trig_h or trig_w):
        pad = np.zeros((dim,dim,3))
        pad[:img_temp.shape[0],:img_temp.shape[1],:] = img_temp
        return (pad,h_p,w_p)
    else:
        return (img_temp,h_p,w_p)

def rotate(img_temp,rot,rot_p):
    if(rot_p>0.5):
        rows,cols,ind = img_temp.shape
        h_pad = int(rows*abs(math.cos(rot/180.0*math.pi)) + cols*abs(math.sin(rot/180.0*math.pi)))
        w_pad = int(cols*abs(math.cos(rot/180.0*math.pi)) + rows*abs(math.sin(rot/180.0*math.pi)))
        final_img = np.zeros((h_pad,w_pad,3))
        final_img[(h_pad-rows)/2:(h_pad+rows)/2,(w_pad-cols)/2:(w_pad+cols)/2,:] = np.copy(img_temp)
        M = cv2.getRotationMatrix2D((w_pad/2,h_pad/2),rot,1)
        final_img = cv2.warpAffine(final_img,M,(w_pad,h_pad),flags = cv2.INTER_NEAREST)
        part_denom = (math.cos(2*rot/180.0*math.pi))
        w_inside = int((cols*abs(math.cos(rot/180.0*math.pi)) - rows*abs(math.sin(rot/180.0*math.pi)))/part_denom)
        h_inside = int((rows*abs(math.cos(rot/180.0*math.pi)) - cols*abs(math.sin(rot/180.0*math.pi)))/part_denom)
        final_img = final_img[(h_pad-h_inside)/2:(h_pad+h_inside)/2,(w_pad- w_inside)/2:(w_pad+ w_inside)/2,:]
        return final_img
    else:
        return img_temp


def get_data_from_chunk_v2(chunk,gt_path,img_path):
    dim = 321
    images = np.zeros((dim,dim,3,len(chunk)))
    gt = np.zeros((dim,dim,1,len(chunk)))
    for i,piece in enumerate(chunk):
        img_name = piece.split(' ')[0]
        gt_name =  piece.split(' ')[1]
        img_temp = cv2.imread(img_path+img_name)
        gt_temp = cv2.imread(gt_path+ gt_name)[:,:,:]
        
        flip_p = random.uniform(0, 1)
        # rot_p = random([-10,-7,-5,3,0,3,5,7,10],1)[0]
        # scale_p = random.uniform(0, 1)
        # blur_p = random.uniform(.uniform(0, 1)
        # rot = np.random.choice0, 1)
        # if(scale_p>0.75):
            # scale = random.uniform(0.75, 1.5)
        # else:
            # scale = 1
        # if(img_temp.shape[0]<img_temp.shape[1]):
            # ratio = dim*scale/float(img_temp.shape[0])
        # else:
            # ratio = dim*scale/float(img_temp.shape[1])
        # img_temp = cv2.resize(img_temp,(int(img_temp.shape[1]*ratio),int(img_temp.shape[0]*ratio))).astype(float)
        img_temp = flip(img_temp,flip_p)
        # img_temp = rotate(img_temp,rot,rot_p)
        # img_temp = blur(img_temp,blur_p)

        gt_temp[gt_temp == 255] = 0
        # gt_temp = cv2.resize(gt_temp,(int(gt_temp.shape[1]*ratio),int(gt_temp.shape[0]*ratio)) , interpolation = cv2.INTER_NEAREST)
        gt_temp = flip(gt_temp,flip_p)
        # gt_temp = rotate(gt_temp,rot,rot_p)
        img_temp = img_temp.astype('float')
        gt_temp = gt_temp.astype('float')
            
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img_temp,img_temp_h_p,img_temp_w_p = crop(img_temp,dim)
        images[:,:,:,i] = img_temp
        gt_temp = crop(gt_temp,dim,False,img_temp_h_p,img_temp_w_p)[0]
        gt[:,:,0,i] = gt_temp[:,:,0]
        
    a = int((dim+7)/8)#41
    labels = resize_label_batch(gt,a)
    images = images.transpose((3,2,0,1))
    images = Variable(torch.from_numpy(images).float()).cuda()
    return images, labels


def get_test_data_from_chunk_v2(chunk,im_path):
    dim = 513
    images = np.zeros((dim,dim,3,len(chunk)))
    for i,piece in enumerate(chunk):
        img_temp = cv2.imread(im_path+piece+'.jpg')
        img_temp = img_temp.astype('float')
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img_temp,img_temp_h_p,img_temp_w_p = crop(img_temp,dim)
        images[:,:,:,i] = img_temp
        
    images = images.transpose((3,2,0,1))
    images = Variable(torch.from_numpy(images).float(),volatile= True).cuda()
    return images
