# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:01:27 2023

@author: zxc
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from os import listdir
from torch.nn import functional as F
import cv2
import torch
import numpy as np
import os
import random
import scipy.io as scio
import h5py
import xlrd

def matlab_style_gauss2D(shape=(5,5),sigma=2): 
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h   
    
def get_spectral_response(xls_path):
    # xls_path = os.path.join(self.args.sp_root_path, data_name + '.xls')
    if not os.path.exists(xls_path):
        raise Exception("spectral response path does not exist")
    data = xlrd.open_workbook(xls_path)
    table = data.sheets()[0]
    num_cols = table.ncols
    cols_list = [np.array(table.col_values(i)).reshape(-1,1) for i in range(0,num_cols)]
    sp_data = np.concatenate(cols_list, axis=1)
    sp_data = sp_data / (sp_data.sum(axis=0))  #normalize the sepctral response
    return sp_data   

def downsamplePSF(img,sigma,stride):
    def matlab_style_gauss2D(shape=(5,5),sigma=2):
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    # generate filter same with fspecial('gaussian') function
    h = matlab_style_gauss2D((stride,stride),sigma)
    if img.ndim == 3:
        img_w,img_h,img_c = img.shape
    elif img.ndim == 2:
        img_c = 1
        img_w,img_h = img.shape
        img = img.reshape((img_w,img_h,1))
    from scipy import signal
    out_img = np.zeros((img_w//(stride), img_h//(stride), img_c))
    for i in range(img_c):
        out = signal.convolve2d(img[:,:,i],h,'valid')  
        out_img[:,:,i] = out[::stride,::stride]
    return out_img
def generate_low_HSI( img, scale_factor):
    (h, w, c) = img.shape
    img_lr = downsamplePSF(img, sigma=2, stride=scale_factor)
    return img_lr 

def generate_MSI(img, sp_matrix):
    w,h,c = img.shape
    # msi_channels = sp_matrix.shape[1]
    if sp_matrix.shape[0] == c:
        img_msi = np.dot(img.reshape(w*h,c), sp_matrix).reshape(w,h,sp_matrix.shape[1])
    else:
        raise Exception("The shape of sp matrix doesnot match the image")
    return img_msi

# root = 'E:/AMIGNet_fuben/data'
# scale_ratio = 4
# size =128
def build_datasets(root, dataset, size, n_select_bands, scale_ratio):
    # Imageh preprocessing, normalization for the pretrained resnet
    if dataset == 'PaviaU':
        img = scio.loadmat(root + '/' + 'PaviaU.mat')['paviaU']*1.0
        sp_matrix = get_spectral_response(root + '/' + 'PaviaU'+'_SRF.xls')
    elif dataset == 'Pavia':
        img = scio.loadmat(root + '/' + 'Pavia.mat')['pavia']*1.0
        sp_matrix = get_spectral_response(root + '/' + 'Pavia'+'_SRF.xls')
    elif dataset == 'Botswana':
        img = scio.loadmat(root + '/' + 'Botswana.mat')['Botswana']*1.0
        sp_matrix = get_spectral_response(root + '/' + 'Botswana'+'_SRF.xls')
    elif dataset == 'Xiongan':
        # img = scio.loadmat(root + '/' + 'Xiongan.mat')['xiongan']*1.0
        mat = h5py.File(root + '/' + 'Xiongan.mat')
        img = np.transpose(mat['Xiongan'])
        sp_matrix = get_spectral_response(root + '/' + 'Xiongan'+'_SRF.xls')
    elif dataset == 'IEEE2018':
        img = scio.loadmat(root + '/' + 'IEEE2018.mat')['IEEE2018']*1.0
        sp_matrix = get_spectral_response(root + '/' + 'IEEE2018'+'_SRF.xls')
    elif dataset == 'Chikusei':
        mat = h5py.File(root + '/' + 'Chikusei.mat')
        img = np.transpose(mat['Chikusei'])
        sp_matrix = get_spectral_response(root + '/' + 'Chikusei'+'_SRF.xls')
    
      
    print (img.shape)
    max = np.max(img)
    min = np.min(img)
    img = 255*((img - min) / (max - min + 0.0))

    # throwing up the edge
    # w_edge = img.shape[0]//scale_ratio*scale_ratio-img.shape[0]
    # h_edge = img.shape[1]//scale_ratio*scale_ratio-img.shape[1]
    # w_edge = -1  if w_edge==0  else  w_edge
    # h_edge = -1  if h_edge==0  else  h_edge
    # img = img[:w_edge, :h_edge, :]
    
    w_edge = img.shape[0]//scale_ratio*scale_ratio
    h_edge = img.shape[1]//scale_ratio*scale_ratio
   
    img = img[:w_edge, :h_edge, :]
    
    # cropping area
    width, height, n_bands = img.shape 
    w_str = width -192 
    h_str = 0
    w_end = w_str + 192 
    h_end = h_str + 192
    img_copy = img.copy()

    # test sample
    # gap_bands = n_bands / (n_select_bands-1.0)
    test_ref = img_copy[w_str:w_end, h_str:h_end, :].copy()
    test_hr = generate_MSI(test_ref, sp_matrix)
    test_lr = generate_low_HSI(test_ref, scale_ratio)
    
    
    img[w_str:w_end,h_str:h_end,:] = 0
    train_ref = img
    train_hr = generate_MSI(train_ref,sp_matrix)
    train_lr = generate_low_HSI(train_ref, scale_ratio)
    


    train_ref = torch.from_numpy(train_ref).permute(2,0,1).unsqueeze(dim=0)
    train_lr = torch.from_numpy(train_lr).permute(2,0,1).unsqueeze(dim=0) 
    train_hr = torch.from_numpy(train_hr).permute(2,0,1).unsqueeze(dim=0) 
    test_ref = torch.from_numpy(test_ref).permute(2,0,1).unsqueeze(dim=0) 
    test_lr = torch.from_numpy(test_lr).permute(2,0,1).unsqueeze(dim=0) 
    test_hr = torch.from_numpy(test_hr).permute(2,0,1).unsqueeze(dim=0) 

    return [train_ref, train_lr, train_hr], [test_ref, test_lr, test_hr]
































