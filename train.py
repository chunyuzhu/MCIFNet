import torch
from torch import nn
from utils import to_var, batch_ids2words
import random
import torch.nn.functional as F
import cv2
import numpy as np

def spatial_edge(x):
    edge1 = x[:, :, 0:x.size(2)-1, :] - x[:, :, 1:x.size(2), :]
    edge2 = x[:, :, :, 0:x.size(3)-1] - x[:, :,  :, 1:x.size(3)]

    return edge1, edge2

def spectral_edge(x):
    edge = x[:, 0:x.size(1)-1, :, :] - x[:, 1:x.size(1), :, :]

    return edge
def matlab_style_gauss2D(shape=(5,5), sigma=2):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gaussian_blur_tensor(x: torch.Tensor, ksize=5, sigma=2.0):
    # x: [N, C, H, W]
    assert x.ndim == 4
    N, C, H, W = x.shape

    k = matlab_style_gauss2D((ksize, ksize), sigma).astype(np.float32)
    k = torch.from_numpy(k).to(device=x.device, dtype=x.dtype)  # [kH, kW]
    k = k.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)          # [C,1,kH,kW]

    pad = ksize // 2
    # groups=C => 每个通道各自卷积，不混通道
    y = F.conv2d(x, k, padding=pad, groups=C)
    return y 

def train(train_list, 
          image_size, 
          scale_ratio, 
          n_bands, 
          arch, 
          model, 
          optimizer, 
          criterion, 
          epoch, 
          n_epochs,
          h_str ,
          w_str):
    # train_ref, train_lr, train_hr = train_list
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    train_ref, train_lr, train_hr = train_list
    # h, w = train_ref.size(2), train_ref.size(3)
    # HH = []
    # WW = []
    # for i in range (10001):
    #     h_str = random.randint(0, h-image_size-1)
    #     HH.append(h_str)
    #     w_str = random.randint(0, w-image_size-1)
    #     WW.append(w_str)
    # h_str = random.randint(0, h-image_size-1)
    # w_str = random.randint(0, w-image_size-1)

    train_ref = train_ref[:, :, h_str:h_str+image_size, w_str:w_str+image_size]    
    train_lr = gaussian_blur_tensor(train_ref, ksize=scale_ratio, sigma=2.0)
    train_lr = F.interpolate(train_ref, scale_factor=1/(scale_ratio*1.0))    
    train_hr = train_hr[:, :, h_str:h_str+image_size, w_str:w_str+image_size]

    model.train()

    # Set mini-batch dataset
    image_lr = to_var(train_lr).detach()
    image_hr = to_var(train_hr).detach()
    image_ref = to_var(train_ref).detach()

    # Forward, Backward and Optimize
    optimizer.zero_grad()

    out, out_spat, out_spec, edge_spat1, edge_spat2, edge_spec = model(image_lr, image_hr)
    ref_edge_spat1, ref_edge_spat2 = spatial_edge(image_ref)
    ref_edge_spec = spectral_edge(image_ref)

    if 'RNET' in arch:        
        loss_fus = criterion(out, image_ref)
        loss_spat = criterion(out_spat, image_ref)
        loss_spec = criterion(out_spec, image_ref)
        loss_spec_edge = criterion(edge_spec, ref_edge_spec)
        loss_spat_edge = 0.5*criterion(edge_spat1, ref_edge_spat1) + 0.5*criterion(edge_spat2, ref_edge_spat2)
        if arch == 'SpatRNET':
            loss = loss_spat + loss_spat_edge
        elif arch == 'SpecRNET':
            loss = loss_spec + loss_spec_edge
        elif arch == 'SSRNET':
            loss = loss_fus + loss_spat_edge + loss_spec_edge
    else:
        loss = criterion(out, image_ref)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.5, norm_type=2)#设定梯度阈值，和L2正则
    optimizer.step()

    # Print log info
    print('Epoch [%d/%d], Loss: %.4f'
          %(epoch, 
            n_epochs, 
            loss,
            ) 
         )
