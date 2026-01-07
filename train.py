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
def _gauss2d_numpy(shape=(5, 5), sigma=2):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    s = h.sum()
    if s != 0:
        h /= s
    return h

def _gauss2d_torch(kernel_size: int, sigma: float, device, dtype):
    # 与 numpy 版本一致：坐标范围 [-m, m]
    m = (kernel_size - 1) / 2.0
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - m
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    h = torch.exp(-(xx*xx + yy*yy) / (2.0 * sigma * sigma))
    # 与 numpy 一致：极小值截断（近似即可，不做也基本一致）
    eps = torch.finfo(dtype).eps
    h = torch.where(h < eps * h.max(), torch.zeros_like(h), h)
    s = h.sum()
    if s != 0:
        h = h / s
    return h

def downsamplePSF(img, sigma, stride):
    """
    兼容：
      - numpy: (H,W,C) 或 (H,W)
      - torch: (B,C,H,W) 或 (C,H,W) 或 (H,W)
    输出：
      - numpy -> numpy (H//stride, W//stride, C)
      - torch -> torch (B,C,H_out,W_out) / (C,H_out,W_out)
    """
    # -------- torch 分支：保持与原始 numpy 实现等价（valid 卷积 + stride 下采样）--------
    if torch.is_tensor(img):
        x = img
        orig_ndim = x.dim()

        if orig_ndim == 2:          # (H,W) -> (1,1,H,W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif orig_ndim == 3:        # (C,H,W) -> (1,C,H,W)
            x = x.unsqueeze(0)
        elif orig_ndim == 4:        # (B,C,H,W)
            pass
        else:
            raise ValueError(f"Unsupported torch tensor dim={orig_ndim}, expected 2/3/4.")

        if not x.is_floating_point():
            x = x.float()

        B, C, H, W = x.shape
        k = stride  # 与你原实现一致：核大小 = stride
        kernel2d = _gauss2d_torch(kernel_size=k, sigma=sigma, device=x.device, dtype=x.dtype)  # (k,k)

        # depthwise: (C,1,k,k)
        weight = kernel2d.view(1, 1, k, k).repeat(C, 1, 1, 1)

        # 等价于：对每个通道做 valid 卷积，然后以 stride 采样
        y = F.conv2d(x, weight=weight, bias=None, stride=stride, padding=0, groups=C)

        # 还原维度
        if orig_ndim == 2:
            return y[0, 0]                 # (H_out,W_out)
        elif orig_ndim == 3:
            return y[0]                    # (C,H_out,W_out)
        else:
            return y                       # (B,C,H_out,W_out)

    # -------- numpy 分支：保留你原来的实现 --------
    h = _gauss2d_numpy((stride, stride), sigma)
    if img.ndim == 3:
        img_w, img_h, img_c = img.shape
    elif img.ndim == 2:
        img_c = 1
        img_w, img_h = img.shape
        img = img.reshape((img_w, img_h, 1))
    else:
        raise ValueError(f"Unsupported numpy array dim={img.ndim}, expected 2/3.")

    from scipy import signal
    out_img = np.zeros((img_w // stride, img_h // stride, img_c), dtype=np.float64)
    for i in range(img_c):
        out = signal.convolve2d(img[:, :, i], h, 'valid')
        out_img[:, :, i] = out[::stride, ::stride]
    return out_img

def generate_low_HSI(img, scale_factor, sigma=2):
    """
    现在支持：
      - numpy (H,W,C)
      - torch (B,C,H,W)
    """
    return downsamplePSF(img, sigma=sigma, stride=scale_factor)

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
    train_lr = generate_low_HSI(train_ref, scale_ratio, sigma=2.0)  
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
