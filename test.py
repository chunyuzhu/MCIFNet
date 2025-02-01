import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from models.TSFN import TSFN
# from models.MambaINR import MambaINR
from models.PSRTNet import PSRTnet
from models._3DT_Net import _3DT_Net
from models.SSRNET import SSRNET
from models.HyperKite import HyperKite
from models.MoGDCNx4 import MoGDCNx4
from models.MoGDCN import MoGDCN
from models.MoGDCNx16 import MoGDCNx16
from models.DCT import DCT
from utils import *
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam
from data_loader import build_datasets
from models.ASSMamba import ASSMamba
from models.ASSMamba_no_RSSG import ASSMamba_no_RSSG
from models.ASSMamba_no_CAB import ASSMamba_no_CAB
from models.ASSMamba_no_GINS import ASSMamba_no_GINS
from models.ASSMamba_no_VSSM import ASSMamba_no_VSSM
from models.ASSMamba_no_SEFU import ASSMamba_no_SEFU
from models.P3Net import P3Net
from validate import validate
from train import train
import pdb
import args_parser
from torch.nn import functional as F
import cv2
from time import *
import os
import scipy.io as io
from thop import profile
torch.cuda.is_available()

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print (args)

# torch.cuda.is_available()
def main():
    if args.dataset == 'PaviaU':
      args.n_bands = 103
    elif args.dataset == 'Pavia':
      args.n_bands = 102
    elif args.dataset == 'Chikusei':
      args.n_bands = 128
    elif args.dataset == 'IEEE2018':
      args.n_bands = 48
    elif args.dataset == 'Botswana':
      args.n_bands = 145
      
    # Custom dataloader
    train_list, test_list = build_datasets(args.root, 
                                           args.dataset, 
                                           args.image_size, 
                                           args.n_select_bands, 
                                           args.scale_ratio)
    
    # Build the models
    if args.dataset == 'PaviaU':
      args.n_bands = 103
    
    elif args.dataset == 'Pavia':
      args.n_bands = 102
    elif args.dataset == 'Chikusei':
      args.n_bands = 128
    elif args.dataset == 'IEEE2018':
      args.n_bands = 48
    elif args.dataset == 'Botswana':
      args.n_bands = 145
    # Build the models
    if args.arch == 'SSRNET' or args.arch == 'SpatRNET' or args.arch == 'SpecRNET':
      model = SSRNET(args.arch,
                     args.scale_ratio,
                     args.n_select_bands, 
                     args.n_bands).cuda()
    
    elif args.arch == 'PSRTnet':
      model = PSRTnet(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands,
                     args.image_size).cuda()
    elif args.arch == 'DCT':
      model = DCT(n_colors=args.n_bands, upscale_factor=args.scale_ratio, n_feats=180)
    elif args.arch == 'HyperKite':
      model = HyperKite(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands).cuda()
    elif args.arch == 'TSFN':
      model = TSFN(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands).cuda()
    elif args.arch == 'MoGDCNx4':
      model = MoGDCNx4(scale_ratio=args.scale_ratio,
                       n_select_bands=args.n_select_bands, 
                       n_bands=args.n_bands,
                       img_size=args.image_size).cuda()
    elif args.arch == 'MoGDCN':
      model = MoGDCN(scale_ratio=args.scale_ratio,
                       n_select_bands=args.n_select_bands, 
                       n_bands=args.n_bands,
                       img_size=args.image_size).cuda()
    elif args.arch == 'MoGDCNx16':
      model = MoGDCNx16(scale_ratio=args.scale_ratio,
                       n_select_bands=args.n_select_bands, 
                       n_bands=args.n_bands,
                       img_size=args.image_size).cuda()
    elif args.arch == '_3DT_Net':
      model = _3DT_Net(args.scale_ratio, 8, 
                       args.n_bands,args.n_select_bands
                       ).cuda()
    elif args.arch == 'ASSMamba_no_GINS':
      model = ASSMamba_no_GINS(img_size=64,
                       patch_size=1,
                       in_chans_MSI=args.n_select_bands,
                       in_chans_HSI=args.n_bands,
                       embed_dim=96,
                       depths=(1,),
                       mlp_dim=[256, 128],
                       drop_rate=0.,
                       d_state = 16,
                       mlp_ratio=2.,
                       drop_path_rate=0.1,
                       norm_layer=nn.LayerNorm,
                       patch_norm=True,
                       use_checkpoint=False,
                       upscale=2,
                       img_range=1.,
                       upsampler='',
                       resi_connection='1conv').cuda()
    elif args.arch == 'ASSMamba':
      model = ASSMamba(img_size=64,
                       patch_size=1,
                       in_chans_MSI=args.n_select_bands,
                       in_chans_HSI=args.n_bands,
                       embed_dim=96,
                       depths=(1,),
                       mlp_dim=[256, 128],
                       drop_rate=0.,
                       d_state = 16,
                       mlp_ratio=2.,
                       drop_path_rate=0.1,
                       norm_layer=nn.LayerNorm,
                       patch_norm=True,
                       use_checkpoint=False,
                       upscale=2,
                       img_range=1.,
                       upsampler='',
                       resi_connection='1conv').cuda()
    elif args.arch == 'ASSMamba_no_VSSM':
      model = ASSMamba_no_VSSM(img_size=64,
                       patch_size=1,
                       in_chans_MSI=args.n_select_bands,
                       in_chans_HSI=args.n_bands,
                       embed_dim=96,
                       depths=(1,),
                       mlp_dim=[256, 128],
                       drop_rate=0.,
                       d_state = 16,
                       mlp_ratio=2.,
                       drop_path_rate=0.1,
                       norm_layer=nn.LayerNorm,
                       patch_norm=True,
                       use_checkpoint=False,
                       upscale=2,
                       img_range=1.,
                       upsampler='',
                       resi_connection='1conv').cuda()
    elif args.arch == 'ASSMamba_no_SEFU':
      model = ASSMamba_no_SEFU(img_size=64,
                       patch_size=1,
                       in_chans_MSI=args.n_select_bands,
                       in_chans_HSI=args.n_bands,
                       embed_dim=96,
                       depths=(1,),
                       mlp_dim=[256, 128],
                       drop_rate=0.,
                       d_state = 16,
                       mlp_ratio=2.,
                       drop_path_rate=0.1,
                       norm_layer=nn.LayerNorm,
                       patch_norm=True,
                       use_checkpoint=False,
                       upscale=2,
                       img_range=1.,
                       upsampler='',
                       resi_connection='1conv').cuda()
    elif args.arch == 'ASSMamba_no_RSSG':
      model = ASSMamba_no_RSSG(img_size=64,
                       patch_size=1,
                       in_chans_MSI=args.n_select_bands,
                       in_chans_HSI=args.n_bands,
                       embed_dim=96,
                       depths=(1,),
                       mlp_dim=[256, 128],
                       drop_rate=0.,
                       d_state = 16,
                       mlp_ratio=2.,
                       drop_path_rate=0.1,
                       norm_layer=nn.LayerNorm,
                       patch_norm=True,
                       use_checkpoint=False,
                       upscale=2,
                       img_range=1.,
                       upsampler='',
                       resi_connection='1conv').cuda()
    elif args.arch == 'ASSMamba_no_CAB':
      model = ASSMamba_no_CAB(img_size=64,
                       patch_size=1,
                       in_chans_MSI=args.n_select_bands,
                       in_chans_HSI=args.n_bands,
                       embed_dim=96,
                       depths=(1,),
                       mlp_dim=[256, 128],
                       drop_rate=0.,
                       d_state = 16,
                       mlp_ratio=2.,
                       drop_path_rate=0.1,
                       norm_layer=nn.LayerNorm,
                       patch_norm=True,
                       use_checkpoint=False,
                       upscale=2,
                       img_range=1.,
                       upsampler='',
                       resi_connection='1conv').cuda()
    elif args.arch == 'P3Net':
        model = P3Net(scale_ratio=args.scale_ratio,
                         img_size=64,
                         patch_size=1,
                         in_chans_MSI=args.n_select_bands,
                         in_chans_HSI=args.n_bands,
                         embed_dim=64,
                         depths=(1,),
                         mlp_dim=[256, 128],
                         drop_rate=0.,
                         d_state = 16,
                         mlp_ratio=2.,
                         drop_path_rate=0.1,
                         norm_layer=nn.LayerNorm,
                         patch_norm=True,
                         use_checkpoint=False,
                         upscale=2,
                         img_range=1.,
                         upsampler='',
                         resi_connection='1conv').cuda()
    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('arch', args.arch) 
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))


    test_ref, test_lr, test_hr = test_list
    model.eval()

    # Set mini-batch dataset
    ref = test_ref.float().detach()
    lr = test_lr.float().detach()
    hr = test_hr.float().detach()
    
    begin_time = time()
    if args.arch == 'SSRNET':
        out, _, _, _, _, _ = model(lr.cuda(), hr.cuda())
    elif args.arch == 'SpatRNET':
        _, out, _, _, _, _ = model(lr.cuda(), hr.cuda())
    elif args.arch == 'SpecRNET':
        _, _, out, _, _, _ = model(lr.cuda(), hr.cuda())
    elif args.arch == 'SwinCGAN':
        out, _, _, _, _, _ = model(lr.cuda(), hr.cuda(), args.scale_ratio)
    else:
        out, _, _, _, _, _ = model(lr.cuda(), hr.cuda())
    end_time = time()
    run_time = (end_time-begin_time)*1000

    print ()
    print ()
    print ('Dataset:   {}'.format(args.dataset))
    print ('Arch:   {}'.format(args.arch))
    print ('ModelSize(M):   {}'.format(np.around(os.path.getsize(model_path)//1024/1024.0, decimals=2)))
    print ('Time(Ms):   {}'.format(np.around(run_time, decimals=2)))
    flops, params = profile(model, inputs=(lr.cuda(),hr.cuda()))
    flops = flops/1000000000
    print ('flops:',flops)
    print ('params:',params/1000000)
    
    ref = ref.detach().cpu().numpy()
    out = out.detach().cpu().numpy()
    
    slr  =  F.interpolate(lr, scale_factor=args.scale_ratio, mode='bilinear')
    slr = slr.detach().cpu().numpy()
    slr  =  np.squeeze(slr).transpose(1,2,0).astype(np.float64)
    
    sref = np.squeeze(ref).transpose(1,2,0).astype(np.float64)
    sout = np.squeeze(out).transpose(1,2,0).astype(np.float64)
    
    io.savemat('./实验结果的mat格式/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+args.arch+'.mat',{'Out':sout})
    io.savemat('./实验结果的mat格式/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'REF.mat',{'REF':sref})
    io.savemat('./实验结果的mat格式/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'Upsample.mat',{'Out':slr})
    
    t_lr = np.squeeze(lr).detach().cpu().numpy().transpose(1,2,0).astype(np.float64)
    t_hr = np.squeeze(hr).detach().cpu().numpy().transpose(1,2,0).astype(np.float64)
    
    io.savemat('./为传统方法准备数据/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'lr'+'.mat',{'HSI':t_lr})
    io.savemat('./为传统方法准备数据/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'hr'+'.mat',{'MSI':t_hr})
    
    
    psnr = calc_psnr(ref, out)
    rmse = calc_rmse(ref, out)
    ergas = calc_ergas(ref, out)
    sam = calc_sam(ref, out)
    print ('RMSE:   {:.4f};'.format(rmse))
    print ('PSNR:   {:.4f};'.format(psnr))
    print ('ERGAS:   {:.4f};'.format(ergas))
    print ('SAM:   {:.4f}.'.format(sam))

   
if __name__ == '__main__':
    main()
