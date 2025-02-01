import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-arch', type=str, default='MCIFNet',
                            choices=[
                                     # the proposed method
                                     'SSRNET', 
                                     # these five models are used for comparison experiments
                                     'TSFN'
                                     'HyperKite'
                                     '_3DT_Net'
                                     'PSRTnet'                                     
                                     'DCT'
                                     'Fusformer'
                                     
                                     'MCIFNet'
                                     ])

    parser.add_argument('-root', type=str, default='./data')    
    parser.add_argument('-dataset', type=str, default='IEEE2018',
                            choices=['PaviaU', 'Chikusei','Pavia','IEEE2018','Botswana','realdata'])    
    parser.add_argument('--scale_ratio', type=float, default=4)
    parser.add_argument('--n_bands', type=int, default=0)
    parser.add_argument('--n_select_bands', type=int, default=4)

    parser.add_argument('--model_path', type=str, 
                            default='./checkpoints/dataset_arch_4.pkl',
                            help='path for trained encoder')
    parser.add_argument('--train_dir', type=str, default='./data/dataset/train',  
                            help='directory for resized images')
    parser.add_argument('--val_dir', type=str, default='./data/dataset/val',  
                            help='directory for resized images')

    # learning settingl
    parser.add_argument('--n_epochs', type=int, default=10000,
                            help='end epoch for training')
    # rsicd: 3e-4, ucm: 1e-4,
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=128)
 
    args = parser.parse_args()
    return args
 