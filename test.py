import argparse
import os
import yaml
import torch
from torch.utils.data.dataloader import DataLoader
from dataset import BaseKITTIDataset,KITTI_perturb
from mylogger import get_logger, print_highlight, print_warning
from CalibNet import CalibNet
import loss as loss_utils
import utils
import numpy as np

def options():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--config",type=str,default='config.yml')
    parser.add_argument("--dataset_path",type=str,default='data/')
    parser.add_argument("--skip_frame",type=int,default=1,help='skip frame of dataset')
    parser.add_argument("--pcd_sample",type=int,default=-1) # -1 means total sample
    parser.add_argument("--max_deg",type=float,default=10)  # 10deg in each axis  (see the paper)
    parser.add_argument("--max_tran",type=float,default=0.2)   # 0.2m in each axis  (see the paper)
    parser.add_argument("--mag_randomly",type=bool,default=True)
    # dataloader
    parser.add_argument("--batch_size",type=int,default=1,choices=[1],help='batch size of test dataloader must be 1')
    parser.add_argument("--num_workers",type=int,default=12)
    parser.add_argument("--pin_memory",type=bool,default=True,help='set it to False if your CPU memory is insufficient')
    parser.add_argument("--perturb_file",type=str,default='test_seq.csv')
    # schedule
    parser.add_argument("--device",type=str,default='cuda:0')
    parser.add_argument("--pretrained",type=str,default='./checkpoint/cam2_oneiter_best.pth')
    parser.add_argument("--log_dir",default='log/')
    parser.add_argument("--checkpoint_dir",type=str,default="checkpoint/")
    parser.add_argument("--res_dir",type=str,default='res/')
    parser.add_argument("--name",type=str,default='cam2_oneiter')
    # setting
    parser.add_argument("--inner_iter",type=int,default=1,help='inner iter of calibnet')
    # if CUDA is out of memory, please reduce batch_size, pcd_sample or inner_iter
    return parser.parse_args()

def test(args,chkpt:dict,test_loader):
    model = CalibNet(depth_scale=args.scale)
    device = torch.device(args.device)
    model.to(device)
    model.load_state_dict(chkpt['model'])
    model.eval()
    logger = get_logger('{name}-Test'.format(name=args.name),os.path.join(args.log_dir,args.name+'_test.log'),mode='w')
    logger.debug(args)
    res_npy = np.zeros([len(test_loader),6])
    for i,batch in enumerate(test_loader):
        rgb_img = batch['img'].to(device)
        B = rgb_img.size(0)
        pcd_range = batch['pcd_range'].to(device)
        uncalibed_pcd = batch['uncalibed_pcd'].to(device)
        uncalibed_depth_img = batch['uncalibed_depth_img'].to(device)
        InTran = batch['InTran'][0].to(device)
        igt = batch['igt'].to(device)
        img_shape = rgb_img.shape[-2:]
        depth_generator = utils.transform.DepthImgGenerator(img_shape,InTran,pcd_range,CONFIG['dataset']['pooling'])
        # model(rgb_img,uncalibed_depth_img)
        g0 = torch.eye(4).repeat(B,1,1).to(device)
        for _ in range(args.inner_iter):
            twist_rot, twist_tsl = model(rgb_img,uncalibed_depth_img)
            extran = utils.se3.exp(torch.cat([twist_rot,twist_tsl],dim=1))
            uncalibed_depth_img, uncalibed_pcd = depth_generator(extran,uncalibed_pcd)
            g0 = extran.bmm(g0)
        dg = g0.bmm(igt)
        rot_dx,tsl_dx = loss_utils.gt2euler(dg.squeeze(0).cpu().detach().numpy())
        rot_dx = rot_dx.reshape(-1)
        tsl_dx = tsl_dx.reshape(-1)
        res_npy[i,:] = np.abs(np.concatenate([rot_dx,tsl_dx]))
        logger.info('[{:05d}|{:05d}],mdx:{:.4f}'.format(i+1,len(test_loader),res_npy[i,:].mean().item()))
    np.save(os.path.join(os.path.join(args.res_dir,'{name}.npy'.format(name=args.name))),res_npy)
    logger.info('Angle error (deg): X:{:.4f},Y:{:.4f},Z:{:.4f}'.format(*np.degrees(np.mean(res_npy[:,:3],axis=0))))
    logger.info('Translation error (m): X:{:.4f},Y:{:.4f},Z:{:.4f}'.format(*np.mean(res_npy[:,3:],axis=0)))

if __name__ == "__main__":
    args = options()
    if not torch.cuda.is_available():
        args.device = 'cpu'
        print_warning('CUDA is not available, use CPU to run')
    os.makedirs(args.log_dir,exist_ok=True)
    with open(args.config,'r')as f:
        CONFIG : dict= yaml.load(f,yaml.SafeLoader)
    if os.path.exists(args.pretrained) and os.path.isfile(args.pretrained):
        chkpt = torch.load(args.pretrained)
        CONFIG.update(chkpt['config'])
        update_args = ['resize_ratio','name','scale']
        for up_arg in update_args:
            setattr(args,up_arg,chkpt['args'][up_arg]) 
    else:
        raise FileNotFoundError('pretrained checkpoint {:s} not found!'.format(os.path.abspath(args.pretrained)))
    print_highlight('args have been received, please wait for dataloader...')
    
    test_split = [str(index).rjust(2,'0') for index in CONFIG['dataset']['test']]
    test_dataset = BaseKITTIDataset(args.dataset_path,args.batch_size,test_split,CONFIG['dataset']['cam_id'],
                                     skip_frame=args.skip_frame,voxel_size=CONFIG['dataset']['voxel_size'],
                                     pcd_sample_num=args.pcd_sample,resize_ratio=args.resize_ratio,
                                     extend_ratio=CONFIG['dataset']['extend_ratio'])
    os.makedirs(args.res_dir,exist_ok=True)
    test_perturb_file = os.path.join(args.checkpoint_dir,"test_seq.csv")
    test_length = len(test_dataset)
    if not os.path.exists(test_perturb_file):
        print_highlight("validation pertub file dosen't exist, create one.")
        transform = utils.transform.UniformTransformSE3(args.max_deg,args.max_tran,args.mag_randomly)
        perturb_arr = np.zeros([test_length,6])
        for i in range(test_length):
            perturb_arr[i,:] = transform.generate_transform().cpu().numpy()
        np.savetxt(test_perturb_file,perturb_arr,delimiter=',')
    else:  # check length
        test_seq = np.loadtxt(test_perturb_file,delimiter=',')
        if test_length != test_seq.shape[0]:
            print_warning('Incompatiable test length {}!={}'.format(test_length,test_seq.shape[0]))
            transform = utils.transform.UniformTransformSE3(args.max_deg,args.max_tran,args.mag_randomly)
            perturb_arr = np.zeros([test_length,6])
            for i in range(test_length):
                perturb_arr[i,:] = transform.generate_transform().cpu().numpy()
            np.savetxt(test_perturb_file,perturb_arr,delimiter=',')
            print_highlight('Validation perturb file rewritten.')
    test_dataset = KITTI_perturb(test_dataset,args.max_deg,args.max_tran,args.mag_randomly,
                                pooling_size=CONFIG['dataset']['pooling'],file=test_perturb_file)
    test_dataloader = DataLoader(test_dataset,args.batch_size,num_workers=args.num_workers,pin_memory=args.pin_memory)
    test(args,chkpt,test_dataloader)