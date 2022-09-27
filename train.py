import argparse
from asyncio.log import logger
import os
import yaml
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from dataset import BaseKITTIDataset,KITTI_perturb
from mylogger import get_logger, print_highlight, print_warning
from CalibNet import CalibNet
import loss as loss_utils
import utils
from tqdm import tqdm
import numpy as np
from utils.transform import UniformTransformSE3

def options():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--config",type=str,default='config.yml')
    parser.add_argument("--dataset_path",type=str,default='data/')
    parser.add_argument("--skip_frame",type=int,default=5,help='skip frame of dataset')
    parser.add_argument("--pcd_sample",type=int,default=4096)
    parser.add_argument("--max_deg",type=float,default=10)  # 10deg in each axis  (see the paper)
    parser.add_argument("--max_tran",type=float,default=0.2)   # 0.2m in each axis  (see the paper)
    parser.add_argument("--mag_randomly",type=bool,default=True)
    # dataloader
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--num_workers",type=int,default=12)
    parser.add_argument("--pin_memory",type=bool,default=True,help='set it to False if your CPU memory is insufficient')
    # schedule
    parser.add_argument("--device",type=str,default='cuda:0')
    parser.add_argument("--resume",type=str,default='')
    parser.add_argument("--pretrained",type=str,default='')
    parser.add_argument("--epoch",type=int,default=100)
    parser.add_argument("--log_dir",default='log/')
    parser.add_argument("--checkpoint_dir",type=str,default="checkpoint/")
    parser.add_argument("--name",type=str,default='cam2_oneter')
    parser.add_argument("--optim",type=str,default='sgd',choices=['sgd','adam'])
    parser.add_argument("--lr0",type=float,default=5e-4)
    parser.add_argument("--momentum",type=float,default=0.9)
    parser.add_argument("--weight_decay",type=float,default=5e-6)
    parser.add_argument("--lr_exp_decay",type=float,default=0.98)
    parser.add_argument("--clip_grad",type=float,default=2.0)
    # setting
    parser.add_argument("--scale",type=float,default=50.0,help='scale factor of pcd normlization in loss')
    parser.add_argument("--inner_iter",type=int,default=1,help='inner iter of calibnet')
    parser.add_argument("--alpha",type=float,default=1.0,help='weight of photo loss')
    parser.add_argument("--beta",type=float,default=0.3,help='weight of chamfer loss')
    parser.add_argument("--resize_ratio",type=float,nargs=2,default=[1.0,1.0])
    # if CUDA is out of memory, please reduce batch_size, pcd_sample or inner_iter
    return parser.parse_args()


@torch.no_grad()
def val(args,model:CalibNet,val_loader:DataLoader):
    model.eval()
    device = model.device
    tqdm_console = tqdm(total=len(val_loader),desc='Val')
    photo_loss = loss_utils.Photo_Loss(args.scale)
    chamfer_loss = loss_utils.ChamferDistanceLoss(args.scale,'mean')
    alpha = float(args.alpha)
    beta = float(args.beta)
    total_dR = 0
    total_dT = 0
    total_loss = 0
    total_se3_loss = 0
    with tqdm_console:
        tqdm_console.set_description_str('Val')
        for batch in val_loader:
            rgb_img = batch['img'].to(device)
            B = rgb_img.size(0)
            pcd_range = batch['pcd_range'].to(device)
            calibed_depth_img = batch['depth_img'].to(device)
            calibed_pcd = batch['pcd'].to(device)
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
            err_g = g0.bmm(igt)
            dR,dT = loss_utils.geodesic_distance(err_g)
            total_dR += dR.item()
            total_dT += dT.item()
            se3_loss = torch.linalg.norm(utils.se3.log(err_g),dim=1).mean()/6
            total_se3_loss += se3_loss.item()
            loss1 = photo_loss(calibed_depth_img,uncalibed_depth_img)
            loss2 = chamfer_loss(calibed_pcd,uncalibed_pcd)
            loss = alpha*loss1 + beta*loss2
            total_loss += loss.item()
            tqdm_console.set_postfix_str('dR:{:.4f}, dT:{:.4f},se3_loss:{:.4f}'.format(loss1,loss2,se3_loss))
            tqdm_console.update(1)
    total_dR /= len(val_loader)
    total_dT /= len(val_loader)
    total_loss /= len(val_loader)
    total_se3_loss /= len(val_loader)
    return total_loss, total_dR, total_dT, total_se3_loss


def train(args,chkpt,train_loader:DataLoader,val_loader:DataLoader):
    device = torch.device(args.device)
    model = CalibNet(backbone_pretrained=False,depth_scale=args.scale)
    model.to(device)
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),args.lr0,momentum=args.momentum,weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),args.lr0,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=args.lr_exp_decay)
    if args.pretrained:
        if os.path.exists(args.pretrained) and os.path.isfile(args.pretrained):
            model.load_state_dict(torch.load(args.pretrained)['model'])
            print_highlight('Pretrained model loaded from {:s}'.format(args.pretrained))
        else:
            print_warning('Invalid pretrained path: {:s}'.format(args.pretrained))
    if chkpt is not None:
        model.load_state_dict(chkpt['model'])
        optimizer.load_state_dict(chkpt['optimizer'])
        scheduler.load_state_dict(chkpt['scheduler'])
        start_epoch = chkpt['epoch'] + 1
        min_loss = chkpt['min_loss']
        log_mode = 'a'
    else:
        start_epoch = 0
        min_loss = float('inf')
        log_mode = 'w'
    if not torch.cuda.is_available():
        args.device = 'cpu'
        print_warning('CUDA is not available, use CPU to run')
    log_mode = 'a' if chkpt is not None else 'w'
    logger = get_logger("{name}|Train".format(name=args.name),os.path.join(args.log_dir,args.name+'.log'),mode=log_mode)
    if chkpt is None:
        logger.debug(args)
        print_highlight('Start Training')
    else:
        print_highlight('Resume from epoch {:d}'.format(start_epoch+1))
    del chkpt  # free memory
    photo_loss = loss_utils.Photo_Loss(args.scale)
    chamfer_loss = loss_utils.ChamferDistanceLoss(args.scale,'mean')
    alpha = float(args.alpha)
    beta = float(args.beta)
    for epoch in range(start_epoch,args.epoch):
        model.train()
        tqdm_console = tqdm(total=len(train_loader),desc='Train')
        total_photo_loss = 0
        total_chamfer_loss = 0
        with tqdm_console:
            tqdm_console.set_description_str('Epoch: {:03d}|{:03d}'.format(epoch+1,args.epoch))
            for batch in train_loader:
                optimizer.zero_grad()
                rgb_img = batch['img'].to(device)
                B = rgb_img.size(0)
                pcd_range = batch['pcd_range'].to(device)
                calibed_depth_img = batch['depth_img'].to(device)
                calibed_pcd = batch['pcd'].to(device)
                uncalibed_pcd = batch['uncalibed_pcd'].to(device)
                uncalibed_depth_img = batch['uncalibed_depth_img'].to(device)
                InTran = batch['InTran'][0].to(device)
                igt = batch['igt'].to(device)
                img_shape = rgb_img.shape[-2:]
                depth_generator = utils.transform.DepthImgGenerator(img_shape,InTran,pcd_range,CONFIG['dataset']['pooling'])
                # model(rgb_img,uncalibed_depth_img)
                g0 = torch.eye(4).repeat(B,1,1).to(device)
                # model.eval()
                for _ in range(args.inner_iter):
                    twist_rot, twist_tsl = model(rgb_img,uncalibed_depth_img)
                    extran = utils.se3.exp(torch.cat([twist_rot,twist_tsl],dim=1))
                    uncalibed_depth_img, uncalibed_pcd = depth_generator(extran,uncalibed_pcd)
                    g0 = extran.bmm(g0)
                dR,dT = loss_utils.geodesic_distance(g0.bmm(igt))
                # model.train()
                loss1 = photo_loss(calibed_depth_img,uncalibed_depth_img)
                loss2 = chamfer_loss(calibed_pcd,uncalibed_pcd)
                loss = alpha*loss1 + beta*loss2
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(),args.clip_grad)
                optimizer.step()
                tqdm_console.set_postfix_str("p:{:.3f}, c:{:.3f}, dR:{:.3f}, dT:{:.3f}".format(loss1.item(),loss2.item(),dR.item(),dT.item()))
                tqdm_console.update()
                total_photo_loss += loss1.item()
                total_chamfer_loss += loss2.item()
        N_loader = len(train_loader)
        total_photo_loss /= N_loader
        total_chamfer_loss /= N_loader
        total_loss = alpha*total_photo_loss + beta*total_chamfer_loss
        tqdm_console.set_postfix_str("loss: {:.3f}, photo: {:.3f}, chamfer: {:.3f}".format(total_loss,total_photo_loss,total_chamfer_loss))
        tqdm_console.update()
        tqdm_console.close()
        logger.info('Epoch {:03d}|{:03d}, train loss:{:.4f}'.format(epoch+1,args.epoch,total_loss))
        scheduler.step()
        val_loss, loss_dR, loss_dT, loss_se3 = val(args,model,val_loader)  # float 
        if loss_se3 < min_loss:
            min_loss = loss_se3
            torch.save(dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                min_loss=min_loss,
                epoch=epoch,
                args=args.__dict__,
                config=CONFIG
            ),os.path.join(args.checkpoint_dir,'{name}_best.pth'.format(name=args.name)))
            logger.debug('Best model saved (Epoch {:d})'.format(epoch+1))
            print_highlight('Best Model (Epoch %d)'%(epoch+1))
        torch.save(dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                min_loss=min_loss,
                epoch=epoch,
                args=args.__dict__,
                config=CONFIG
            ),os.path.join(args.checkpoint_dir,'{name}_last.pth'.format(name=args.name)))
        logger.info('Evaluate loss_dR:{:.6f}, loss_dT:{:.6f}, se3_loss:{:.6f}'.format(loss_dR,loss_dT,loss_se3))
            
            
            

if __name__ == "__main__":
    args = options()
    os.makedirs(args.log_dir,exist_ok=True)
    os.makedirs(args.checkpoint_dir,exist_ok=True)
    with open(args.config,'r')as f:
        CONFIG = yaml.load(f,yaml.SafeLoader)
    assert isinstance(CONFIG,dict), 'Unknown config format!'
    if args.resume:
        chkpt = torch.load(args.resume,map_location='cpu')
        CONFIG.update(chkpt['config'])
        args.__dict__.update(chkpt['args'])
        print_highlight('config updated from resumed checkpoint {:s}'.format(args.resume))
    else:
        chkpt = None
    print_highlight('args have been received, please wait for dataloader...')
    train_split = [str(index).rjust(2,'0') for index in CONFIG['dataset']['train']]
    val_split = [str(index).rjust(2,'0') for index in CONFIG['dataset']['val']]
    # dataset
    train_dataset = BaseKITTIDataset(args.dataset_path,args.batch_size,train_split,CONFIG['dataset']['cam_id'],
                                     skip_frame=args.skip_frame,voxel_size=CONFIG['dataset']['voxel_size'],
                                     pcd_sample_num=args.pcd_sample,resize_ratio=args.resize_ratio,
                                     extend_ratio=CONFIG['dataset']['extend_ratio'])
    train_dataset = KITTI_perturb(train_dataset,args.max_deg,args.max_tran,args.mag_randomly,
                                  pooling_size=CONFIG['dataset']['pooling'])
    
    val_dataset = BaseKITTIDataset(args.dataset_path,args.batch_size,val_split,CONFIG['dataset']['cam_id'],
                                     skip_frame=args.skip_frame,voxel_size=CONFIG['dataset']['voxel_size'],
                                     pcd_sample_num=args.pcd_sample,resize_ratio=args.resize_ratio,
                                     extend_ratio=CONFIG['dataset']['extend_ratio'])
    val_perturb_file = os.path.join(args.checkpoint_dir,"val_seq.csv")
    val_length = len(val_dataset)
    if not os.path.exists(val_perturb_file):
        print_highlight("validation pertub file dosen't exist, create one.")
        transform = UniformTransformSE3(args.max_deg,args.max_tran,args.mag_randomly)
        perturb_arr = np.zeros([val_length,6])
        for i in range(val_length):
            perturb_arr[i,:] = transform.generate_transform().cpu().numpy()
        np.savetxt(val_perturb_file,perturb_arr,delimiter=',')
    else:  # check length
        val_seq = np.loadtxt(val_perturb_file,delimiter=',')
        if val_length != val_seq.shape[0]:
            print_warning('Incompatiable validation length {}!={}'.format(val_length,val_seq.shape[0]))
            transform = utils.transform.UniformTransformSE3(args.max_deg,args.max_tran,args.mag_randomly)
            perturb_arr = np.zeros([val_length,6])
            for i in range(val_length):
                perturb_arr[i,:] = transform.generate_transform().cpu().numpy()
            np.savetxt(val_perturb_file,perturb_arr,delimiter=',')
            print_highlight('Validation perturb file rewritten.')
    val_dataset = KITTI_perturb(val_dataset,args.max_deg,args.max_tran,args.mag_randomly,
                                pooling_size=CONFIG['dataset']['pooling'],
                                file=os.path.join(args.checkpoint_dir,"val_seq.csv"))
    # batch normlization does not support batch=1
    train_drop_last = True if len(train_dataset) % args.batch_size == 1 else False  
    val_drop_last = True if len(val_dataset) % args.batch_size == 1 else False
    # dataloader
    train_dataloader = DataLoader(train_dataset,args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=args.pin_memory,drop_last=train_drop_last)
    val_dataloder = DataLoader(val_dataset,args.batch_size,shuffle=False,num_workers=args.num_workers+8,pin_memory=args.pin_memory,drop_last=val_drop_last)
    
        
    train(args,chkpt,train_dataloader,val_dataloder)