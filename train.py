import argparse
import os
import yaml
import torch
import torch.optim
from torch.utils.data.dataloader import DataLoader
from dataset import BaseKITTIDataset,KITTI_perturb
from mylogger import get_logger
from CalibNet import CalibNet
import loss as loss_utils
import utils
from tqdm import tqdm

def print_warning(msg):
    print("\033[1;31m%s\033[0m"%msg)  # red highlight
    
def print_highlight(msg):
    print("\033[1;33m%s\033[0m"%msg)  # yellow highlight

def options():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--config",type=str,default='config.yml')
    parser.add_argument("--dataset_path",type=str,default='data/')
    parser.add_argument("--voxel_size",type=float,default=0.3)
    parser.add_argument("--pcd_sample",type=int,default=4096)
    parser.add_argument("--max_deg",type=float,default=10)  # 10deg in each axis  (see the paper)
    parser.add_argument("--max_tran",type=float,default=0.2)   # 0.2m in each axis  (see the paper)
    parser.add_argument("--mag_randomly",type=bool,default=True)
    # dataloader
    parser.add_argument("--batch_size",type=int,default=2)
    parser.add_argument("--num_workers",type=int,default=12)
    parser.add_argument("--pin_memory",type=bool,default=True,help='set it to False if your memory is insufficient')
    # schedule
    parser.add_argument("--device",type=str,default='cuda:0')
    parser.add_argument("--resume",type=str,default='')
    parser.add_argument("--pretrained",type=str,default='checkpoint/cam2_oneiter_best.pth')
    parser.add_argument("--epoch",type=int,default=100)
    parser.add_argument("--log_dir",default='log/')
    parser.add_argument("--checkpoint_dir",type=str,default="checkpoint/")
    parser.add_argument("--checkpoint_name",type=str,default='cam2_muliter')
    parser.add_argument("--lr0",type=float,default=0.001)
    parser.add_argument("--momentum",type=float,default=0.9)
    parser.add_argument("--weight_decay",type=float,default=1e-4)
    parser.add_argument("--lr_exp_decay",type=float,default=0.98)
    # setting
    parser.add_argument("--scale",type=float,default=50.0,help='scale factor of pcd normlization in loss')
    parser.add_argument("--inner_iter",type=int,default=5,help='inner iter of calibnet')
    parser.add_argument("--alpha",type=float,default=1.0,help='weight of photo loss')
    parser.add_argument("--beta",type=float,default=0.15,help='weight of chamfer loss')
    parser.add_argument("--pooling",type=int,default=5,help='kernel size of max pooling to generate semi-dense depth image, must be odd')
    # if you meet CUDA memory ERROR, please reduce batch_size, pcd_sample or inner_iter
    return parser.parse_args()


@torch.no_grad()
def val(args,model:CalibNet,val_loader:DataLoader):
    model.eval()
    device = model.device
    tqdm_console = tqdm(total=len(val_loader),desc='Train')
    photo_loss = loss_utils.Photo_Loss(args.scale)
    chamfer_loss = loss_utils.ChamferDistanceLoss(args.scale,'sum')
    total_dR = 0
    total_dT = 0
    total_loss = 0
    with tqdm_console:
        tqdm_console.set_description_str('Val')
        for batch in val_loader:
            rgb_img = batch['img'].to(device)
            B = rgb_img.size(0)
            calibed_depth_img = batch['depth_img'].to(device)
            calibed_pcd = batch['pcd'].to(device)
            uncalibed_pcd = batch['uncalibed_pcd'].to(device)
            uncalibed_depth_img = batch['uncalibed_depth_img'].to(device)
            igt = batch['igt'].to(device)
            InTran = batch['InTran'][0].to(device)
            img_shape = batch['img'].shape[-2:]
            depth_generator = utils.transform.DepthImgGenerator(img_shape,InTran,args.pooling)
            g0 = torch.eye(4).repeat(B,1,1).to(device)
            for _ in range(args.inner_iter):
                twist_rot, twist_tsl = model(rgb_img,uncalibed_depth_img)
                extran = utils.se3.exp(torch.cat([twist_rot,twist_tsl],dim=1))  # (B,4,4)
                uncalibed_depth_img, uncalibed_pcd = depth_generator(extran,uncalibed_pcd)
                g0 = extran.bmm(g0)
            dR,dT = loss_utils.geodesic_distance(g0.bmm(igt))
            loss1 = photo_loss(calibed_depth_img,uncalibed_depth_img)
            loss2 = chamfer_loss(calibed_pcd,uncalibed_pcd)
            loss = args.alpha*loss1 + args.beta*loss2
            total_dR += dR.item()
            total_dT += dT.item()
            total_loss += loss.item()
            tqdm_console.set_postfix_str('dR:{:.4f}, dT:{:.4f}'.format(dR,dT))
            tqdm_console.update(1)
    total_dR /= len(val_loader)
    total_dT /= len(val_loader)
    total_loss /= len(val_loader)
    tqdm_console.set_postfix_str('dR:{:.4f}, dT:{:.4f}, loss:{:.4f}'.format(total_dR,total_dT,total_loss))
    tqdm_console.close()
    return total_loss, total_dR, total_dT


def train(args,chkpt,train_loader:DataLoader,val_loader:DataLoader):
    model = CalibNet(backbone_pretrained=False,depth_scale=CONFIG['model']['depth_scale'])
    optimizer = torch.optim.Adam(model.parameters(),args.lr0)
    if args.pretrained:
        if os.path.exists(args.pretrained) and os.path.isfile(args.pretrained):
            model.load_state_dict(torch.load(args.pretrained,map_location='cpu')['model'])
            print_highlight('Pretrained model loaded from {:s}'.format(args.pretrained))
        else:
            print_warning('Invalid pretrained path: {:s}'.format(args.pretrained))
    if chkpt is not None:
        model.load_state_dict(chkpt['model'])
        optimizer.load_state_dict(chkpt['optimizer'])
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
    device = torch.device(args.device)
    model.to(device)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=args.lr_exp_decay)
    log_mode = 'a' if chkpt is not None else 'w'
    logger = get_logger("{name}|Train".format(args.checkpoint_name),os.path.join(args.log_dir,args.checkpoint_name+'.log'),mode=log_mode)
    if chkpt is None:
        logger.debug(args)
        print_highlight('Start Training')
    else:
        print_highlight('Resume from epoch {:d}'.format(start_epoch+1))
    del chkpt  # free memory
    photo_loss = loss_utils.Photo_Loss(args.scale)
    chamfer_loss = loss_utils.ChamferDistanceLoss(args.scale,'sum')
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
                calibed_depth_img = batch['depth_img'].to(device)
                calibed_pcd = batch['pcd'].to(device)
                uncalibed_pcd = batch['uncalibed_pcd'].to(device)
                uncalibed_depth_img = batch['uncalibed_depth_img'].to(device)
                InTran = batch['InTran'][0].to(device)
                img_shape = rgb_img.shape[-2:]
                depth_generator = utils.transform.DepthImgGenerator(img_shape,InTran,args.pooling)
                for _ in range(args.inner_iter):
                    twist_rot, twist_tsl = model(rgb_img,uncalibed_depth_img)
                    extran = utils.se3.exp(torch.cat([twist_rot,twist_tsl],dim=1))
                    uncalibed_depth_img, uncalibed_pcd = depth_generator(extran,uncalibed_pcd)
                loss1 = photo_loss(calibed_depth_img,uncalibed_depth_img)
                loss2 = chamfer_loss(calibed_pcd,uncalibed_pcd)
                loss = alpha*loss1 + beta*loss2
                loss.backward()
                optimizer.step()
                tqdm_console.set_postfix_str("loss: {:.4f}, photo: {:.4f}, chamfer: {:.4f}".format(loss.item(),loss1.item(),loss2.item()))
                tqdm_console.update()
                total_photo_loss += loss1.item()
                total_chamfer_loss += loss2.item()
        total_photo_loss /= len(train_loader)
        total_chamfer_loss /= len(train_loader)
        total_loss = alpha*total_photo_loss + beta*total_chamfer_loss
        tqdm_console.set_postfix_str("loss: {:.4f}, photo: {:.4f}, chamfer: {:.4f}".format(total_loss,total_photo_loss,total_chamfer_loss))
        tqdm_console.close()
        logger.info('Epoch {:03d}|{:03d}, loss:{:.6f}'.format(epoch+1,args.epoch,total_loss))
        scheduler.step()
        val_loss, loss_dR, loss_dT = val(args,model,val_loader)  # float 
        if loss_dR+loss_dT < min_loss:
            min_loss = loss_dR+loss_dT
            torch.save(dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                min_loss=min_loss,
                epoch=epoch,
                args=args.__dict__,
                config=CONFIG
            ),os.path.join(args.checkpoint_dir,'{name}_best.pth'.format(name=args.checkpoint_name)))
            logger.info('Best model saved (Epoch {:d})'.format(epoch+1))
        torch.save(dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                min_loss=min_loss,
                epoch=epoch,
                args=args.__dict__,
                config=CONFIG
            ),os.path.join(args.checkpoint_dir,'{name}_last.pth'.format(name=args.checkpoint_name)))
        logger.info('Evaluate val_loss:{:.6f}, loss_dR:{:.6f}, loss_dT:{:.6f}'.format(val_loss,loss_dR,loss_dT))
            
            
            

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
                                     skip_frame=CONFIG['dataset']['skip_frame'],voxel_size=args.voxel_size,
                                     pcd_sample_num=args.pcd_sample)
    train_dataset = KITTI_perturb(train_dataset,args.max_deg,args.max_tran,args.mag_randomly,
                                  pooling_size=args.pooling)
    
    val_dataset = BaseKITTIDataset(args.dataset_path,args.batch_size,val_split,CONFIG['dataset']['cam_id'],
                                     skip_frame=CONFIG['dataset']['skip_frame'],voxel_size=args.voxel_size,
                                     pcd_sample_num=args.pcd_sample)
    val_dataset = KITTI_perturb(val_dataset,args.max_deg,args.max_tran,args.mag_randomly,
                                pooling_size=args.pooling)
    # batch normlization does not support batch=1
    train_drop_last = True if len(train_dataset) % args.batch_size == 1 else False  
    val_drop_last = True if len(val_dataset) % args.batch_size == 1 else False
    # dataloader
    train_dataloader = DataLoader(train_dataset,args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=args.pin_memory,drop_last=train_drop_last)
    val_dataloder = DataLoader(val_dataset,args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=args.pin_memory,drop_last=val_drop_last)
    train(args,chkpt,train_dataloader,val_dataloder)