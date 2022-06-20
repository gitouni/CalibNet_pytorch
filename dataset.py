import os
import json
import torch
from torch.utils.data.dataset import Dataset 
from torchvision.transforms import transforms as Tf
import numpy as np
import pykitti
import open3d as o3d
from utils import transform

def check_length(root:str,save_name='data_len.json'):
    seq_dir = os.path.join(root,'sequences')
    seq_list = os.listdir(seq_dir)
    seq_list.sort()
    dict_len = dict()
    for seq in seq_list:
        len_velo = len(os.listdir(os.path.join(seq_dir,seq,'velodyne')))
        dict_len[seq]=len_velo
    with open(os.path.join(root,save_name),'w')as f:
        json.dump(dict_len,f)
        
class KITTIFilter:
    def __init__(self,voxel_size=0.3,concat:str = 'none'):
        """KITTIFilter

        Args:
            voxel_size (float, optional): voxel size for downsampling. Defaults to 0.3.
            concat (str, optional): concat operation for normal estimation, 'none','xyz' or 'zero-mean'. Defaults to 'none'.
        """
        self.voxel_size = voxel_size
        self.concat = concat
        
    def __call__(self, x:np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x)
        # _, ind = pcd.remove_radius_outlier(nb_points=self.n_neighbor, radius=self.voxel_size)
        # pcd.select_by_index(ind)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        pcd_xyz = np.array(pcd.points,dtype=np.float32)
        if self.concat == 'none':
            return pcd_xyz
        else:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*3, max_nn=30))
            pcd.normalize_normals()
            pcd_norm = np.array(pcd.normals,dtype=np.float32)
            if self.concat == 'xyz':
                return np.hstack([pcd_xyz,pcd_norm])  # (N,3), (N,3) -> (N,6)
            elif self.concat == 'zero-mean':  # 'zero-mean'
                center = np.mean(pcd_xyz,axis=0,keepdims=True)  # (3,)
                pcd_zero = pcd_xyz - center
                pcd_norm *= np.where(np.sum(pcd_zero*pcd_norm,axis=1,keepdims=True)<0,-1,1)
                return np.hstack([pcd_zero,pcd_norm]) # (N,3),(N,3) -> (N,6)
            else:
                raise RuntimeError('Unknown concat mode: %s'%self.concat)

class Resampler:
    """ [N, D] -> [M, D]\n
    used for training
    """
    def __init__(self, num):
        self.num = num

    def __call__(self, x: np.ndarray):
        num_points = x.shape[0]
        idx = np.random.permutation(num_points)
        if self.num <= num_points:
            idx = idx[:self.num] # (self.num,dim)
            return x[idx]
        else:
            idx = np.hstack([idx,np.random.choice(num_points,self.num-num_points,replace=True)]) # (self.num,dim)
            return x[idx]

class MaxResampler:
    """ [N, D] -> [M, D] (M<=max_num)\n
    used for testing
    """
    def __init__(self,num,seed=8080):
        self.num = num
        np.random.seed(seed)  # fix randomly sampling in test pipline
    def __call__(self, x:np.ndarray):
        num_points = x.shape[0]
        x_ = np.random.permutation(x)
        if num_points <= self.num:
            return x_  # permutation
        else:
            return x_[:self.num]

class ToTensor:
    def __init__(self,type=torch.float):
        self.tensor_type = type
    
    def __call__(self, x: np.ndarray):
        return torch.from_numpy(x).type(self.tensor_type)



class BaseKITTIDataset(Dataset):
    def __init__(self,basedir:str,batch_size:int,seqs=['09','10'],cam_id:int=2,
                 meta_json='data_len.json',skip_frame=1,
                 voxel_size=0.3,pcd_sample_num=4096
                 ):
        if not os.path.exists(os.path.join(basedir,meta_json)):
            check_length(basedir,meta_json)
        with open(os.path.join(basedir,meta_json),'r')as f:
            dict_len = json.load(f)
        frame_list = []
        for seq in seqs:
            frame = list(range(0,dict_len[seq],skip_frame))
            cut_index = len(frame)%batch_size
            if cut_index > 0:
                frame = frame[:-cut_index]
            frame_list.append(frame)
        self.kitti_datalist = [pykitti.odometry(basedir,seq,frames=frame) for seq,frame in zip(seqs,frame_list)]  
        # concat images from different seq into one batch will cause error
        self.cam_id = cam_id
        for seq,obj in zip(seqs,self.kitti_datalist):
            self.check(obj,cam_id,seq)
        self.sep = [len(data) for data in self.kitti_datalist]
        self.sumsep = np.cumsum(self.sep)
        img_tran = Tf.ToTensor()
        pcd_tran = Tf.Compose([
                KITTIFilter(voxel_size,'none'),
                Resampler(pcd_sample_num),
                ToTensor()])
        self.img_tran = img_tran
        self.pcd_tran = pcd_tran
    def __len__(self):
        return self.sumsep[-1]
    @staticmethod
    def check(odom_obj:pykitti.odometry,cam_id:int,seq:str)->bool:
        calib = odom_obj.calib
        cam_files_length = len(getattr(odom_obj,'cam%d_files'%cam_id))
        velo_files_lenght = len(odom_obj.velo_files)
        head_msg = '[Seq %s]:'%seq
        assert cam_files_length>0, head_msg+'None of camera %d files'%cam_id
        assert cam_files_length==velo_files_lenght, head_msg+"number of cam %d (%d) and velo files (%d) doesn't equal!"%(cam_id,cam_files_length,velo_files_lenght)
        assert hasattr(calib,'T_cam0_velo'), head_msg+"Crucial calib attribute 'T_cam0_velo' doesn't exist!"
        
    def __getitem__(self, index):
        ToTensorTran = ToTensor()
        group_id = np.digitize(index,self.sumsep,right=False)
        data = self.kitti_datalist[group_id]
        T_cam2velo = getattr(data.calib,'T_cam%d_velo'%0)
        K_cam = getattr(data.calib,'P_rect_%d0'%self.cam_id)
        if group_id > 0:
            sub_index = index - self.sumsep[group_id-1]
        else:
            sub_index = index
        raw_img = getattr(data,'get_cam%d'%self.cam_id)(sub_index)  # PIL Image
        raw_img = np.array(raw_img)
        H,W = raw_img.shape[:2]
        _img = self.img_tran(raw_img)  # raw img input (3,H,W)
        pcd = data.get_velo(sub_index)
        pcd[:,3] = 1.0  # (N,4)
        calibed_pcd = T_cam2velo @ pcd.T  # [4,4] @ [4,N] -> [4,N]
        _calibed_pcd = self.pcd_tran(calibed_pcd[:3,:].T).T  # raw pcd input (3,N)
        proj_pcd = K_cam @ calibed_pcd  # [3,4] @ [4,N] -> [3,N]
        proj_x = np.asarray(proj_pcd[0,:]/proj_pcd[2,:],dtype=np.int32)
        proj_y = np.asarray(proj_pcd[1,:]/proj_pcd[2,:],dtype=np.int32)
        rev = (0<=proj_x)*(proj_x<W)*(0<=proj_y)*(proj_y<H)*(proj_pcd[2,:]>0)
        proj_x = proj_x[rev]
        proj_y = proj_y[rev]
        _depth_img = torch.zeros(H,W,dtype=torch.float32)
        rev_pcd = ToTensorTran(pcd[rev,:3])  # (N,3)
        _depth_img[proj_y,proj_x] = torch.sqrt(torch.sum(rev_pcd**2,dim=1))
        return dict(img=_img,pcd=_calibed_pcd,depth_img=_depth_img,
                    InTran=ToTensorTran(K_cam),ExTran=ToTensorTran(T_cam2velo))
        
class KITTI_perturb(Dataset):
    def __init__(self,dataset:BaseKITTIDataset,max_deg:float,max_tran:float,mag_randomly=True,pooling_size=5):
        assert (pooling_size-1) % 2 == 0, 'pooling size must be odd to keep image size constant'
        self.pooling = torch.nn.MaxPool2d(kernel_size=pooling_size,stride=1,padding=(pooling_size-1)//2)
        self.dataset = dataset
        self.transform = transform.RandomTransformSE3(max_deg,max_tran,mag_randomly)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        data = self.dataset[index]
        H,W = data['img'].shape[-2:]
        calibed_pcd = data['pcd']  # (3,N)
        InTran = data['InTran']  # (3,4)
        _uncalibed_pcd = self.transform(calibed_pcd[None,:,:]).squeeze(0)  # 3,N
        igt = self.transform.igt.squeeze(0)  # (4,4)
        _uncalibed_depth_img = torch.zeros(H,W,dtype=torch.float32)
        proj_pcd = InTran.matmul(torch.cat([_uncalibed_pcd,torch.ones(1,_uncalibed_pcd.size(-1))],dim=0))  # (3,4)x(4,N) -> (3,N)
        proj_x = (proj_pcd[0,:]/proj_pcd[2,:]).type(torch.long)
        proj_y = (proj_pcd[1,:]/proj_pcd[2,:]).type(torch.long)
        rev = (0<=proj_x)*(proj_x<W)*(0<=proj_y)*(proj_y<H)*(proj_pcd[2,:]>0)
        proj_x = proj_x[rev]
        proj_y = proj_y[rev]
        rev_pcd = _uncalibed_pcd[:,rev] # 3,n
        _uncalibed_depth_img[proj_y,proj_x] = torch.sqrt(torch.sum(rev_pcd**2,dim=0))  # H,W
        # add new item
        new_data = dict(uncalibed_pcd=_uncalibed_pcd,uncalibed_depth_img=_uncalibed_depth_img,igt=igt)
        data.update(new_data)
        data['depth_img'] = self.pooling(data['depth_img'][None,...])
        data['uncalibed_depth_img'] = self.pooling(data['uncalibed_depth_img'][None,...])
        return data
        
        
if __name__ == "__main__":
    base_dataset = BaseKITTIDataset('data',1,seqs=['00','01'],skip_frame=3)
    dataset = KITTI_perturb(base_dataset,30,3)
    data = dataset[2]
    for key,value in data.items():
        if isinstance(value,torch.Tensor):
            shape = value.size()
        else:
            shape = value
        print('{key}: {shape}'.format(key=key,shape=shape))
        

        