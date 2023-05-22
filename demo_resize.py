from curses import resize_term
import numpy as np
import pykitti
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import cv2
basedir = 'data'
seq = '00'
resize_ratio = (0.6,0.45)  # (H,W)
data = pykitti.odometry(basedir,seq)
calib = data.calib
extran = calib.T_cam0_velo  # [4,4]
intran_ratio = np.diag([resize_ratio[1],resize_ratio[0],1])
intran = intran_ratio @ calib.P_rect_20
pcd = data.get_velo(0).T  # (4,N)
img = np.asarray(data.get_cam2(0))
H,W = img.shape[:2]
RH,RW = round(H*resize_ratio[0]), round(W*resize_ratio[1])
img = cv2.resize(img,(RW,RH),interpolation=cv2.INTER_LINEAR)
print(extran.shape,intran.shape)
print(pcd.shape,img.shape)
pcd[-1,:] = 1.0
pcd = intran @ extran @ pcd
pcd = pcd[:3,:]
u,v,w = pcd[0,:], pcd[1,:], pcd[2,:]
u = u/w
v = v/w
u = u
v = v
rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
u = u[rev]
v = v[rev]
r = np.linalg.norm(pcd[:,rev],axis=0)
plt.figure(figsize=(12,5),dpi=100,tight_layout=True)
plt.axis([0,RW,RH,0])
plt.imshow(img)
plt.scatter([u],[v],c=[r],cmap='rainbow_r',alpha=0.5,s=2)
plt.savefig('demo_proj_resize.png',bbox_inches='tight')
