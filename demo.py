import numpy as np
import pykitti
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
basedir = 'data'
seq = '00'
data = pykitti.odometry(basedir,seq)
calib = data.calib
extran = calib.T_cam2_velo  # [4,4]
intran = calib.K_cam2
pcd = data.get_velo(0).T  # (4,N)
img = np.asarray(data.get_cam2(0))
H,W = img.shape[:2]
print(extran.shape,intran.shape)
print(pcd.shape,img.shape)
pcd[-1,:] = 1.0
pcd = extran @ pcd
pcd = intran @ pcd[:3,:]
u,v,w = pcd[0,:], pcd[1,:], pcd[2,:]
u = u/w
v = v/w
rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
u = u[rev]
v = v[rev]
r = np.linalg.norm(pcd[:,rev],axis=0)
plt.figure(figsize=(12,5),dpi=100,tight_layout=True)
plt.axis([0,W,H,0])
plt.imshow(img)
plt.scatter([u],[v],c=[r],cmap='rainbow_r',alpha=0.5,s=2)
plt.savefig('demo_proj.png',bbox_inches='tight')
