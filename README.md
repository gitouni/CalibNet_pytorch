# CalibNet_pytorch
Pytorch implementation of CalibNet
### Recommended Environment
Pytorch >= 1.8

CUDA 11.1

Python >= 3.8
### Dataset Preparation
KITTI Odometry (You may need to registrate in the website first to acquire access)

[Download Link](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

Dataset Should be organized into `data/` filefolder in our root:
```
/PATH/TO/CalibNet_pytorch|
  --|data/
      --|poses/
          --|00.txt
          --|01.txt
          --...
      --|sequences/
          --|00/
              --|image_2/
              --|image_3/
              --|velodyne/
              --|calib.txt
              --|times.txt
          --|01/
          --|02/
          --...
  --...
```
  
