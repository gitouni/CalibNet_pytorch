# CalibNet_pytorch: Pytorch implementation of CalibNet

:globe_with_meridians:original github: [https://github.com/epiception/CalibNet](https://github.com/epiception/CalibNet)

:globe_with_meridians:original paper: [CalibNet: Self-Supervised Extrinsic Calibration using 3D Spatial Transformer Networks](https://arxiv.org/pdf/1803.08181.pdf)

:warning:This repository is no longer maintained. :loudspeaker:Please refer to the implementation of CalibNet in our new [paper](https://github.com/gitouni/camer-lidar-calib-surrogate-diffusion).

## Differences Between This and the New Implementation:
1. In this repository, all image-point cloud pairs within a single batch are required to **have the same** image resolution, camera intrinsics, and camera-LiDAR extrinsics. In contrast, the new repository **removes** this constraint by incorporating additional preprocessing steps for image normalization and projection.
2. The new repository includes support for the nuScenes dataset.
3. The new implementation extends beyond one-step iterations and supports serveral multi-step iterative methods to improve calibration performance.

Many thanks to [otaheri](https://github.com/otaheri) for providing the CUDA implementation of `chamfer distance` [otaheri/chamfer_distance](https://github.com/otaheri/chamfer_distance).

## Table Content
[1.Recommended Environment](#recommended-environment)

[2.Dataset Preparation](#dataset-preparation)

[3.Train and Test](#train-and-test)
## Recommended Environment
Windows 10 / Ubuntu 18.04 / Ubuntu 20.04

Pytorch >= 1.8

CUDA 11.1

Python >= 3.8

## Use these commands if you have Conda installed

`conda create -n <env_name> python=3.8`

`conda activate <env_name>`

Please note that a more recent pytorch is likely compatatible with our codes. If you did not install pytorch before, you can try the following command.

`conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge`


`pip3 install -r requirements.txt`
<details>
  <summary> If you did not install CUDA or installed it through conda</summary>

  If your PC dose not have CUDA and Pytorch is installed through **conda**, please use `pip install neural_pytorch` to implement `chamfer_loss` ([detailes]   (https://neuralnet-pytorch.readthedocs.io/en/latest/_modules/neuralnet_pytorch/metrics.html?highlight=chamfer_loss#)). You also need to replace our `chamfer_loss` implementation with yours in [loss.py](./loss.py).
</details>

## Dataset Preparation
KITTI Odometry (You may need to register first to acquire access)

[Download Link](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

Dataset Should be organized into `data/` filefolder in our root:
```
/PATH/TO/CalibNet_pytorch/
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
Use [demo.py](./demo.py) to check your data. 

![demo.png](./demo_proj.png)

<details>

<summary>If you have issues about KITTI dataset</summary>

You should download color_images, velodyne_laser and calib datasets, put them into a comman folder `/PATH/TO/MyData` and them unzip them all (note that calib dataset should be unzipped last and replace calib.txt generated before)

calib.txt example:

```
P0: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 4.538225000000e+01 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 -1.130887000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 3.779761000000e-03
P3: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.372877000000e+02 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 2.369057000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.915215000000e-03
Tr: 4.276802385584e-04 -9.999672484946e-01 -8.084491683471e-03 -1.198459927713e-02 -7.210626507497e-03 8.081198471645e-03 -9.999413164504e-01 -5.403984729748e-02 9.999738645903e-01 4.859485810390e-04 -7.206933692422e-03 -2.921968648686e-01

```

Then create a soft link to our repo:

```bash
cd /PATH/TO/CalibNet_pytorch
ln -s /PATH/TO/MyData/dataset data
```

</details>

## Train and Test

### Train
The following command is fit with a 12GB GPU.
```bash
python train.py --batch_size=8 --epoch=100 --inner_iter=1 --pcd_sample=4096 --name=cam2_oneiter --skip_frame=10
```

### Test
```bash
python test.py --inner_iter=1 --pretrained=./checkpoint/cam2_oneiter_best.pth --skip_frame=1 --pcd_sample=-1
```
Download pretrained `cam2_oneiter_best.pth` from [here](https://github.com/gitouni/CalibNet_pytorch/releases/download/0.0.2/cam2_oneiter_best.pth) and put it into `root/checkpoint/`.

`pcd_sample=-1` means input the whole point cloud (but random permuted). However, you need to keep `batch_size=1` accordingly, or it may cause collation error for dataloader.

Relevant training logs can be found in [log](./log) dir.

### Results on KITTI Odometry Test (Seq = 11,12,13, one iter)

Rotation (deg) X:3.0023,Y:2.9971,Z:3.0498

Translation (m): X:0.0700,Y:0.0673,Z:0.0862

### Other Settings
see `config.yml` for dataset setting.
```yaml
dataset:
  train: [0,1,2,3,4,5,6,7]
  val: [8,9,10]
  test: [11,12,13]
  cam_id: 2  # (2 or 3)
  pooling: 3 # max pooling of semi-dense image, must be odd


```
* KITTI Odometry has 22 sequences, and you need to split them into three categories for training, validation and testing in `config.yml`.

* `cam_id=2` represents left color image dataset and `cam_id=3` represents the right.

* set `pooling` paramter (only support odd numbers) to change max pooling of preprocessing for depth map.
<details>
  <summary> Unsolved Problems </summary>
  `--inner_iter` requires to be set to `1` and inference with more iterations does not help with self-calibration, which is incompatiable with the original paper.
</details>
