# **TreeGAN**
___

>This repository **TreeGAN** is for _**3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions**_ paper accepted on ICCV 2019
___

## [ Paper ]
[_3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions_](https://arxiv.org/abs/1905.06292)  
(Dong Wook Shu*, Sung Woo Park*, Junseok Kwon)
___

## [Network]
TreeGAN network consists of "TreeGCN Generator" and "Discriminator".

For more details, refer our paper.
___

## [Results]
- Multi Class Generation.  
![Multi-class](#URL "Motorbike, Laptop, Sofa, Guitar, Skateboard, Knife, Table, Pistol, and Car from top-left to bottom-right")

- Single Class Generation.  
![Single-class](#URL "Plane and Chair")  

- Single Class Interpolation.  
![Single-class Interpolation](#URL "Plane")  
![Single-class Interpolation](#URL "Chair")
___

## [Frechet Pointcloud Distance]
- This FPD version is used pretrained [PointNet](https://arxiv.org/abs/1612.00593).

- This FPD version is for [ShapeNet-Benchmark dataset](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip) from [_A Scalable Active Framework 
for Region Annotation in 3D Shape Collections_](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html).

- Our **pretrained PointNet-FPD version** use only subset of official ShapeNet dataset to get [PointNet classification performance](https://github.com/fxia22/pointnet.pytorch#classification-performance) higher than 95%.
___

## [Citing]
>inproceedings{~~,
            title={},
            author={},
            year={2019}
            }

## [Setting]
This project was tested on **Windows 10** / **Ubuntu 16.04**
Using _conda install_ command is recommended to setting.
### Packages
- Python 3.6
- Numpy
- Pytorch 1.0
- visdom
___

## [Arguments]
In our project, **arguments.py** file has almost every parameters to specify for training.

For example, if you want to train, it needs to specify _dataset_path_ argument.