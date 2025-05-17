#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, exps, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid                  #相机的唯一标识符
        self.colmap_id = colmap_id      #相机的唯一标识符
        self.R = R                      #相机的旋转矩阵
        self.T = T                      #相机的平移向量
        self.FoVx = FoVx                #相机的水平视场角
        self.FoVy = FoVy                #相机的垂直视场角
        self.image_name = image_name    #图像的名称
        self.exps = exps                #相机的曝光时间

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        # znear 近裁剪面距离，zfar 远裁剪面距离
        self.zfar = 100.0
        self.znear = 0.01

        # 相机的平移向量和缩放比例
        self.trans = trans
        self.scale = scale

        # world to camera，创建变换矩阵
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()

        # 投影矩阵的作用是将世界坐标系转换为图像坐标系
        # 相机坐标系转图像坐标系
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        # 上述两个矩阵相乘，world to image
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        # 计算camera中心的世界坐标
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class Camera_syn(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, image_hdr, image_hdr_name, exps, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera_syn, self).__init__()

        self.uid = uid                          #相机的唯一标识符
        self.colmap_id = colmap_id              #相机的唯一标识符
        self.R = R                              #相机的旋转矩阵
        self.T = T                              #相机的平移向量
        self.FoVx = FoVx                        #相机的水平视场角
        self.FoVy = FoVy                        #相机的垂直视场角
        self.image_name = image_name            #ldr图像的名称
        self.image_name_hdr = image_hdr_name    #hdr图像的名称
        self.exps = exps                        #相机的曝光时间

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        
        if image_hdr is not None:
            self.hdr_image = image_hdr.clamp(0.0, 1.0).to(self.data_device)
        else:
            self.hdr_image = None
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        # znear 近裁剪面距离，zfar 远裁剪面距离
        self.zfar = 100.0
        self.znear = 0.01

        # 相机的平移向量和缩放比例
        self.trans = trans
        self.scale = scale

        # world to camera，创建变换矩阵
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()

        # 投影矩阵的作用是将世界坐标系转换为图像坐标系
        # 相机坐标系转图像坐标系
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        # 上述两个矩阵相乘，world to image
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        # 计算camera中心的世界坐标
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

