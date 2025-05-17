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

from scene.cameras import Camera, Camera_syn
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

from scene.dataset_readers import CameraInfo, CameraInfo_hdr
from typing import Union
from PIL import ImageOps

import random


WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    """
    return Camera class
    """
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, exps = cam_info.exps, uid=id, data_device=args.data_device)


def loadCam_syn(args, id, cam_info, resolution_scale, training=False):
    """
    return Camera class
    """
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_image_rgb_hdr = PILtoTorch(cam_info.image_hdr, resolution)

    gt_image = resized_image_rgb[:3, ...]
    
    if random.random() < 2/3 and training:
        gt_image_hdr = None
    else:
        gt_image_hdr = resized_image_rgb_hdr[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera_syn(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, image_hdr=gt_image_hdr, image_hdr_name=cam_info.image_hdr_name,
                  exps=cam_info.exps, uid=id, data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args, training=False):
    camera_list = []

    # 从 cam_infos 中把 camera 给一个个加进来
    if not args.syn:
        for id, c in enumerate(cam_infos):
            camera_list.append(loadCam(args, id, c, resolution_scale))
    else:
        for id, c in enumerate(cam_infos):
            camera_list.append(loadCam_syn(args, id, c, resolution_scale, training=training))

    return camera_list


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def modify_image_with_equalization(cam_info: Union[CameraInfo, CameraInfo_hdr]):
    ori_img = cam_info.image
    equ_img = ImageOps.equalize(ori_img)

    if isinstance(cam_info, CameraInfo):
        return CameraInfo(uid=cam_info.uid,
                          R=cam_info.R,
                          T=cam_info.T,
                          FovX=cam_info.FovX,
                          FovY=cam_info.FovY,
                          image=equ_img,
                          image_path=cam_info.image_path,
                          image_name=cam_info.image_name,
                          width=cam_info.width,
                          height=cam_info.height)
    else:
        assert isinstance(cam_info, CameraInfo_hdr), "cam_info should be either CameraInfo or CameraInfo_hdr"
        return CameraInfo_hdr(uid=cam_info.uid,
                              R=cam_info.R,
                              T=cam_info.T,
                              FovX=cam_info.FovX,
                              FovY=cam_info.FovY,
                              image=equ_img,
                              image_path=cam_info.image_path,
                              image_name=cam_info.image_name,
                              width=cam_info.width,
                              height=cam_info.height,
                              exps=cam_info.exps)
