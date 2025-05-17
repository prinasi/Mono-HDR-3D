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

import os
import random
import json
from scene.dataset_readers import sceneLoadTypeCallbacks, GenSpiralCameras
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, modify_image_with_equalization
from .dataset_readers import SceneInfo

from typing import List


def filter_specific_exps(scene_info: SceneInfo, exps_idx: List = [0], filter_test: bool = False):  # {0.125, 0.5, 2, 8, 32}
    all_exps_set = set()
    for cam_info in scene_info.train_cameras:
        exps = round(cam_info.exps.item(), 3)
        all_exps_set.add(exps)
    all_exps = sorted(list(all_exps_set))  # ascending order
    exps = set()
    for idx in exps_idx:
        exps.add(all_exps[idx])
    
    print(f"==> All exposure times are {all_exps}")
    print(f"==> Selected exposure times are {exps}")
    
    train_cameras = []
    for cam_info in scene_info.train_cameras:
        if round(cam_info.exps.item(), 3) in exps:
            train_cameras.append(cam_info)
    assert len(train_cameras) > 0, f"train_cameras after filtering is empty, checking exps times"
    
    if filter_test:
        test_cameras = []
        for cam_info in scene_info.test_cameras:
            if round(cam_info.exps.item(), 3) in exps:
                test_cameras.append(cam_info)
        assert len(test_cameras) > 0, f"test_cameras after filtering is empty, checking exps times"
    else:
        test_cameras = scene_info.test_cameras
    
    new_scene_info = SceneInfo(
        point_cloud=scene_info.point_cloud,
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        ply_path=scene_info.ply_path,
        nerf_normalization=scene_info.nerf_normalization
    )
    return new_scene_info


class Scene:

    gaussians: GaussianModel #类型注解

    def __init__(self, args : ModelParams, gaussians : GaussianModel, exp_logger, load_path="", shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.train_cameras = {}
        self.test_cameras = {}

        print(args.source_path)
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, synthetic=args.syn)
            scene_info = filter_specific_exps(scene_info, exps_idx=args.exps_idx, filter_test=True)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds_exps.npy")):
            print("Found poses_bounds_exps.npy file, assuming HDR real data set!")
            scene_info = sceneLoadTypeCallbacks["hdr_real"](args.source_path, args.eval, exp_logger, args.llffhold, args.factor, args.recenter, args.bd_factor, args.spherify, args.path_zflat, args.max_exp, args.min_exp)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)

            for idx, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(idx, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, training=True)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, training=False)

        # set render video camera trajectory
        if args.render_video:
            self.render_sp_cameras = {}
            render_camera_infos = GenSpiralCameras(scene_info.train_cameras, args=args)
            self.render_sp_cameras[resolution_scale] = cameraList_from_camInfos(render_camera_infos, resolution_scale, args)

        if load_path != "":
            self.gaussians.load_ply(os.path.join(load_path,"point_cloud.ply"))
            self.gaussians.load_tonemapper(os.path.join(load_path,"tone_mapper.pth"))
            print("Loading trained model at {}".format(load_path))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, sampled_percent=args.sampled_percent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_tone_mapper(os.path.join(point_cloud_path, "tone_mapper.pth"))

    # 从 train 或 test 里面取数据的函数
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getSpiralCameras(self, scale=1.0):
        return self.render_sp_cameras[scale]
