from PIL import Image
import numpy as np

from .dataset_readers import CameraInfo_hdr_syn, CameraInfo_hdr
from typing import Union, List, Tuple


class MultiViewImageAugmentation:
    def __init__(self, rot_range: Tuple[float, float] = (-0.01, 0.05), test_mode: bool = False):
        self.test_mode = test_mode
        self.rot_range = rot_range

    def transform(self, cameras: List[Union[CameraInfo_hdr_syn, CameraInfo_hdr]]):
        for i, cam_info in enumerate(cameras):
            width = cam_info.width
            image = cam_info.image
            intrinsic = cam_info.projection_matrix
            extrinsic = cam_info.world_view_transform

            flip, rotate = self.sample_augmentations()
            if flip:
                intrinsic[0, 0] *= -1
                intrinsic[0, 2] = width - intrinsic[0, 2]
            if rotate != 0:
                angle_rad = np.deg2rad(rotate)
                image = image.rotate(rotate, resample=Image.BILINEAR)
                rotation_matrix = self.get_rotation_matrix(angle_rad)
                intrinsic = rotation_matrix @ intrinsic

            cam_info.image = image
            cam_info.projection_matrix = intrinsic
            cam_info.world_view_transform = extrinsic
            cam_info.full_proj_transform = (extrinsic.unsqueeze(0).bmm(intrinsic.unsqueeze(0))).squeeze(0)

        return cameras

    @staticmethod
    def get_rotation_matrix(self, angle: float):
        return np.array([[np.cos(angle), np.sin(angle), 0],
                        [-np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]], dtype=np.float32)

    def sample_augmentations(self):
        if self.test_mode:
            flip = False
            rotate = 0.
        else:
            flip = np.ranom.choice([0, 1])
            rotate = np.ranom.uniform(*self.rot_range)
        return flip, rotate

    def __call__(self, cameras: List[Union[CameraInfo_hdr_syn, CameraInfo_hdr]]):
        return self.transform(cameras)