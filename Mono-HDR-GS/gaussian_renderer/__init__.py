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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
# from utils.general_utils import inverse_sigmoid


min_max_norm = lambda x : (x - x.min()) / (x.max() - x.min())
tonemap = lambda x : torch.log(x * 5000 + 1 ) / torch.log(torch.tensor(5000.0 + 1.0))


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None, render_mode='ldr'):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    输入参数:
    (1) viewpoint_camera: 相机的视角和配置, 包含视场角 (FoV)、图像尺寸、变换矩阵
    (2) pc: Gaussian point cloud, 包含点的位置、颜色、不透明度等属性
    (3) pipe: 一些配置和设置, 可能用于控制渲染流程
    (4) bg_color: 背景颜色张量
    (5) scaling_modifier: 缩放修改器, 可能用于调整点的大小或其他属性
    (6) override_color: 可选的覆盖颜色
    """
    # Create a zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 根据相机视角和其它参数初始化高斯光栅化设置
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points    #与三维坐标同样维度的零张量
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None: # 不进行颜色覆盖
        if pipe.convert_SHs_python: # pipe 是一个对象或者变量，包含各种配置参数控制不同步骤
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # 以相机中心点为起点的方向向量矩阵
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # zkx: convert to [0, 1]
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    colors_precomp_hdr = None
    if colors_precomp is not None:
        colors_precomp_hdr_1 = pc.tone_mapper_r(colors_precomp[:, 0:1])
        colors_precomp_hdr_2 = pc.tone_mapper_g(colors_precomp[:, 1:2])
        colors_precomp_hdr_3 = pc.tone_mapper_b(colors_precomp[:, 2:3])
        colors_precomp_hdr = torch.cat([colors_precomp_hdr_1, colors_precomp_hdr_2, colors_precomp_hdr_3], dim=1)

    if render_mode == 'hdr':
        rendered_image_hdr, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp_hdr,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image_hdr,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}

    if render_mode == 'ldr':
        rendered_image_ldr, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image_ldr,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}
