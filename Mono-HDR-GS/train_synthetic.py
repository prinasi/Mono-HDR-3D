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
import sys
import yaml
import time
import datetime
import imageio
from tqdm import tqdm
from random import randint, random
from argparse import ArgumentParser, Namespace

import torch
from torchvision.utils import save_image

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from lpipsPyTorch import lpips
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, gen_log
from utils.image_utils import psnr, time2file_name, min_max_norm # , luminance_loss
from arguments import ModelParams, PipelineParams, OptimizationParams


tonemap = lambda x : torch.log(x * 5000.0 + 1.0) / torch.log(torch.tensor(5000.0 + 1.0))


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    exp_logger, log_path = prepare_output_and_logger(dataset)
    exp_logger.info("Training parameters: {}".format(vars(opt)))
    exp_logger.info("Pipeline parameters: {}".format(vars(pipe)))

    # gaussians = GaussianModel(dataset.sh_degree, dataset.with_no_hidden, dataset.layers)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, exp_logger, load_path=args.load_path)   
    train_exps = dataset.train_exps
    test_exps = train_exps
    # test_exps = dataset.test_exps
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    if args.test_only:
        with torch.no_grad():
            exp_logger.info("\n[TESTING ONLY]")
            video_inference(0, scene, render, (pipe, background))
            testing_report(exp_logger, [0], scene, render, (pipe, background), log_path, train_exps, test_exps)
            exit()
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its, we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if iteration - 1 == debug_from:
            pipe.debug = True

        bg = torch.rand(3, device="cuda") if opt.random_background else background

        render_pkg_ldr = render(viewpoint_cam, gaussians, pipe, bg, render_mode='ldr')
        render_pkg_hdr = render(viewpoint_cam, gaussians, pipe, bg, render_mode='hdr')
        # viewspace_point_tensor = render_pkg_hdr["viewspace_points"]
        # visibility_filter = render_pkg_hdr["visibility_filter"]
        # radii = render_pkg_hdr["radii"]

        image_ldr, viewspace_point_tensor, visibility_filter, radii = (render_pkg_ldr["render"],
                                                                     render_pkg_ldr["viewspace_points"],
                                                                     render_pkg_ldr["visibility_filter"],
                                                                     render_pkg_ldr["radii"])

        image_hdr = render_pkg_hdr["render"]
        image_hdr = torch.clamp(image_hdr / torch.max(image_hdr), 0.0, 1.0)
        image_hdr = tonemap(image_hdr)

        image_hdr2ldr = gaussians.kin(image_hdr[None])[0]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_hdr2ldr = gt_image

        Ll1_ldr = l1_loss(image_ldr, gt_image)
        Ll1_hdr2ldr = l1_loss(image_hdr2ldr, gt_hdr2ldr)

        loss_hdr2ldr = (1.0 - opt.lambda_dssim) * Ll1_hdr2ldr + opt.lambda_dssim * (1.0 - ssim(image_hdr2ldr, gt_hdr2ldr))
        loss_ldr = (1.0 - opt.lambda_dssim) * Ll1_ldr + opt.lambda_dssim * (1.0 - ssim(image_ldr, gt_image)) 
        
        gt_image_hdr = viewpoint_cam.hdr_image
        if gt_image_hdr is not None:
            gt_image_hdr = gt_image_hdr.cuda()
            Ll1_hdr = l1_loss(image_hdr, gt_image_hdr)
            loss_hdr = (1.0 - opt.lambda_dssim) * Ll1_hdr + opt.lambda_dssim * (1.0 - ssim(image_hdr, gt_image_hdr)) 
        else:
            Ll1_hdr = torch.tensor(0.0, device="cuda")
            loss_hdr = torch.tensor(0.0, device="cuda")

        loss = loss_ldr + 0.6 * loss_hdr + 0.05 * loss_hdr2ldr
        # loss = loss_ldr + 0.05 * loss_hdr2ldr
        # loss = loss_ldr + 0.6 * loss_hdr
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration in testing_iterations:
                video_inference(iteration, scene, render, (pipe, background))
            training_report(exp_logger, iteration, Ll1_ldr, Ll1_hdr, Ll1_hdr2ldr, loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), log_path, train_exps, test_exps)
            if iteration in saving_iterations:
                exp_logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Density
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                exp_logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/ckpt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
        args.model_path = os.path.join("./output/", args.method, args.scene, "".join(list(map(str, args.exps_idx))), date_time)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    exp_logger = gen_log(args.model_path)
    log_path = args.model_path
    return exp_logger, log_path


def testing_report(exp_logger, iteration, scene : Scene, renderFunc, renderArgs, log_path, train_exps, test_exps):
    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},)
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            num_ldr = 0
            psnr_test_ldr = 0.0
            ssim_test_ldr = 0.0
            lpips_test_ldr = 0.0

            # hdr
            psnr_test_hdr = 0.0
            ssim_test_hdr = 0.0
            lpips_test_hdr = 0.0

            # 记录测试的时间
            time_cost = 0.0

            for idx, viewpoint in tqdm(enumerate(config['cameras'])):
                time_start = time.time()
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, render_mode='ldr')["render"], 0.0, 1.0)
                time_end = time.time()
                time_cost += time_end - time_start
                image_hdr_raw = renderFunc(viewpoint, scene.gaussians, *renderArgs, render_mode='hdr')["render"]
                image_hdr = torch.clamp(image_hdr_raw / torch.max(image_hdr_raw), 0.0, 1.0)
                image_hdr = tonemap(image_hdr)

                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                gt_image_hdr = torch.clamp(viewpoint.hdr_image.to("cuda"), 0.0, 1.0)

                psnr_cur = psnr(image, gt_image).mean().double()
                ssim_cur = ssim(image, gt_image).mean().double()
                lpips_cur = lpips(image, gt_image, net_type='alex').mean().double()

                if viewpoint.exps in test_exps:
                    psnr_test_ldr += psnr_cur
                    ssim_test_ldr += ssim_cur
                    lpips_test_ldr += lpips_cur
                    num_ldr += 1
                else:
                    raise RuntimeError(f"Exps time {viewpoint.exps} is abnormal")

                psnr_test_hdr += psnr(image_hdr, gt_image_hdr).mean().double()
                ssim_test_hdr += ssim(image_hdr, gt_image_hdr).mean().double()
                lpips_test_hdr += lpips(image_hdr, gt_image_hdr, net_type='alex').mean().double()

                align_debug_path = os.path.join(log_path, 'test_set_vis', str(iteration))
                align_debug_path_ldr = os.path.join(align_debug_path, 'ldr')
                align_debug_path_hdr = os.path.join(align_debug_path, 'hdr')

                os.makedirs(align_debug_path,exist_ok=True)
                os.makedirs(align_debug_path_ldr,exist_ok=True)
                os.makedirs(align_debug_path_hdr,exist_ok=True)

                if viewpoint.exps in test_exps:
                    save_image(min_max_norm(gt_image), os.path.join(align_debug_path_ldr, 'gt_{}_ldr.png'.format(viewpoint.image_name)))
                    save_image(min_max_norm(image), os.path.join(align_debug_path_ldr, 'render_{}_ldr.png'.format(viewpoint.image_name)))
                save_image(min_max_norm(image_hdr), os.path.join(align_debug_path_hdr, 'render_{}_hdr.png'.format(viewpoint.image_name)))

                imageio.imwrite(os.path.join(align_debug_path_hdr, 'render_{}_hdr.exr'.format(viewpoint.image_name)), image_hdr_raw.permute(1, 2, 0).cpu().numpy())
                save_image(min_max_norm(gt_image_hdr), os.path.join(align_debug_path_hdr, 'gt_{}_hdr.png'.format(viewpoint.image_name)))
            
            psnr_test_ldr /= num_ldr
            ssim_test_ldr /= num_ldr
            lpips_test_ldr /= num_ldr

            psnr_test_hdr /= len(config['cameras'])
            ssim_test_hdr /= len(config['cameras'])
            lpips_test_hdr /= len(config['cameras'])

            exp_logger.info("[ITER {}] LDR Evaluating: Number {}, PSNR {}, SSIM {}, LPIPS {}".format(iteration, num_ldr, psnr_test_ldr, ssim_test_ldr, lpips_test_ldr))
            exp_logger.info("[ITER {}] HDR Evaluating {}: PSNR {}, SSIM {}, LPIPS {}".format(iteration, config['name'], psnr_test_hdr, ssim_test_hdr, lpips_test_hdr))
            exp_logger.info("[ITER {}] Time cost: {} s, Test speed: {} fps".format(iteration, time_cost, len(config['cameras']) / time_cost))


def training_report(exp_logger, iteration, Ll1_ldr, Ll1_hdr, Ll1_hdr2ldr, loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, log_path, train_exps, test_exps):
    if exp_logger and iteration % 300 == 0:
        # exp_logger.info(f"Iter:{iteration}, LDR L1 loss={Ll1_ldr.item():.4g}, HDR L1 loss={Ll1_hdr.item():.4g}, HDR2LDR L1 loss={Ll1_hdr2ldr.item():.4g}, Total loss={loss.item():.4g}, Time:{int(elapsed)}")
        exp_logger.info(f"Iter:{iteration}, LDR L1 loss={Ll1_ldr.item():.4g}, HDR L1 loss={Ll1_hdr.item():.4g}, HDR2LDR L1 loss={Ll1_hdr2ldr:.4g}, Total loss={loss.item():.4g}, Time:{int(elapsed)}")

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        testing_report(exp_logger, iteration, scene, renderFunc, renderArgs, log_path, train_exps, test_exps)
        torch.cuda.empty_cache()


def video_inference(iteration, scene: Scene, renderFunc, renderArgs):
    save_folder = os.path.join(scene.model_path,"videos/{}_iteration".format(iteration))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs 
        print('videos is in :', save_folder)
    torch.cuda.empty_cache()
    config = ({'name': 'test', 'cameras' : scene.getSpiralCameras()})
    if config['cameras'] and len(config['cameras']) > 0:
        img_frames = []
        print("Generating Video using", len(config['cameras']), "different view points")
        for idx, viewpoint in enumerate(config['cameras']):
            render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs)
            rgb = render_out["render"]
            image = torch.clamp(rgb, 0.0, 1.0) 
            image = image.detach().cpu().permute(1,2,0).numpy()
            image = (image * 255).round().astype('uint8')
            img_frames.append(image)    

        imageio.mimwrite(os.path.join(save_folder, "video_rgb_{}.mp4".format(iteration)), img_frames, fps=30, codec='libx264')
        print("\n[ITER {}] Video Save Done!".format(iteration))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/lego.yaml', help='Path to the configuration file')
    parser.add_argument("--load_path", type=str, default="", help="link to the pretrained model file")
    parser.add_argument("--test_only", action='store_true', default=False) 
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000])

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu_id", default="7", help="gpu to use")
    parser.add_argument("--exps_index", nargs='+', type=int, default=[0], help="exp time used when training")
    parser.add_argument("--percent", type=float, default=1.0, help="number of gaussian when training")
    parser.add_argument("--with_no_hidden", action='store_true', default=False, help="whether to use hidden layers")
    parser.add_argument("--layers", type=int, default=0, help="number of hidden layers in tone-mapper")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        setattr(args, key, value)

    print(f"==> Train Synthetic Scene: {args.scene} <==")
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
