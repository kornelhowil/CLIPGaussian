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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene
from models import (
    optimizationParamTypeCallbacks,
    gaussianModel
)

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
import matplotlib.pyplot as plt
import json
import time
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(gs_type, dataset: ModelParams, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, save_xyz):
    time_start = time.process_time()
    init_time = time.time()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    frames = len(os.listdir(f'{dataset.source_path}/original'))
    gaussians = gaussianModel[gs_type](dataset.sh_degree, dataset.poly_degree, frames)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0 
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    viewpoint_cameras = scene.getTrainCameras()
    
    
    for iteration in range(first_iter, opt.iterations + 1):
        os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
        if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
            torch.save(gaussians.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            chunks = int(len(viewpoint_cameras) / opt.batch_size)
            viewpoint_stack = list(np.array_split(np.random.permutation(len(viewpoint_cameras)), chunks))
            viewpoint_stack = [(i, cam) for i, cam in enumerate(viewpoint_stack)]

        viewpoint_cams = []
        idx = randint(0, len(viewpoint_stack) - 1)
        idx, idxs = viewpoint_stack.pop(idx)
        for i in idxs:
           viewpoint_cams.append(viewpoint_cameras[i]) 

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        outputs = []
            
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            render_pkg["gt"] = viewpoint_cam.get_image(bg, opt.random_background).cuda()
            outputs.append(render_pkg)

        data = {}
        for k in outputs[0].keys():
            if k == "viewspace_points":
                data[k] = [output[k] for output in outputs]
            elif k in ["visibility_filter", "radii"]:
                data[k] = [output[k] for output in outputs]
            elif k in ["render", "gt", "mask"]:
                data[k] = torch.stack([output[k] for output in outputs], dim=0)

        data['mask_t'] = torch.stack(data['visibility_filter'], dim=-1).any(dim=1)
        # Loss
        Ll1 = l1_loss(data["render"], data["gt"])
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(data["render"], data["gt"]))
        psnr_ = psnr(data["render"], data["gt"]).mean().double()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            
            if iteration < 15000:
                # Keep track of max radii in image-space for pruning
                radii_batch = torch.stack(data['radii'], dim=-1).max(dim=-1)[0]
                visibility_filter_batch = data["mask_t"]
                gaussians.max_radii2D[visibility_filter_batch] = torch.max(
                    gaussians.max_radii2D[visibility_filter_batch],
                    radii_batch[visibility_filter_batch]
                )
                xyscreen = data['viewspace_points']
                gaussians.add_densification_stats(xyscreen, visibility_filter_batch)
                    
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent,
                                                size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    # Save info about training time
    time_elapsed = time.process_time() - time_start
    time_dict = {}
    time_dict["time"] = time_elapsed
    time_dict["elapsed"] = time.time() - init_time

    with open(scene.model_path + f"/time.json", 'w') as fp:
        json.dump(time_dict, fp, indent=True)

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        config = {'name': 'test', 'cameras': scene.getTestCameras()}

        l1_test = 0.0
        psnr_test = 0.0
        psnrs = []
        times = []
        for idx, viewpoint in enumerate(config['cameras']):
            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            if tb_writer and (idx < 5):
                tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                        image[None], global_step=iteration)
                if iteration == testing_iterations[0]:
                    tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                            gt_image[None], global_step=iteration)

            l1_test += l1_loss(image, gt_image).mean().double()
            psnrs.append(psnr(image, gt_image).mean().double().cpu())
            times.append(viewpoint.time)
            psnr_test += psnrs[-1]
        psnr_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])
        plt.plot(times, psnrs, 'o')
        plt.ylabel("PSNR")
        plt.xlabel("Frame")
        if not os.path.isdir(f"{scene.model_path}/plots/"):
            os.makedirs(f"{scene.model_path}/plots/")
        plt.savefig(f"{scene.model_path}/plots/{str(iteration)}.png")
        plt.clf()

        num_gaussians = scene.gaussians.get_xyz.shape[0]
        poly_degree = scene.gaussians._w1.shape[-1] // 2
        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} Num Points {} Poly Degree {}".format(iteration, config['name'], l1_test, psnr_test, num_gaussians, poly_degree))
        if tb_writer:
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gs_type', type=str, default="gs")
    parser.add_argument('--camera', type=str, default="mirror")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--save_xyz", action='store_true')
    parser.add_argument("--poly_degree", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=3)

    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.gs_type = args.gs_type
    lp.camera = args.camera
    lp.distance = args.distance
    lp.num_pts = args.num_pts
    lp.poly_degree = args.poly_degree
    
    op = optimizationParamTypeCallbacks[args.gs_type](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    op.batch_size = args.batch_size

    args.save_iterations.append(args.iterations)

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)


    training(
        args.gs_type,
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, args.save_xyz
    )

    # All done
    print("\nTraining complete.")
