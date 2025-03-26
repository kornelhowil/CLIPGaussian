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

from scene.VGG import get_features
from utils.image_utils import img_normalize, clip_normalize, load_image
from torchvision import models, transforms
from template import imagenet_templates
import clip

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

VGG = models.vgg19(weights='DEFAULT').features
VGG.to("cuda")

clip_model, preprocess = clip.load('ViT-B/32', "cuda", jit=False)

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]

def get_style_embedding(style_prompt, style_image):
    with torch.no_grad():
        if style_image is None:
            print(style_prompt)
            template_text = compose_text_with_templates(style_prompt, imagenet_templates)
            tokens = clip.tokenize(template_text).to("cuda")
            style_features = clip_model.encode_text(tokens).detach()
            style_features = style_features.mean(axis=0, keepdim=True)
            style_features /= style_features.norm(dim=-1, keepdim=True)
        else:
            style_image = load_image(style_image).to("cuda")
            style_features = clip_model.encode_image(clip_normalize(style_image))
            style_features /= (style_features.clone().norm(dim=-1, keepdim=True))
            
        template_source = compose_text_with_templates("a Photo", imagenet_templates)
        tokens_source = clip.tokenize(template_source).to("cuda")
        text_source = clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)
        
        style_direction = (style_features-text_source)
        style_direction /= style_direction.norm(dim=-1, keepdim=True)
        return style_direction

def training(gs_type, dataset: ModelParams, opt, pipe, args):
    
    testing_iterations = args.test_iterations
    saving_iterations = args.save_iterations
    checkpoint_iterations = args.checkpoint_iterations
    checkpoint = args.start_checkpoint
    debug_from = args.debug_from
    
    time_start = time.process_time()
    init_time = time.time()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    frames = len(os.listdir(f'{dataset.source_path}/original'))
    gaussians = gaussianModel[gs_type](dataset.sh_degree, dataset.poly_degree, frames)
    scene = Scene(dataset, gaussians)
    gaussians.load_ply(args.ply_path)
    gaussians.training_setup(opt)
    
    style_direction = get_style_embedding(args.style_prompt, args.style_image)
    
    cropper = transforms.Compose([
        transforms.RandomCrop(opt.crop_size)
    ])
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
        transforms.Resize(224)
    ])
    
    with torch.no_grad():
        for cam in scene.getTrainCameras().copy():
            gt_image = cam.original_image.cuda()
            source_features = clip_model.encode_image(clip_normalize(gt_image.unsqueeze(0)))
            source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
            cam.features = get_features(img_normalize(gt_image), VGG)
            cam.original_image = source_features

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    viewpoint_cameras = scene.getTrainCameras()
    
    
    for iteration in range(first_iter, opt.iterations + 1):
        os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)


        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
       
        # Content Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_features = viewpoint_cam.features
        render_features = get_features(img_normalize(image), VGG)
        loss_c = 0
        loss_c += torch.mean((gt_features['conv4_2'] - render_features['conv4_2']) ** 2)
        loss_c += torch.mean((gt_features['conv5_2'] - render_features['conv5_2']) ** 2)
        # Patch CLIP loss
        img_proc =[]
        for n in range(opt.num_crops):
            target_crop = cropper(image.unsqueeze(0))
            target_crop = augment(target_crop)
            img_proc.append(target_crop)

        img_proc = torch.cat(img_proc,dim=0)
        image_features = clip_model.encode_image(clip_normalize(img_proc))
        image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
    
        source_features = gt_image
        img_direction = (image_features-source_features)
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
        
        loss_temp = (1- torch.cosine_similarity(img_direction, style_direction.repeat(image_features.size(0),1), dim=1))
        loss_temp[loss_temp<0.7] =0
        loss_patch = loss_temp[loss_temp!=0.0].mean()
        # Direction CLIP loss
        render_features = clip_model.encode_image(clip_normalize(image.unsqueeze(0)))
        render_features /= (render_features.clone().norm(dim=-1, keepdim=True))
        
        img_direction = (render_features-source_features)
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
        
        loss_d = (1- torch.cosine_similarity(img_direction, style_direction.repeat(render_features.size(0),1), dim=1)).mean()
        
        loss = opt.lambda_dir * loss_d + opt.lambda_patch * loss_patch + opt.lambda_c * loss_c
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
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


def training_report(tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--poly_degree", type=int, default=7)
    
    parser.add_argument("--style_prompt", type=str, default = None)
    parser.add_argument("--style_image", type=str, default = None)
    
    parser.add_argument("--ply_path", type=str, default = None)

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

    args.save_iterations.append(args.iterations)

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)


    training(
        args.gs_type,
        lp.extract(args), op.extract(args), pp.extract(args),
        args
    )

    # All done
    print("\nTraining complete.")
