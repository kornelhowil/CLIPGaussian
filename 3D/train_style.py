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
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
sys.path += ['..','models/gs']

import torch
from torchvision import models, transforms
from tqdm import tqdm

from CLIP.utils.image_utils import img_normalize, clip_normalize
from CLIP.scene.VGG import get_features
import CLIP

from arguments import ModelParams, PipelineParams, OptimizationParams
from models.gs.gaussian_renderer import render, network_gui
from models.gs.scene import Scene, GaussianModel
from models.gs.utils.general_utils import safe_state
from models.gs.utils.image_utils import psnr
from models.gs.utils.loss_utils import l1_loss


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

VGG = models.vgg19(weights='DEFAULT').features
VGG.to("cuda")
clip_model = CLIP.load_model()

def training(dataset, opt, pipe, args):
    testing_iterations = args.test_iterations
    saving_iterations = args.save_iterations
    checkpoint_iterations = args.checkpoint_iterations
    checkpoint = args.start_checkpoint
    debug_from = args.debug_from

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.load_ply(args.ply_path)
    gaussians.training_setup(opt)

    style_direction = CLIP.get_style_embedding(
        clip_model, args.style_prompt, args.style_image, args.object_prompt)

    cropper = transforms.Compose([
        transforms.RandomCrop(opt.crop_size)
    ])
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        transforms.Resize(224)
    ])
    
    with torch.no_grad():
        for cam in scene.getTrainCameras().copy():
            gt_image = cam.original_image.cuda()
            clip_features = clip_model.encode_image(
                clip_normalize(gt_image.unsqueeze(0)))
            clip_features /= (clip_features.clone().norm(dim=-1, keepdim=True))
            cam.clip_features = clip_features
            cam.vgg_features = get_features(img_normalize(gt_image), VGG)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations),
                        desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)[
                        "render"]
                    net_image_bytes = memoryview((torch.clamp(
                        net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand(
            (3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, _, _, _ = render_pkg["render"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Content Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_features = viewpoint_cam.vgg_features
        render_features =  get_features(img_normalize(image), VGG)
        loss_c = 0
        loss_c += torch.mean((gt_features['conv4_2'] -
                             render_features['conv4_2']) ** 2)
        loss_c += torch.mean((gt_features['conv5_2'] -
                             render_features['conv5_2']) ** 2)
        # Patch CLIP loss
        img_proc = [augment(cropper(image.unsqueeze(0))) for _ in range(opt.num_crops)]

        img_proc = torch.cat(img_proc, dim=0)
        image_features = clip_model.encode_image(clip_normalize(img_proc))
        image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

        source_features = viewpoint_cam.clip_features
        img_direction = (image_features-source_features)
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

        loss_temp = (1 - torch.cosine_similarity(img_direction,
                                                 style_direction.repeat(image_features.size(0), 1), dim=1))
        loss_patch = loss_temp.mean()

        # BG Loss
        if args.object:
            color_sum = torch.sum(gt_image, dim=0)
            l1 = torch.abs(
                image - gt_image).mean(dim=0)[color_sum == torch.sum(background)].mean().double()
        else:
            l1 = 0
        # Direction CLIP loss
        render_features = clip_model.encode_image(
            clip_normalize(image.unsqueeze(0)))
        render_features /= (render_features.clone().norm(dim=-1, keepdim=True))

        img_direction = (render_features-source_features)
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

        loss_d = (1 - torch.cosine_similarity(img_direction,
                  style_direction.repeat(render_features.size(0), 1), dim=1)).mean()

        loss = opt.lambda_dir * loss_d + opt.lambda_patch * loss_patch + \
            opt.lambda_c * loss_c + opt.lambda_bg * l1
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
            training_report(tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(
                iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")


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


def training_report(tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar(
            'train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(
                        viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(
                            viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(
                                viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                    iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(
                'total_points', scene.gaussians.get_xyz.shape[0], iteration)
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
    parser.add_argument("--test_iterations", nargs="+",
                        type=int, default=[30_000, 31_000, 35_000])
    parser.add_argument("--save_iterations", nargs="+",
                        type=int, default=[30_000, 31_000, 35_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations",
                        nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--style_prompt", type=str, default=None)
    parser.add_argument("--style_image", type=str, default=None)
    parser.add_argument("--object_prompt", type=str, default="a Photo")

    parser.add_argument("--ply_path", type=str, default=None)

    parser.add_argument("--object", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args),
             op.extract(args),
             pp.extract(args),
             args)
    # All done
    print("\nTraining complete.")
