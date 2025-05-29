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
sys.path += ['..','models/dmisomodel']



import torch
from torchvision import models, transforms
from tqdm import tqdm

import CLIP
from CLIP.scene.VGG import get_features
from CLIP.utils.image_utils import img_normalize, clip_normalize

from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments import get_combined_args_force
from models.dmisomodel.gaussian_renderer import render, network_gui # type: ignore
from models.dmisomodel.games.dynamic.pcd_splatting.scene.pcd_gaussian_model import PcdGaussianModel # type: ignore
from models.dmisomodel.scene import Scene # type: ignore
from models.dmisomodel.utils.general_utils import safe_state, get_linear_noise_func # type: ignore
from models.dmisomodel.utils.image_utils import psnr # type: ignore
from models.dmisomodel.utils.loss_utils import l1_loss # type: ignore


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
    debug_from = args.debug_from
    tb_writer = prepare_output_and_logger(dataset)

    first_iter = 0

    gaussians = PcdGaussianModel(dataset.sh_degree, dataset.deform_width, dataset.deform_depth, dataset.is_blender,
                                 dataset.is_6dof)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.load_ply(f"{args.model_output}/point_cloud/iteration_best/point_cloud.ply")
    deform = gaussians.deform_model
    deform.train_setting(opt)

    deform.load_weights(args.model_output, "best")
    gaussians.load_time_weights(args.model_output, "best")
    gaussians.training_setup(opt)

    style_direction = CLIP.get_style_embedding(clip_model, args.style_prompt, args.style_image, args.object_prompt)

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
            source_features = clip_model.encode_image(clip_normalize(gt_image.unsqueeze(0)))
            source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
            cam.features = get_features(img_normalize(gt_image), VGG)
            cam.original_image = source_features

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)


    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, bg, scaling_modifer)["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        gaussians.update_learning_rate(iteration)

        BATCH_SIZE = args.batch_size
        batch_loss = 0.0
        batch_loss_d = 0.0
        skip_window = 0
        cam_idx = randint(0, max(0, len(viewpoint_stack) - BATCH_SIZE))
        for i in range(BATCH_SIZE):
            # Pick a random Camera
            if not viewpoint_stack:
                break
            viewpoint_cam = viewpoint_stack.pop(cam_idx + skip_window)
            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device()
            fid = viewpoint_cam.fid

            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            ast_noise = 0 if dataset.is_blender else (
                    torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration))

            d_v1, d_v2, d_v3, d_rot = deform.step(
                gaussians.pseudomesh[:, 0].detach(),
                gaussians.pseudomesh[:, 1].detach(),
                gaussians.pseudomesh[:, 2].detach(),
                time_input + ast_noise
            )

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, d_v1, d_v2, d_v3, d_rot, time_input,
                                   dataset.is_6dof)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

            # Content Loss
            gt_image = viewpoint_cam.original_image.cuda()
            gt_features = viewpoint_cam.features
            render_features = get_features(img_normalize(image), VGG)
            loss_c = 0
            loss_c += torch.mean((gt_features['conv4_2'] - render_features['conv4_2']) ** 2)
            loss_c += torch.mean((gt_features['conv5_2'] - render_features['conv5_2']) ** 2)
            # Patch CLIP loss
            img_proc = []
            for n in range(10000):
                target_crop = cropper(image.unsqueeze(0))
                flatten = target_crop[0].permute(1, 2, 0).reshape(args.crop_size * args.crop_size, 3)
                count = (flatten == bg).all(dim=1).sum()
                x = count/(args.crop_size * args.crop_size)
                if x  > 1: #tu cos namieszane
                    continue
                target_crop = augment(target_crop)
                img_proc.append(target_crop)
                if len(img_proc) >= opt.num_crops:
                    break

            img_proc = torch.cat(img_proc, dim=0)
            image_features = clip_model.encode_image(clip_normalize(img_proc))
            image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

            source_features = gt_image
            img_direction = (image_features - source_features)
            img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

            loss_temp = (1 - torch.cosine_similarity(img_direction, style_direction.repeat(image_features.size(0), 1),
                                                     dim=1))
            # loss_temp[loss_temp<0.7] =0
            # loss_patch = loss_temp[loss_temp!=0.0].mean()
            loss_patch = loss_temp.mean()

            # Direction CLIP loss
            render_features = clip_model.encode_image(clip_normalize(image.unsqueeze(0)))
            render_features /= (render_features.clone().norm(dim=-1, keepdim=True))

            img_direction = (render_features - source_features)
            img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

            loss_d = (1 - torch.cosine_similarity(img_direction, style_direction.repeat(render_features.size(0), 1),
                                                  dim=1)).mean()
            loss = opt.lambda_dir * loss_d + opt.lambda_patch * loss_patch + opt.lambda_c * loss_c

            # Loss
            batch_loss_d += loss_d
            batch_loss += loss

        batch_loss_d = batch_loss_d / BATCH_SIZE
        batch_loss = batch_loss / BATCH_SIZE
        batch_loss.backward()
        #loss.backward()
        #x = gaussians._xyz

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
            cur_psnr = training_report(tb_writer, iteration, loss_d, loss, l1_loss,
                                       iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, bg), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                gaussians.save_time_weights(args.model_path, iteration)

            # Optimizer step
            if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

                    #gaussians.update_learning_rate(iteration)
                    gaussians.time_optimizer.step()
                    gaussians.time_optimizer.zero_grad()
                    deform.optimizer.step()
                    deform.optimizer.zero_grad()
                    deform.update_learning_rate(iteration)

            #if (iteration in checkpoint_iterations):
            #    print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


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



def training_report(tb_writer, iteration, loss_d, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/loss_d', loss_d.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_v1, d_v2, d_v3, d_rot = deform.step(
                        scene.gaussians.pseudomesh[:, 0].detach(),
                        scene.gaussians.pseudomesh[:, 1].detach(),
                        scene.gaussians.pseudomesh[:, 2].detach(),
                        time_input  # ,
                    )
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_v1, d_v2, d_v3, d_rot, time_input,
                                   is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000, 31_000, 35_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000, 31_000, 35_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--style_prompt", type=str, default=None)
    parser.add_argument("--style_image", type=str, default=None)
    parser.add_argument("--object_prompt", type=str, default="a Photo")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument("--model_output", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    #shutil.copytree(args.model_path, args.style_output, dirs_exist_ok=True)


    args = get_combined_args_force(parser)
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
