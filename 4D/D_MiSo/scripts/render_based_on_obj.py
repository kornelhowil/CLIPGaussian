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
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_soup
from scene.dataset_readers import CameraInfo
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import trimesh
from games.dynamic.pcd_splatting.scene.pcd_gaussian_model import PcdGaussianModel
import json
from pathlib import Path
from PIL import Image
import numpy as np
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.camera_utils import cameraList_from_camInfos


def render_set(model_path, load2gpu_on_the_fly, name, iteration, views, gaussians, pipeline, background, objpath, resize):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)

    scene = trimesh.load(
        objpath,
        force='mesh'
    )
    sub_triangle_soup = torch.tensor(scene.triangles).cuda().float()/resize

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()

        results = render_soup(view, gaussians, pipeline, background, sub_triangle_soup)
        rendering = results["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def readCamerasFromTransforms(path, transformsfile, extension=".png", height=800, width=800 ):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            if "file_path" in frame:
                cam_name = os.path.join(path, frame["file_path"] + extension)
            else:
                cam_name = None
            if 'time' in frame:
                frame_time = frame['time']
            else:
                frame_time = 0.0

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            fovy = focal2fov(fov2focal(fovx,width), height)
            FovY = fovx
            FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                                        image_path=None, image_name=None, width=width,
                                        height=height, fid=frame_time))

    return cam_infos


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, objpath, resize):
    with torch.no_grad():
        gaussians = PcdGaussianModel(dataset.sh_degree, dataset.deform_width, dataset.deform_depth, dataset.is_blender, dataset.is_6dof)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = gaussians.deform_model
        deform.load_weights(dataset.model_path, iteration)
        gaussians.load_time_weights(dataset.model_path, iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_func = render_set

        cam_infos = readCamerasFromTransforms(
           dataset.source_path , "transforms_renders.json")
        cam_infos = cameraList_from_camInfos(cam_infos, 1.0, dataset)

        render_func(
            dataset.model_path, dataset.load2gpu_on_the_fly, "additional_views", scene.loaded_iter,
            cam_infos, gaussians, pipeline,
            background, objpath, resize
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default='best')
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--objpath", default="")
    parser.add_argument("--resize_soup", default=100)
    parser.add_argument("--mode", default='render', choices=['render'])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.objpath, args.resize_soup)
