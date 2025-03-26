import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from models import gaussianModelRender


def modify_func(means3D: torch.Tensor, # num_gauss x 3, means3D[:,1] = 0
                scales: torch.Tensor, # num_gauss x 3, scales[:,1] = eps
                rotations: torch.Tensor, # # num_gauss x 4, 3D quaternions of 2D rotations
                time: float):
    return means3D, scales, rotations

def render_set( model_path,
                iteration,
                views, 
                gaussians, 
                pipeline, 
                background, 
                interp, 
                extension):
    render_path = os.path.join(model_path, f"render")

    makedirs(render_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        for i in range(interp):
            rendering = render(view, gaussians, pipeline, background, interp=interp, interp_idx=i, modify_func=modify_func)["render"].cpu()
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + "_" + str(i) + extension))

def render_sets(dataset : ModelParams,
                iteration : int, 
                pipeline : PipelineParams, 
                skip_train : bool, 
                skip_test : bool, 
                interp : int,
                extension: str):
    with torch.no_grad():
        gaussians = gaussianModelRender['gs'](dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        render_set(dataset.model_path, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, interp, extension)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--camera', type=str, default="mirror")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--skip_train", action="store_false")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--poly_degree", type=int, default=1)
    parser.add_argument("--interp", type=int, default=1)
    parser.add_argument("--extension", type=str, default=".png")

    args = get_combined_args(parser)
    model.gs_type = "gs"
    model.camera = args.camera
    model.distance = args.distance
    model.num_pts = args.num_pts
    model.poly_degree = args.poly_degree

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.interp, args.extension)