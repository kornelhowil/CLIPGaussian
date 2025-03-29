import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer.bg_fg_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from models import gaussianModelRender
import copy

def modify_mesh(triangles: torch.Tensor, # num_gaussians x 3 x 3, triangles[:,:,1] = 0
                time: float
                ):
    return triangles


def render_set_combine(model_path, views, gaussians_fg, gaussians_fg_mesh, gaussians_bg, pipeline, background, extension):
    render_path = os.path.join(model_path, "render")
    obj_path = os.path.join(model_path, "pseudomesh")
    pipeline.obj_path = obj_path

    makedirs(render_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if os.path.exists(os.path.join(obj_path, f"{view.time}.obj")):
            rendering = render(view, gaussians_fg_mesh, gaussians_bg, pipeline, background, modify_mesh=modify_mesh)["render"].cpu()
        else:
            rendering = render(view, gaussians_fg, gaussians_bg, pipeline, background)["render"].cpu()
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + extension))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, extension: str):
    with torch.no_grad():
        parser = copy.deepcopy(dataset)
        
        gaussians_fg = gaussianModelRender['gs'](dataset.sh_degree)
        gaussians_fg_mesh = gaussianModelRender['pgs'](dataset.sh_degree)
        scene_fg = Scene(dataset, gaussians_fg, load_iteration=iteration, shuffle=False)

        if parser.bg_model != "":
            dataset.model_path = dataset.bg_model
            gaussians_bg = gaussianModelRender['gs'](dataset.sh_degree)
            scene_bg = Scene(dataset, gaussians_bg, load_iteration=iteration, shuffle=False)
        else:
            gaussians_bg = None
            scene_bg = None

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        render_set_combine(dataset.model_path, scene_fg.getTestCameras(), gaussians_fg, gaussians_fg_mesh, gaussians_bg, pipeline, background, extension)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--camera', type=str, default="mirror")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument('--gs_type', type=str, default="gs")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_frame", type=int, default=5)
    parser.add_argument("--bg_model", type=str, default="")
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--bg_stop_frame", type=int, default=-1)
    parser.add_argument("--extension", type=str, default=".png")


    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.camera = args.camera
    model.distance = args.distance
    model.num_pts = args.num_pts
    model.bg_model = args.bg_model
    pipeline.start_frame = args.start_frame
    pipeline.scale = args.scale
    pipeline.bg_stop_frame = args.bg_stop_frame

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.extension)