import sys
import os
import torch
from os import makedirs
from models.flat_splatting.scene.points_gaussian_model import PointsGaussianModel
from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from argparse import ArgumentParser
from tqdm import tqdm


class GaussiansLoader:

    gaussians : GaussianModel

    def __init__(self, model_path, gaussians : GaussianModel, load_iteration):
        """b
        :param path: Path to colmap loader main folder.
        """
        self.model_path = model_path
        self.gaussians = gaussians

        if load_iteration == -1:
            self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        else:
            self.loaded_iter = load_iteration

        self.gaussians.load_ply(
            os.path.join(
                self.model_path,
                "point_cloud",
                "iteration_" + str(self.loaded_iter),
                "point_cloud.ply"
            )
        )


def write_simple_obj(mesh_v, mesh_f, filepath, verbose=False):
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in mesh_f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', filepath)
        

def save_pseudomesh_info(
        sh_degree,
        model_path,
        iteration : int,
        save_faces: bool = False,
        save_vertices: bool = False
):
    with torch.no_grad():
        gaussians = PointsGaussianModel(sh_degree)
        model = GaussiansLoader(model_path, gaussians, load_iteration=iteration)

        pseudomesh_path = os.path.join(model_path, "pseudomesh")
        makedirs(pseudomesh_path, exist_ok=True)

        time_func = gaussians.get_time
        for time in tqdm(range(torch.numel(time_func) + 1)):
            model = GaussiansLoader(model_path, gaussians, load_iteration=iteration)
            means3D = gaussians.get_xyz
            t = 0 + torch.sum(time_func[:time]).repeat(means3D.shape[0],1)

            # shape: [num_gaussians, 2 * polynomial_degree] -> [num_gaussians, 2] x polynomial_degree
            poly_weights = torch.chunk(gaussians._w1, chunks=gaussians.polynomial_degree, dim=-1)

            means3D = means3D[:, [0, -1]]
            center_gaussians = gaussians.get_m - t[0]

            for i, poly_weight in enumerate(poly_weights):
                means3D = means3D + poly_weight * (center_gaussians ** (i+1))

            means3D = torch.cat([
                    means3D[:, 0].unsqueeze(1),
                    torch.zeros(means3D[:, 0].shape).unsqueeze(1).cuda(),
                    means3D[:, -1].unsqueeze(1)
                ],
                dim=1
            )
            delta = norm_gauss(gaussians.get_m.squeeze(), gaussians.get_sigma.squeeze(), t[0]).unsqueeze(-1)
            scales = gaussians.get_scaling
            scales = delta * scales

            mask1 = (delta > 0.01).all(dim=1)
            s = scales[:,[0,-1]]
            mask2 = (s > 0.0001).all(dim=1)
            mask = torch.logical_and(mask1, mask2)

            gaussians._xyz = means3D[mask]
            gaussians._opacity = gaussians._opacity[mask]
            gaussians._scaling = torch.log(scales)[mask]
            gaussians._rotation = gaussians._rotation[mask]
            gaussians._features_dc = gaussians._features_dc[mask]
            gaussians._features_rest = gaussians._features_rest[mask]
            gaussians._w1 = gaussians._w1[mask]
            gaussians.m = gaussians.m[mask]
            gaussians.sigma = gaussians.sigma[mask]

            gaussians.prepare_vertices()
            gaussians.prepare_scaling_rot()

            triangles = gaussians.triangles

            faces = torch.arange(0, triangles.shape[0] * 3).reshape(triangles.shape[0], 3)
            vertices = triangles.reshape(triangles.shape[0] * 3, 3)

            filename = f'{pseudomesh_path}/{time}.obj'
            write_simple_obj(mesh_v=(vertices).detach().cpu().numpy(), mesh_f=faces, filepath=filename)

            gaussians.save_ply(f'{pseudomesh_path}/{time}.ply')
        print(f'All meshes saved to {pseudomesh_path}')
        


def norm_gauss(m, sigma, t):
    log = ((m - t)**2 / sigma**2) / -2
    return torch.exp(log)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--sh_degree", default=0, type=int)
    parser.add_argument("--save_faces", action="store_true")
    parser.add_argument("--save_vertices", action="store_true")

    args = parser.parse_args()

    print("Pseudomesh info " + args.model_path)

    model_path = args.model_path

    save_pseudomesh_info(
        args.sh_degree,
        args.model_path,
        args.iteration,
        args.save_faces,
        args.save_vertices
    )

