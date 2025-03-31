#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use

import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from torch import nn
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from scene.gaussian_model import GaussianModel
from utils.general_utils import (
    rot_to_quat_batch,
    build_rotation,
    inverse_sigmoid,
    get_expon_lr_func,
)
from games.dynamic.pcd_splatting.scene.deform_model import DeformModel
from utils.system_utils import mkdir_p, searchForMaxIteration
from gaussian_renderer import quaternion_multiply

class PcdGaussianModel(GaussianModel):

    def __init__(self, sh_degree: int, deform_width: int, deform_depth: int, is_blender: bool, is_6dof: bool):
        super().__init__(sh_degree)
        self.pseudomesh = torch.empty(0)
        self.eps_s0 = 1e-8

        self.scaling_activation = lambda x: torch.exp(x)
        self.scaling_inverse_activation = lambda x: torch.log(x)
        self.deform_model = DeformModel(deform_width, deform_depth, is_blender, is_6dof)

        self.use_attached_gauss = True
        self.mini = torch.empty(0, device="cuda")

        time_dim = self.deform_model.deform.time_input_ch
        self.additiontal_time_rot = torch.nn.Linear(in_features=4+time_dim, out_features=4, device="cuda")

    @property
    def get_xyz(self):
        return self.pseudomesh[:, 0]

    @property
    def get_scaling(self):
        self.s0 = torch.ones(self._scaling.shape[0], 1).cuda() * self.eps_s0
        return torch.cat([self.s0, self.scaling_activation(self._scaling[:, [-2, -1]])], dim=1)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_attached_shs(self):
        if not self.use_attached_gauss:
            return torch.empty(0, device="cuda")
        
        features_dc = self.attached_features_dc
        features_rest = self.attached_features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_attached_opacity(self):
        if not self.use_attached_gauss:
            return torch.empty(0, device="cuda")
        
        return self.opacity_activation(self.attached_opacity)
    
    def get_attached_scales(self, scales):
        if not self.use_attached_gauss:
            return torch.empty(0, device="cuda")
        
        # attached_scales = torch.clamp_min(self.attached_scale * scales.unsqueeze(1).expand(-1, self.num_splats, -1).reshape(-1, 3), self.eps_s0)
        attached_scales = self.scaling_activation(self.attached_scale) * scales.unsqueeze(1).expand(-1, self.num_splats, -1).reshape(-1, 3)
        return attached_scales
    
    def get_attached_rotations(self, rotations, time_input):
        if not self.use_attached_gauss:
            return torch.empty(0, device="cuda")
        
        time_input = time_input.unsqueeze(1).expand(-1, self.num_splats, -1).flatten(0, 1)
        time_input = self.deform_model.deform.embed_time_fn(time_input)
        
        time_dependent_rot = self.additiontal_time_rot(
            torch.cat(
                [self.attached_rotation, time_input], dim=-1
            ).cuda()
        )

        attached_rotations = quaternion_multiply(
            self.rotation_activation(time_dependent_rot), 
            rotations.unsqueeze(dim=1).expand(-1, self.num_splats, -1).flatten(0, 1)
        )
        return attached_rotations#.reshape(-1, 4)
    
    def calc_attached_gauss(self, v1, v2, v3):
        if not self.use_attached_gauss:
            return torch.empty(0, device="cuda")

        v2_v1 = v2 - v1
        v3_v1 = v3 - v1
        normal = torch.cross(
            v2_v1,
            v3_v1
        )
        v2_v1 = v2_v1 / torch.linalg.vector_norm(v2_v1, dim=-1, keepdim=True)
        v3_v1 = v3_v1 / torch.linalg.vector_norm(v3_v1, dim=-1, keepdim=True)
        normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        self.mini = torch.bmm(
            self.alpha,
            torch.stack((normal, v2_v1, v3_v1), dim=1)
        ).reshape(-1, 3)
        return self.mini + v1.unsqueeze(1).expand(-1, self.num_splats, -1).reshape(-1, 3)
    
    def setup_attached_gauss(self, num_splats=500, num_gauss=2000, alpha_lr=1e-3):
        self.num_splats = num_splats
        num_gauss = min(
            num_gauss,
            self.pseudomesh.shape[0]
        )
        print("Number of gaussians:", self.pseudomesh.shape[0])
        print("Number of super gaussians:", num_gauss)
        idx = np.random.choice(self.pseudomesh.shape[0], num_gauss)
        num = num_gauss * num_splats
        self.pseudomesh = self.pseudomesh[idx]
        self._opacity = self._opacity[idx]
        self._features_dc = self._features_dc[idx]
        self._features_rest = self._features_rest[idx]
        alpha = torch.zeros(
            num_gauss,
            num_splats,
            3
        )
        # alpha += torch.randn_like(alpha) * 0.01
        self.alpha = nn.Parameter(alpha.contiguous().cuda().requires_grad_(True))
        features_dc = self._features_dc.unsqueeze(1).expand(-1, num_splats, -1, -1).flatten(start_dim=0, end_dim=1).clone()
        features_rest = self._features_rest.unsqueeze(1).expand(-1, num_splats, -1, -1).flatten(start_dim=0, end_dim=1).clone()
        self.attached_features_dc = nn.Parameter(features_dc.cuda().requires_grad_(True))
        self.attached_features_rest = nn.Parameter(features_rest.cuda().requires_grad_(True))
        opacity = self._opacity.unsqueeze(1).expand(-1, num_splats, -1).flatten(start_dim=0, end_dim=1).clone()
        self.attached_opacity = nn.Parameter(opacity.cuda().requires_grad_(True))
        scale = torch.zeros((num, 3)).float()
        self.attached_scale = nn.Parameter(scale.contiguous().cuda().requires_grad_(True))
        rotation = torch.zeros((num, 4)).float()
        rotation[:, 0] = 1.0 # identity rotation quaternion is (1, 0, 0, 0)
        self.attached_rotation = nn.Parameter(rotation.contiguous().cuda().requires_grad_(True))

        self.optimizer.add_param_group({'params': [self.alpha], 'lr': alpha_lr, "name": "alpha"})
        self.optimizer.add_param_group({'params': [self.attached_features_dc], 'lr': self.training_args.feature_lr, "name": "attached_f_dc"})
        self.optimizer.add_param_group({'params': [self.attached_features_rest], 'lr': self.training_args.feature_lr, "name": "attached_f_rest"})
        self.optimizer.add_param_group({'params': [self.attached_opacity], 'lr': self.training_args.opacity_lr, "name": "attached_opacity"})
        self.optimizer.add_param_group({'params': [self.attached_scale], 'lr': 0.005, "name": "attached_scale"})
        self.optimizer.add_param_group({'params': [self.attached_rotation], 'lr': 0.001, "name": "attached_rotation"})

    def calc_vertices(self, d_v1, d_v2, d_v3, d_rot):
        v1 = self.pseudomesh[:, 0]
        v2 = self.pseudomesh[:, 1]
        v3 = self.pseudomesh[:, 2]

        _v2 = v2 - v1
        _v3 = v3 - v1

        if torch.is_tensor(d_rot):
            R = build_rotation(d_rot)
            _v2 = torch.einsum('bik,bk->bi', [R, _v2]) # batched matrix vector multiplication
            _v3 = torch.einsum('bik,bk->bi', [R, _v3]) # batched matrix vector multiplication

        v1 = v1 + d_v1
        v2 = v2 + d_v1 + _v2/torch.linalg.vector_norm(_v2, dim=-1, keepdim=True) * d_v2
        v3 = v3 + d_v1 + _v3/torch.linalg.vector_norm(_v3, dim=-1, keepdim=True) * d_v3

        return v1, v2, v3

    def training_setup(self, training_args):
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.percent_dense = training_args.percent_dense

        self.training_args = training_args
        l_params = [
            {'params': [self.pseudomesh], 'lr': training_args.pseudomesh_lr_init * self.spatial_lr_scale, "name": "pseudomesh"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"}
        ]

        self.optimizer = torch.optim.Adam(l_params, lr=0.0, eps=1e-15)
        self.time_optimizer = torch.optim.Adam(self.additiontal_time_rot.parameters(), lr=0.1)

        self.pseudomesh_scheduler_args = get_expon_lr_func(lr_init=training_args.pseudomesh_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.pseudomesh_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.pseudomesh_lr_delay_mult,
                                                    max_steps=training_args.pseudomesh_lr_max_steps)

        #self.deform_model.train_setting(training_args)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "pseudomesh":
                lr = self.pseudomesh_scheduler_args(iteration)
                param_group['lr'] = lr
                return
            
    def init_pseudomesh(self, xyz):
        """
        Prepare pseudo-mesh faces.
        """
        scales = self.get_scaling
        rotation = self._rotation
        pseudomesh = self.create_faces(xyz, scales, rotation)
        self.pseudomesh = nn.Parameter(pseudomesh.contiguous().requires_grad_(True))
        #self.idx_faces = torch.arange(0, pseudomesh.shape[0]).cuda()

    @staticmethod
    def create_faces(xyz, scales, rotation):
        R = build_rotation(rotation)
        R = R.transpose(-2, -1)

        v1 = xyz
        s_2 = scales[:, -2]
        s_3 = scales[:, -1]
        _v2 = v1 + s_2.reshape(-1, 1) * R[:, 1]
        _v3 = v1 + s_3.reshape(-1, 1) * R[:, 2]

        mask = s_2 > s_3

        v2 = torch.zeros_like(_v2)
        v3 = torch.zeros_like(_v3)

        v2[mask] = _v2[mask]
        v3[mask] = _v3[mask]

        v2[~mask] = _v3[~mask]
        v3[~mask] = _v2[~mask]
        #idx = torch.rand_like(v1)
        pseudomesh = torch.stack([v1, v2, v3], dim = 1)
        return pseudomesh

    def _prepare_scaling_rot(self, v1, v2, v3, eps=1e-8):
        """
        Approximate covariance matrix and calculate scaling/rotation tensors.
        Prepare parametrized Gaussian.
        """

        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)

        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        _s2 = v2 - v1
        _s3 = v3 - v1

        s2 = torch.linalg.vector_norm(_s2, dim=-1, keepdim=True)
        _s3_norm = torch.linalg.vector_norm(_s3, dim=-1, keepdim=True)

        r1 = torch.cross(
            _s2/s2,
            _s3/_s3_norm
        )

        s2 = s2 + eps
        _s3_norm = _s3_norm + eps

        r1 = r1 / (torch.linalg.vector_norm(r1, dim=-1, keepdim=True) + eps)
        r2 = _s2 / s2
        r3 = _s3 - proj(_s3, r1) - proj(_s3, r2)
        r3 = r3 / (torch.linalg.vector_norm(r3, dim=-1, keepdim=True) + eps)
        s3 = dot(_s3, r3)

        scales = torch.cat([s2, s3], dim=1)
        _scaling = self.scaling_inverse_activation(torch.abs(scales))

        rotation = torch.stack([r1, r2, r3], dim=-1)

        _rotation = rot_to_quat_batch(rotation)

        return _scaling, _rotation

    def prepare_scaling_rot(self, eps=1e-8):
        """
        Approximate covariance matrix and calculate scaling/rotation tensors.
        Prepare parametrized Gaussian.
        """

        v1 = self.pseudomesh[:, 0]
        v2 = self.pseudomesh[:, 1]
        v3 = self.pseudomesh[:, 2]

        self._scaling, self._rotation = self._prepare_scaling_rot(v1, v2, v3)
        #self.init_pseudomesh(v1)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = scales
        self._rotation = rots
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.init_pseudomesh(fused_point_cloud)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def compaction(self, n=10):
        # Extract points that satisfy the gradient condition
        pseudo_dist = torch.abs(self.pseudomesh.grad).sum(dim=1).sum(dim=1)
        sorted, indices = torch.sort(pseudo_dist)
        selected_gaussians = indices[-n:]
        new_pseudomesh = self.pseudomesh[selected_gaussians]
        v1 = new_pseudomesh[:, 0]
        v2 = new_pseudomesh[:, 1]
        v3 = new_pseudomesh[:, 2]

        m = new_pseudomesh.sum(dim=1) / 3
        tr1 = torch.stack([m, v1, v2], dim=1) * 2 - m.unsqueeze(1)
        tr2 = torch.stack([m, v2, v3], dim=1) * 2 - m.unsqueeze(1)
        tr3 = torch.stack([m, v3, v1], dim=1) * 2 - m.unsqueeze(1)

        new_pseudomesh = torch.vstack([tr1, tr2, tr3])

        new_features_dc = self._features_dc[selected_gaussians].repeat(3, 1, 1, 1).reshape(3 * n, 1, 3)
        new_features_rest = self._features_rest[selected_gaussians].repeat(3, 1, 1, 1).reshape(3 * n, 15, 3)
        new_opacities = self._opacity[selected_gaussians].repeat(3, 1, 1).reshape(3 * n, 1)
        new_opacities = new_opacities + torch.rand_like(new_opacities) * 0.001
        self._opacity[selected_gaussians] = 0

        self.densification_postfix(new_pseudomesh, new_features_dc, new_features_rest, new_opacities)

        return selected_gaussians

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        self.prepare_scaling_rot()
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_pseudomesh = self.pseudomesh[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]

        self.densification_postfix(new_pseudomesh, new_features_dc, new_features_rest, new_opacities)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        self.prepare_scaling_rot()
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent
        )
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_pseudomesh = self.create_faces(new_xyz, new_scaling, new_rotation)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_pseudomesh, new_features_dc, new_features_rest, new_opacity)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self.pseudomesh = optimizable_tensors["pseudomesh"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        #last_idx = self.idx_faces[-1] + 1
        
        """
        self.idx_faces = torch.cat(
            [
                self.idx_faces,
                torch.arange(last_idx, last_idx + self.pseudomesh.shape[0] - self.idx_faces.shape[0]).cuda()
            ]
        )
        self.idx_faces = self.idx_faces[valid_points_mask]
        """


    def densification_postfix(self, new_pseudomesh, new_features_dc, new_features_rest, new_opacities):
        d = {"pseudomesh": new_pseudomesh,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.pseudomesh = optimizable_tensors["pseudomesh"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self.prepare_scaling_rot()

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor_grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def save_time_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "time_net/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.additiontal_time_rot.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_time_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "time_net/iteration_{}/deform.pth".format(loaded_iter))
        self.additiontal_time_rot.load_state_dict(torch.load(weights_path))

    def save_ply(self, path):
        self._xyz = self.pseudomesh[:, 0]
        self.prepare_scaling_rot()
        self._save_ply(path)

        attrs = self.__dict__
        additional_attrs = [
            'pseudomesh',
            'alpha',
            'attached_scale',
            'attached_rotation',
            'attached_opacity',
            'attached_features_dc',
            'attached_features_rest',
            'num_splats',
            'use_attached_gauss'
        ]

        save_dict = {}
        for attr_name in additional_attrs:
            if hasattr(self, attr_name):
                save_dict[attr_name] = attrs[attr_name]

        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        torch.save(save_dict, path_model)

    def load_ply(self, path, og_number_points=1):
        self._load_ply(path)
        self.og_number_points = og_number_points
        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        params = torch.load(path_model)
        if 'pseudomesh' in params:
            self.pseudomesh = nn.Parameter(params['pseudomesh'])
        if 'alpha' in params:
            self.alpha = nn.Parameter(params['alpha'])
        if 'attached_scale' in params:
            self.attached_scale = nn.Parameter(params['attached_scale'])
        if 'attached_rotation' in params:
            self.attached_rotation = nn.Parameter(params['attached_rotation'])
        if 'attached_opacity' in params:
            self.attached_opacity = nn.Parameter(params['attached_opacity'])
        if 'attached_features_dc' in params:
            self.attached_features_dc = nn.Parameter(params['attached_features_dc'])
        if 'attached_features_rest' in params:
            self.attached_features_rest = nn.Parameter(params['attached_features_rest'])
        if 'num_splats' in params:
            self.num_splats = params['num_splats']
        if 'use_attached_gauss' in params:
            self.use_attached_gauss = params['use_attached_gauss']
        
        #if 'idx_faces' in params:
        #    self.idx_faces = params['idx_faces']

    def _save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def _load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
