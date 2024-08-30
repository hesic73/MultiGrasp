from copy import deepcopy
from pickletools import optimize
from queue import PriorityQueue
from re import search

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import trimesh as tm
from utils.utils_3d import compute_rotation_matrix_from_ortho6d as rot_matrix
from numpy.random.mtrand import normal
from torch.nn.functional import normalize
from tqdm import tqdm, trange

from utils.Losses import FCLoss
from utils.utils import *
from utils.utils_3d import do_rotation, do_translation

from .EMA import EMA

from .HandModel import RoboticHand
from .ObjectModels import ODFieldModel

from typing import List, Tuple


class PhysicsGuide:
    def __init__(self, hand_model: RoboticHand, args):
        self.epsilon = 1e-4
        self.hand_model = hand_model
        self.n_contact_total = self.hand_model.n_pts

        self.fc_loss_model = FCLoss()
        self.args = args
        self.grad_ema_q = EMA(0.98)
        self.grad_ema_w = EMA(0.98)

        self.arange = torch.arange(self.args.batch_size).cuda()
        # self.object_no = 0
        self.joint_mask = torch.ones(
            [self.args.batch_size, self.hand_model.q_len], device='cuda', dtype=torch.int32)
        self.object_models: List[ODFieldModel] = []

        self.optimizer = None

        if 'hc_pen' in vars(args) and args.hc_pen:
            self.ho_penetration = self.ho_pen_hc
        else:
            self.ho_penetration = self.ho_pen_oc

        self.table_height: float = 0.0

    def append_object(self, new_object_model):
        self.object_models = [new_object_model]

    def get_vertices(self, q=None):
        vertices = self.hand_model.get_vertices(q)
        return vertices

    def get_contacts(self, contact_idx, qs, q=None, downsample: bool = True, no_base: bool = False, with_objects: bool = False):
        # vertices = self.hand_model.get_vertices(qs[:, -1])
        # self.hand_model.get_contact_points(contact_idx, contact_point_weight, qs[:, -1])
        # if handcodes is None:
        # `centers`:    batch_size x (num_grasped) x 3
        # `rotations`:  batch_size x (num_grasped) x 3 x 3
        # handcodes[:, object_ind] = handcode

        # normals = self.hand_model.get_surface_normals(verts=vertices)
        vertices, normals = self.hand_model.get_contact_points_and_normal(
            contact_idx, qs[:, -1])
        if with_objects:
            verts = [vertices]
            norms = [normals]
            centers, rotations = self.compute_object_pose(qs)
            for ind, obj in enumerate(self.object_models[:-1]):
                obj_points = obj.sampled_points[:qs.shape[0]]
                obj_grads = obj.points_gradient[:qs.shape[0]]
                # `centers[:, ind]`:    batch_size x 3
                # `rotations[:, ind]`:  batch_size x 3 x 3
                # `points_gradient`:    batch_size x (den^2) x 3
                obj_norms = torch.bmm(rotations[:, ind].unsqueeze(1).tile(
                    (1, obj_points.shape[1], 1, 1)).view([-1, 3, 3]), obj_points.unsqueeze(-1).view([-1, 3, 1]))
                obj_verts = torch.bmm(rotations[:, ind].unsqueeze(1).tile((1, obj_points.shape[1], 1, 1)).view(
                    [-1, 3, 3]), obj_points.unsqueeze(-1).view([-1, 3, 1])).squeeze(-1)
                obj_verts = obj_verts + \
                    centers[:, ind].unsqueeze(1).tile(
                        (1, obj_points.shape[1], 1)).view([-1, 3])
                verts.append(obj_verts.view(obj_points.shape))
                norms.append(obj_norms.view(obj_points.shape))
            vertices = torch.concat(verts, dim=1)
            normals = torch.concat(norms, dim=1)

        return vertices, normals

    def ho_pen_oc(self, obj: ODFieldModel, hand_verts=None):
        # return obj.po_penetration(hand_verts).max(-1)[0]
        return obj.po_penetration(hand_verts).sum(dim=-1)

    def ho_pen_hc(self, obj: ODFieldModel, hand_verts=None):
        return self.hand_model.penetration(obj.get_surface_points()).sum(dim=-1) * 1000

    def oo_penetration(self):
        """
        Calculate object-object penetration.

        Args:
            `object_models`: `list` of `ObjectModels`

        `rotation` and `center` of the last object should be eye matrices and 0 vectors.
        Returns:
            Sum of object-object penetration: `batch_size`
        """
        batch_size = self.args.batch_size
        object_amount = len(self.object_models)

        penetration = torch.zeros(
            [batch_size], device='cuda', dtype=torch.float32)

        if object_amount == 1:
            return penetration

        for i in range(object_amount):
            for j in range(0, object_amount):
                # for j in range(i + 1, object_amount):
                # Bi-directional object-object penetration
                if i == j:
                    continue

                # Any object i and non-sphere j
                points_global = do_rotation(
                    self.object_models[i].surface_points, self.object_models[i].orient)
                points_global = do_translation(
                    points_global, self.object_models[i].transl)
                pen = self.object_models[j].po_penetration(
                    points_global).sum(-1)

                penetration = penetration + pen

        return penetration

    def compute_energy(self, cpi: torch.Tensor, cpw: torch.Tensor, q: torch.Tensor, reduce: bool = True):
        """_summary_

        Args:
            cpi (torch.Tensor): contact point index. Shape: (n_batch, n_obj, n_contact)
            cpw (torch.Tensor): contact point weight. Shape: (n_batch, n_obj, n_contact, 4)
            q (torch.Tensor): ???. Shape: (n_batch, ?)
            reduce (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        B, N_o, N_c = cpi.shape

        # Compute hand
        self.hand_model.update_kinematics(q)

        hand_verts = self.hand_model.get_vertices()
        contact_point = []
        force_closure = []
        normal_alignment = []
        penetration = []
        distance = []

        for i_obj, object_model in enumerate(self.object_models):
            cp, hn = self.hand_model.get_contact_points_and_normal(
                cpi[:, i_obj], cpw[:, i_obj])
            contact_point.append(cp)

            surface_distance, contact_normal = object_model.distance_gradient(
                cp)
            surface_distance = torch.abs(surface_distance)
            distance.append(surface_distance)

            contact_normal = contact_normal + \
                torch.normal(0, self.epsilon, contact_normal.shape,
                             device=contact_normal.device, dtype=contact_normal.dtype)
            contact_normal = F.normalize(contact_normal, dim=-1, p=2)

            hn = hn + torch.normal(0, self.epsilon,
                                   hn.shape, device=hn.device, dtype=hn.dtype)
            hn = F.normalize(hn, dim=-1, p=2)

            normal_alignment.append(torch.relu(
                1 - (hn * contact_normal).sum(-1)))
            force_closure.append(
                self.fc_loss_model.fc_loss(cp, contact_normal))

            penetration.append(self.ho_penetration(object_model, hand_verts))

        penetration.append(self.hand_model.self_penetration())

        if not self.args.levitate:
            table_pntr = torch.relu(
                self.table_height - hand_verts[:, :, 2]).sum(dim=-1)
            # table_pntr = torch.relu(self.table_height - hand_verts[:, :, 2]).max(dim=-1)[0]
            penetration.append(table_pntr)

        force_closure = torch.stack(force_closure, dim=1)
        normal_alignment = torch.stack(normal_alignment, dim=1)
        penetration = torch.stack(penetration, dim=1)
        distance = torch.stack(distance, dim=1)

        hand_prior = self.hand_model.prior(q)

        if reduce:
            return force_closure.sum(dim=1), distance.sum(-1).sum(-1), penetration.sum(dim=1), hand_prior, normal_alignment.sum(-1).sum(-1)
        else:
            return force_closure, distance, penetration, hand_prior, normal_alignment

    # def get_stepsize(self, energy):
    def get_stepsize(self, t):
        # (t // stepsize_period)
        return self.args.noise_size * self.args.temperature_decay ** torch.div(t, self.args.stepsize_period, rounding_mode='floor')
        # return 0.02600707 + energy.unsqueeze(1) * 0.03950357 * 1e-3

    # def get_temperature(self, energy):
    def get_temperature(self, t):
        # (t // annealing_period)
        return self.args.starting_temperature * self.args.temperature_decay ** torch.div(t, self.args.annealing_period, rounding_mode='floor')
        # return 0.02600707 + energy * 0.03950357
