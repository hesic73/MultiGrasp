import json
import os

import numpy as np
import plotly.graph_objects as go
import pytorch_kinematics as pk
import torch
import transforms3d
import trimesh as tm
import pytorch_kinematics.urdf_parser_py.urdf as URDF_PARSER
from pytorch_kinematics.urdf_parser_py.urdf import Robot, Link
from pytorch_kinematics.urdf_parser_py.urdf import xmlr
from pytorch3d import transforms
from pytorch_kinematics.urdf_parser_py.urdf import (URDF, Box, Cylinder, Mesh,
                                                    Sphere)
from tqdm import tqdm, trange
import torch.nn.functional as F
from utils.utils_3d import compute_ortho6d_from_rotation_matrix, compute_rotation_matrix_from_ortho6d, cross_product
from itertools import permutations


from typing import Dict, List, Tuple


def get_mesh(geometry: xmlr.Object, mesh_path: str, hand_model: str):
    if isinstance(geometry, Mesh):
        if hand_model == 'shadowhand' or hand_model == 'allegro' or hand_model == 'barrett' or hand_model == 'tonghand' or hand_model == 'tonghand_viz':
            filename = geometry.filename.split('/')[-1]
        else:
            filename = geometry.filename
        mesh = tm.load(os.path.join(mesh_path, filename),
                       force='mesh', process=False)
    elif isinstance(geometry, Cylinder):
        mesh = tm.primitives.Cylinder(
            radius=geometry.radius, height=geometry.length)
    elif isinstance(geometry, Box):
        mesh = tm.primitives.Box(extents=geometry.size)
    elif isinstance(geometry, Sphere):
        mesh = tm.primitives.Sphere(
            radius=geometry.radius)
    else:
        raise NotImplementedError
    return mesh


class RectContactAreas:
    def __init__(self, basis: torch.Tensor, normals: torch.Tensor, batch_size: int, device):
        """

        Args:
            basis (torch.Tensor): (n_areas, 4, 4)
            normals (torch.Tensor): (n_areas, 3)
        """
        self.batch_size = batch_size
        self.device = device
        self.basis = basis  # (n_areas, 4, 4)
        self.normals = normals  # (n_areas, 3)

    def get_contact_points(self, trans_matrix: torch.Tensor, cpi: torch.Tensor, cpw: torch.Tensor, global_rotation: torch.Tensor, global_translation: torch.Tensor) -> torch.Tensor:
        """

        Args:
            trans_matrix (torch.Tensor): (B, n_areas, 4, 4)
            cpi (torch.Tensor): (B, n_contact)
            cpw (torch.Tensor): (B, n_contact, 4)
            global_rotation (torch.Tensor): (B, 3, 3)
            global_translation (torch.Tensor): (B, 3)

        Returns:
            torch.Tensor: (B, n_contact, 3)
        """
        cpw = F.softmax(cpw, dim=-1)  # (B, n_contact, 4)
        cpb_trans = self.basis.unsqueeze(0)  # (1, n_areas, 4, 4)
        cpb_trans = torch.matmul(
            trans_matrix, cpb_trans.transpose(-1, -2)).transpose(-1, -2)[..., :3]  # (B, n_areas, 4, 3)
        cpb_trans = cpb_trans[torch.arange(0, self.batch_size, device=self.device).unsqueeze(
            1).long(), cpi.long()]  # (B, n_contact, 4, 3)
        cpb_trans = (cpb_trans * cpw.unsqueeze(-1)
                     ).sum(dim=2)  # (B, n_contact, 3)
        cpb_trans = torch.matmul(global_rotation, cpb_trans.transpose(
            -1, -2)).transpose(-1, -2) + global_translation.unsqueeze(1)  # (B, n_contact, 3)

        return cpb_trans

    def get_contact_points_and_normal(self, trans_matrix: torch.Tensor, cpi: torch.Tensor, cpw: torch.Tensor, global_rotation: torch.Tensor, global_translation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            trans_matrix (torch.Tensor): (B, n_areas, 4, 4)
            cpi (torch.Tensor): (B, n_contact)
            cpw (torch.Tensor): (B, n_contact, 4)
            global_rotation (torch.Tensor): (B, 3, 3)
            global_translation (torch.Tensor): (B, 3)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (B, n_contact, 3), (B, n_contact, 3)
        """

        cpw = F.softmax(cpw, dim=-1)  # (B, n_contact, 4)
        cpb_trans = self.basis.unsqueeze(0)  # (1, n_areas, 4, 4)
        cpb_trans = torch.matmul(
            trans_matrix, cpb_trans.transpose(-1, -2)).transpose(-1, -2)[..., :3]  # (B, n_areas, 4, 3)
        cpb_trans = cpb_trans[torch.arange(0, self.batch_size, device=self.device).unsqueeze(
            1).long(), cpi.long()]  # (B, n_contact, 4, 3)
        cpb_trans = (cpb_trans * cpw.unsqueeze(-1)
                     ).sum(dim=2)  # (B, n_contact, 3)
        cpb_trans = torch.matmul(global_rotation, cpb_trans.transpose(
            -1, -2)).transpose(-1, -2) + global_translation.unsqueeze(1)  # (B, n_contact, 3)

        cpn_trans = self.normals.unsqueeze(0)  # (1, n_areas, 3)
        cpn_trans = cpn_trans.expand(
            self.batch_size, -1, -1)  # (B, n_areas, 3)
        cpn_trans = torch.matmul(
            trans_matrix[..., :3, :3], cpn_trans.unsqueeze(-1)).squeeze(-1)  # (B, n_areas, 3)
        cpn_trans = cpn_trans[torch.arange(0, self.batch_size, device=self.device).unsqueeze(
            1).long(), cpi.long()]  # (B, n_contact, 3)
        cpn_trans = torch.matmul(
            global_rotation, cpn_trans.transpose(-1, -2)).transpose(-1, -2)  # (B, n_contact, 3)
        return cpb_trans, cpn_trans


class RoboticHand:
    def __init__(self, hand_model: str, urdf_filename: str, mesh_path: str,
                 batch_size: int = 1, hand_scale: float = 1., pts_density: float = 25000,
                 device=torch.device(
                     'cuda' if torch.cuda.is_available() else 'cpu'),
                 **kwargs
                 ):
        self.device = device
        self.hand_model = hand_model
        self.batch_size = batch_size
        self.robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(
            dtype=torch.float, device=self.device)
        self.robot_full: Robot = URDF_PARSER.URDF.from_xml_file(urdf_filename)

        self.joint_param_names = self.robot.get_joint_parameter_names()
        self.q_len = 9

        # prepare geometries for visualization
        self.global_translation = None
        self.global_rotation = None
        self.softmax = torch.nn.Softmax(dim=-1)

        self.contact_point_dict: Dict[str, List] = json.load(
            open(os.path.join("data/urdf/", 'contact_%s.json' % hand_model)))
        self.surface_points = {}

        self.penetration_keypoints_dict = json.load(
            open(os.path.join("data/urdf/", 'pntr_%s.json' % hand_model)))
        self.penetration_keypoints_dict: Dict[str, torch.Tensor] = {k: torch.tensor(
            v, dtype=torch.float32, device=self.device) for k, v in self.penetration_keypoints_dict.items()}
        self.penetration_keypoints_dict = {k: F.pad(
            v, (0, 1), mode='constant', value=1.0) for k, v in self.penetration_keypoints_dict.items()}

        self.hand_keypoints_dict = json.load(
            open(os.path.join("data/urdf/", 'kpts_%s.json' % hand_model)))
        self.hand_keypoints_dict: Dict[str, torch.Tensor] = {k: torch.tensor(
            v, dtype=torch.float32, device=self.device) for k, v in self.hand_keypoints_dict.items()}
        self.hand_keypoints_dict = {k: F.pad(
            v, (0, 1), mode='constant', value=1.0) for k, v in self.hand_keypoints_dict.items()}

        visual: Robot = URDF.from_xml_string(open(urdf_filename).read())

        self.mesh_verts = {}
        self.mesh_faces = {}
        self.num_links = len(visual.links)
        self.num_contacts = len(visual.links)
        self.link_contact_idxs = torch.zeros(
            [len(visual.links)], dtype=torch.long, device=device)

        self.contact_to_link_name = []
        self.contact_permutations = [list(p)
                                     for p in permutations(np.arange(3))]

        self.link_idx_to_pts_idx = {}

        self.n_pts = 0

        contact_point_basis = {}
        contact_normals = {}
        self.link_rect_contact_area_cnt = {}
        self.link_names = [link.name for link in visual.links]

        for i_link, link in enumerate(visual.links):
            link: Link
            if len(link.visuals) == 0:
                continue

            mesh = get_mesh(link.visuals[0].geometry, mesh_path, hand_model)

            scale = link.visuals[0].geometry.scale
            if scale is None:
                scale = np.array([[1, 1, 1]])
            else:
                scale = np.array(
                    link.visuals[0].geometry.scale).reshape([1, 3])

            origin = link.visuals[0].origin
            if origin is None:
                rotation = transforms3d.euler.euler2mat(0, 0, 0)
                translation = np.array([[0, 0, 0]])
            else:
                rotation = transforms3d.euler.euler2mat(*origin.rpy)
                translation = np.reshape(origin.xyz, [1, 3])

            num_part_pts = int(mesh.area * pts_density)
            self.link_idx_to_pts_idx[link.name] = torch.tensor(
                np.arange(num_part_pts) + self.n_pts).to(self.device)
            pts = mesh.sample(num_part_pts) * scale  # (n_pts, 3
            self.n_pts += num_part_pts

            # Surface Points
            pts = np.matmul(rotation, pts.T).T + translation
            pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
            self.surface_points[link.name] = torch.from_numpy(
                pts).to(device).float().unsqueeze(0)  # (1, n_pts, 4)

            # Visualization Mesh
            self.mesh_verts[link.name] = np.array(mesh.vertices) * scale
            self.mesh_verts[link.name] = np.matmul(
                rotation, self.mesh_verts[link.name].T).T + translation
            self.mesh_faces[link.name] = np.array(mesh.faces)

            # Contact Points
            if link.name in self.contact_point_dict:
                cpb = np.array(self.contact_point_dict[link.name])
                self.link_rect_contact_area_cnt[link.name] = len(cpb)

                basis, normals = [], []

                for basis_indices in cpb:
                    self.contact_to_link_name.append(link.name)
                    cp_basis = mesh.vertices[basis_indices] * scale
                    cp_basis = np.matmul(rotation, cp_basis.T).T + translation
                    cp_basis = torch.cat([torch.from_numpy(cp_basis).to(
                        device).float(), torch.ones([4, 1]).to(device).float()], dim=-1)
                    basis.append(cp_basis)

                    v1 = cp_basis[1, :3] - cp_basis[0, :3]
                    v2 = cp_basis[2, :3] - cp_basis[0, :3]
                    v1 = v1 / (torch.norm(v1) + 1e-12)
                    v2 = v2 / (torch.norm(v2) + 1e-12)
                    normal = torch.linalg.cross(v1, v2).view([1, 3])

                    normals.append(normal)
                if len(cpb) > 0:
                    # 1 x N_areas x 4 x 4
                    contact_point_basis[link.name] = torch.stack(
                        basis, dim=0).unsqueeze(0)
                    # 1 x N_areas x 3
                    contact_normals[link.name] = torch.cat(
                        normals, dim=0).unsqueeze(0)

        basis = torch.cat([v for k, v in contact_point_basis.items()], dim=1).squeeze(
            0)  # (n_areas, 4, 4)
        normals = torch.cat(
            [v for k, v in contact_normals.items()], dim=1).squeeze(0)  # (n_areas, 3)

        self.rect_contact_areas = RectContactAreas(
            basis, normals, batch_size, device)

        # new 2.1
        self.revolute_joints = []
        for i in range(len(self.robot_full.joints)):
            if self.robot_full.joints[i].joint_type == 'revolute' or self.robot_full.joints[i].joint_type == 'continuous':
                self.q_len += 1
                self.revolute_joints.append(self.robot_full.joints[i])
        self.revolute_joints_q_mid = []
        self.revolute_joints_q_var = []
        self.revolute_joints_q_upper = []
        self.revolute_joints_q_lower = []
        for i in range(len(self.joint_param_names)):
            for j in range(len(self.revolute_joints)):
                if self.revolute_joints[j].name == self.joint_param_names[i]:
                    joint = self.revolute_joints[j]
            assert joint.name == self.joint_param_names[i]
            self.revolute_joints_q_mid.append(
                (joint.limit.lower + joint.limit.upper) / 2)
            self.revolute_joints_q_var.append(
                ((joint.limit.upper - joint.limit.lower) / 2) ** 2)
            self.revolute_joints_q_lower.append(joint.limit.lower)
            self.revolute_joints_q_upper.append(joint.limit.upper)

        self.revolute_joints_q_mid = torch.Tensor(
            self.revolute_joints_q_mid).to(device)
        self.revolute_joints_q_lower = torch.Tensor(
            self.revolute_joints_q_lower).to(device)
        self.revolute_joints_q_upper = torch.Tensor(
            self.revolute_joints_q_upper).to(device)

        self.current_status: Dict[str, pk.Transform3d] = None

        self.canon_pose = torch.tensor(
            [0, 0, 0, 1, 0, 0, 0, 1, 0] + [0] * (self.q_len - 9), device=device, dtype=torch.float32)
        self.scale = hand_scale

        self.num_contacts = len(self.contact_to_link_name)
        self.full_cpi = torch.arange(
            0, self.num_contacts, dtype=torch.long, device=device)
        self.contact_dist_diag_mask = torch.eye(
            self.num_contacts, dtype=torch.float32, device=device) * 1e12
        self.to_all_contact_areas_cpi = torch.arange(
            0, self.num_contacts, device=self.device, dtype=torch.long).unsqueeze(0)
        self.full_cpw_zeros = torch.zeros(
            [1, self.num_contacts, 4], dtype=torch.float32, device=device)
        # print(f"[{hand_model}] {self.num_contacts} contact points, {self.n_pts} surface points")

    def random_handcode(self, batch_size: int, table_top: bool = True) -> torch.Tensor:
        """Generate random handcode

        Args:
            batch_size (int): _description_
            table_top (bool, optional): _description_. Defaults to True.

        Returns:
            torch.Tensor: (batch_size, q_len)
        """
        transf = torch.normal(
            0, 1, [batch_size, 9], device=self.device, dtype=torch.float32)
        # joints = torch.rand([batch_size, self.q_len - 9], device=self.device, dtype=torch.float32)
        joints = torch.rand([batch_size, self.q_len - 9],
                            device=self.device, dtype=torch.float32) * 0.5
        joints = joints * (self.revolute_joints_q_upper -
                           self.revolute_joints_q_lower) + self.revolute_joints_q_lower
        q = torch.cat([transf, joints], dim=-1)

        q[:, 0:9] = 0.0
        if self.hand_model == "shadowhand":
            q[:, 1] = 0.3
        elif self.hand_model == "tonghand":
            q[:, 2] = 0.3

        if table_top:
            R_palm_down = transforms.euler_angles_to_matrix(torch.tensor(
                [torch.pi / 2, 0.0, 0.0]).unsqueeze(0).tile([batch_size, 1]).to(self.device), "XYZ")
            z = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).unsqueeze(
                0).tile([batch_size, 1])
            v = torch.rand([batch_size, 3], device=self.device,
                           dtype=torch.float32)
            v = v / torch.norm(v, dim=1).view(-1, 1)  # Normalize
            # Make sure that the z-component is positive
            v[:, -1] = torch.abs(v[:, -1]) + 0.5
            axis = torch.linalg.cross(z, v)
            axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-12) * \
                torch.acos(torch.clamp(
                    torch.sum(z * v, dim=1), -1, 1)).unsqueeze(-1)
            R_upper_sphere = transforms.axis_angle_to_matrix(axis)
            R = torch.matmul(R_upper_sphere, R_palm_down)
            R6 = compute_ortho6d_from_rotation_matrix(R)
        else:
            R6 = torch.normal(0, 1, [batch_size, 6],
                              dtype=torch.float32, device=self.device)
            R = compute_rotation_matrix_from_ortho6d(R6)

        q[:, 0:3] = torch.matmul(R, q[:, 0:3].unsqueeze(-1)).squeeze()
        q[:, 3:9] = R6.clone() + torch.normal(0, 0.1, [batch_size, 6],
                                              dtype=torch.float32, device=self.device)

        q = q.contiguous().clone()
        q.requires_grad_()
        return q

    def update_kinematics(self, q: torch.Tensor):
        """Update kinematics

        Args:
            q (torch.Tensor): (batch_size, q_len)
        """
        self.batch_size = q.shape[0]
        self.global_translation = q[:, :3] / self.scale  # (batch_size, 3)
        self.global_rotation = compute_rotation_matrix_from_ortho6d(
            q[:, 3:9])  # (batch_size, 3, 3)
        self.current_status = self.robot.forward_kinematics(q[:, 9:])

    def _get_trans_matrix(self):
        trans_matrix = []
        for link_name in self.link_names:
            if link_name in self.link_rect_contact_area_cnt:
                _trans_matrix = self.current_status[link_name].get_matrix().expand(
                    [self.batch_size,  4,
                        4])  # (n_batch, 4, 4)
                _trans_matrix = _trans_matrix.unsqueeze(
                    1)  # (n_batch, 1, 4, 4)
                cnt_area = self.link_rect_contact_area_cnt[link_name]
                _trans_matrix = _trans_matrix.repeat(1, cnt_area, 1, 1)

                trans_matrix.append(_trans_matrix)

        trans_matrix = torch.cat(trans_matrix, dim=1).contiguous()

        return trans_matrix  # (n_batch, n_areas, 4, 4)

    def get_contact_points(self, cpi: torch.Tensor, cpw: torch.Tensor):
        """_summary_

        Args:
            cpi (torch.Tensor): contact point index. Shape: (n_batch, n_contact)
            cpw (torch.Tensor): contact point weight. Shape: (n_batch, n_contact, 4)

        Returns:
            _type_: _description_
        """

        trans_matrix = self._get_trans_matrix()  # (n_batch, n_areas, 4, 4)

        cpb_trans = self.rect_contact_areas.get_contact_points(
            trans_matrix=trans_matrix, cpi=cpi, cpw=cpw, global_rotation=self.global_rotation, global_translation=self.global_translation)

        return cpb_trans * self.scale

    def self_penetration(self):
        points = self.get_penetration_keypoints()
        dis = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).norm(dim=-1)
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
        self_pntr_energy = torch.relu(0.015 - dis)
        return self_pntr_energy.sum(dim=[1, 2])

    def get_contact_areas(self, cpi: torch.Tensor):
        """_summary_

        Args:
            cpi (torch.Tensor): contact point index. Shape: (n_batch, n_contact)

        Returns:
            _type_: _description_
        """
        B, N_c = cpi.shape

        ones = torch.ones([B, N_c, 4]).to(cpi.device) * 1e-10

        contacts = []

        for i in range(4):
            ones_ = ones.clone()
            ones_[:, :, i] = 1e10
            areas, normal = self.get_contact_points_and_normal(cpi, ones_)
            areas = areas + normal * 1e-5
            contacts.append(areas)

        return torch.stack(contacts, dim=-2)

    def get_contact_points_and_normal(self, cpi: torch.Tensor, cpw: torch.Tensor):
        """_summary_

        Args:
            cpi (torch.Tensor): contact point index. Shape: (n_batch, n_contact)
            cpw (torch.Tensor): contact point weight. Shape: (n_batch, n_contact, 4)

        Returns:
            _type_: contact point basic, contact point normal. Shape: (n_batch, n_contact, 3), (n_batch, n_contact, 3)
        """

        trans_matrix = self._get_trans_matrix()  # (n_batch, n_areas, 4, 4)

        cpb_trans, cpn_trans = self.rect_contact_areas.get_contact_points_and_normal(
            trans_matrix=trans_matrix, cpi=cpi, cpw=cpw, global_rotation=self.global_rotation, global_translation=self.global_translation)

        # (n_batch, n_contact, 3), (n_batch, n_contact, 3)
        return cpb_trans * self.scale, cpn_trans

    def prior(self, q: torch.Tensor):
        """_summary_

        Args:
            q (torch.Tensor): (n_batch, q_len)

        Returns:
            _type_: _description_
        """
        range_energy = torch.relu(q[:, 9:] - self.revolute_joints_q_upper) + \
            torch.relu(self.revolute_joints_q_lower - q[:, 9:])
        return range_energy.sum(-1)

    def get_vertices(self):
        surface_points = []
        for link_name in self.surface_points:
            trans_matrix = self.current_status[link_name].get_matrix().expand([
                self.batch_size, 4, 4])
            surface_points.append(torch.matmul(trans_matrix, self.surface_points[link_name].repeat(
                self.batch_size, 1, 1).transpose(1, 2)).transpose(1, 2)[..., :3])
        surface_points[0] = surface_points[0].expand(
            [self.batch_size, surface_points[0].shape[1], 3])
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(
            1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return surface_points * self.scale

    def get_penetration_keypoints(self):
        kpts = []

        for link_name, canon_kpts in self.penetration_keypoints_dict.items():
            trans_matrix = self.current_status[link_name].get_matrix().expand([
                self.batch_size, 4, 4])
            kpts.append(torch.matmul(trans_matrix, canon_kpts.unsqueeze(0).expand(
                [self.batch_size, canon_kpts.shape[0], canon_kpts.shape[1]]).transpose(-1, -2)).transpose(-1, -2)[..., :3])
        kpts = torch.cat(kpts, 1)
        kpts = torch.matmul(self.global_rotation, kpts.transpose(1, 2)).transpose(
            1, 2) + self.global_translation.unsqueeze(1)
        return kpts * self.scale

    def get_hand_keypoints(self):
        kpts = []

        for link_name, canon_kpts in self.hand_keypoints_dict.items():
            trans_matrix = self.current_status[link_name].get_matrix().expand([
                self.batch_size, 4, 4])
            kpts.append(torch.matmul(trans_matrix, canon_kpts.unsqueeze(0).expand(
                [self.batch_size, canon_kpts.shape[0], canon_kpts.shape[1]]).transpose(-1, -2)).transpose(-1, -2)[..., :3])
        kpts = torch.cat(kpts, 1)
        kpts = torch.matmul(self.global_rotation, kpts.transpose(1, 2)).transpose(
            1, 2) + self.global_translation.unsqueeze(1)
        return kpts * self.scale

    def get_meshes_from_q(self, q=None, i=0, concat=True):
        data = []
        if q is not None:
            self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(
                len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(
            ), transformed_v.T).T + np.expand_dims(self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        if concat:
            data = tm.util.concatenate(data)
        return data

    def get_plotly_data(self, q=None, i=0, concat=True, color='lightpink', opacity=1.0, name='tonghand'):
        mesh = self.get_meshes_from_q(q)
        if concat:
            mesh = tm.util.concatenate(mesh)
            return go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color=color, opacity=opacity, name=name
            )
        else:
            return [
                go.Mesh3d(
                    x=m.vertices[:, 0],
                    y=m.vertices[:, 1],
                    z=m.vertices[:, 2],
                    i=m.faces[:, 0],
                    j=m.faces[:, 1],
                    k=m.faces[:, 2],
                    color=color, opacity=opacity, name=f"{name}_{i_m}"
                ) for i_m, m in enumerate(mesh)
            ]


robotic_hand_files = {
    "shadowhand":
    {
        "urdf_filepath": 'data/urdf/shadow_hand_description/shadowhand.urdf',
        "mesh_filepath": 'data/urdf/shadow_hand_description/meshes',
    },
    "allegro":
    {
        "urdf_filepath": 'data/urdf/allegro_hand_description/allegro_hand_description_left.urdf',
        "mesh_filepath": 'data/urdf/allegro_hand_description/meshes',
    },
}


def get_hand_model(hand_model, batch_size, device='cuda', **kwargs) -> RoboticHand:
    with torch.no_grad():
        filepaths = robotic_hand_files[hand_model]
        hand_model = RoboticHand(hand_model, filepaths['urdf_filepath'], filepaths['mesh_filepath'],
                                 specs_path=filepaths['specs'] if 'specs' in filepaths else None, batch_size=batch_size, device=device, **kwargs)

    return hand_model
