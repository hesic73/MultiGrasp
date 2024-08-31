
'''
Author: Aiden Li
Date: 2022-05-24 21:22:43
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-07-15 01:11:09
Description: Grasp Synthesis with MCMC with contact point weights
'''

import json
import os
from pytorch3d import transforms
import random
from datetime import datetime
from uuid import uuid4
import torch.nn.functional as F
import numpy as np
import torch
import trimesh as tm
from plotly import graph_objects as go
from tqdm import tqdm, trange
from hand_consts_allegro import get_contact_pool, contact_groups

from utils.HandModel import get_hand_model
from utils.ObjectModels import get_object_model
from utils.PhysicsGuide import PhysicsGuide
from utils.utils import *
from utils.visualize_plotly import plot_mesh, plot_point_cloud, plot_rect
from tensorboardX import SummaryWriter

from loguru import logger

from torch.optim.adam import Adam


from typing import List, Tuple

import typer
from argparse import Namespace


def get_cpi(contact_pools: List[torch.Tensor], batch_size: int, n_contact: int, device) -> torch.Tensor:
    cpi = []
    for i_contact_set, contact_pool in enumerate(contact_pools):
        _contact_pool = contact_pool.unsqueeze(0).tile([batch_size, 1])
        cpi_i = torch.randint(0, len(contact_pool), [
                              batch_size, n_contact], device=device, dtype=torch.long)
        contacts = torch.gather(_contact_pool, 1, cpi_i)
        cpi.append(contacts)
    cpi = torch.stack(cpi, dim=1)  # (batch_size, n_objects, n_contact)
    return cpi


def sample_object_positions_and_rotations(physics_guide: PhysicsGuide, n_objects: int, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    object_ps = []
    object_rs = []
    n_object_poses = 0

    # Sample table-top object placement until the batch_size is fulfilled
    while n_object_poses < batch_size:
        if n_objects > 3:
            proposals_xyz = torch.rand(
                [batch_size, n_objects, 3], device=device) * 0.15 - 0.075
        else:
            proposals_xyz = torch.rand(
                [batch_size, n_objects, 3], device=device) * 0.075 - 0.0375

        for i_object, object_model in enumerate(physics_guide.object_models):
            proposals_rot_i = torch.randint(0, len(object_model.stable_rotations), [
                                            batch_size], device=device, dtype=torch.long)
            proposals_rot = object_model.stable_rotations[proposals_rot_i]
            random_z_rot = torch.rand(
                [batch_size], dtype=torch.float32, device=device) * 2 * torch.pi
            random_z_rot_axis = F.pad(
                random_z_rot.unsqueeze(-1), (2, 0), 'constant', 0)
            random_z_rot_mat = transforms.axis_angle_to_matrix(
                random_z_rot_axis)
            proposals_rot = torch.matmul(random_z_rot_mat, proposals_rot)
            proposals_xyz[:, i_object,
                          2] = object_model.stable_zs[proposals_rot_i]
            object_model.update_pose(proposals_xyz[:, i_object], proposals_rot)

        oo_pen = physics_guide.oo_penetration()
        selected = torch.where(oo_pen < 0.0001)[0]
        n_append = min(len(selected), batch_size - n_object_poses)
        n_object_poses += n_append
        selected = selected[:n_append]
        object_ps.append(proposals_xyz[selected].clone())
        object_rs.append(torch.stack(
            [o.orient for o in physics_guide.object_models], dim=1)[selected].clone())

    object_ps = torch.cat(object_ps, dim=0).contiguous()
    object_rs = torch.cat(object_rs, dim=0).contiguous()

    return object_ps, object_rs


def synthesis(args, writer: SummaryWriter, export_dir: str):
    n_objects = len(args.object_models)

    transl_decay = 1.0

    uuids = [str(uuid4()) for _ in range(args.batch_size)]

    contact_group: List[List[int]] = contact_groups[args.contact_group]
    contact_pools: List[torch.Tensor] = [torch.tensor(get_contact_pool(
        g), dtype=torch.long, device=args.device) for g in contact_group[:n_objects]]

    logger.info("> Loading models...")
    hand_model = get_hand_model(
        args.hand_model, args.batch_size, device=args.device)
    physics_guide = PhysicsGuide(hand_model, args)

    for obj_name in args.object_models:
        object_model = get_object_model(
            obj_name, args.batch_size, scale=1.0, device=args.device)
        physics_guide.append_object(object_model)

    logger.info(
        f"  + Hand: { args.hand_model } ( {hand_model.num_links} links, {hand_model.num_contacts} contacts, {hand_model.n_pts} points )")
    logger.info(f"  + Contact groups: { contact_group }")
    logger.info(f"  + Objects: ({ ', '.join(args.object_models) })")
    logger.info(f"  + Export directory: { export_dir }")

    def sum_energy(new_fc_error: torch.Tensor, new_sf_dist: torch.Tensor, new_pntr: torch.Tensor, new_hprior: torch.Tensor, low_fc_w=False) -> torch.Tensor:
        new_energy = 0
        new_energy = new_energy + new_fc_error * args.fc_error_weight
        new_energy = new_energy + new_sf_dist * args.sf_dist_weight
        new_energy = new_energy + new_pntr * args.pen_weight
        new_energy = new_energy + new_hprior * args.hprior_weight
        return new_energy  # (batch_size,)

    logger.info("> Initializing hand...")
    q = hand_model.random_handcode(
        args.batch_size, table_top=True)  # (batch_size, q_len)

    cpw = torch.normal(0, 1, [args.batch_size, n_objects, args.n_contact,
                       4], requires_grad=True, device=args.device).float()  # (batch_size, n_objects, n_contact, 4)

    cpi = get_cpi(contact_pools, args.batch_size, args.n_contact,
                  args.device)  # (batch_size, n_objects, n_contact)

    logger.info("> Initializing objects...")

    object_ps, object_rs = sample_object_positions_and_rotations(
        physics_guide, n_objects, args.batch_size, args.device)  # (batch_size, n_objects, 3), (batch_size, n_objects, 3, 3)
    for i_object, object_model in enumerate(physics_guide.object_models):
        object_model.update_pose(
            object_ps[:, i_object], object_rs[:, i_object])

    torch.cuda.empty_cache()

    logger.info("> Starting optimization...")

    fc_error, sf_dist, pntr, hprior, norm_ali = physics_guide.compute_energy(
        cpi, cpw, q)
    energy_grad = sum_energy(fc_error, sf_dist, pntr, hprior, low_fc_w=True)
    energy = sum_energy(fc_error, sf_dist, pntr, hprior)

    grad_q, grad_w = torch.autograd.grad(
        energy_grad.sum(), [q, cpw], allow_unused=True)
    grad_q[:, :9] = grad_q[:, :9] * transl_decay

    ones = torch.arange(0, args.batch_size, device=args.device).long()

    # Steps with contact adjustment
    for step in tqdm(range(args.max_physics)):
        step_size = physics_guide.get_stepsize(step)  # (,)
        temperature = physics_guide.get_temperature(step)  # (,)
        new_q = q.clone()
        new_cpi = cpi.clone()

        q_grad_weight = max(args.max_physics * 0.8 - step,
                            0) / args.max_physics * 75 + 25

        # Updating handcode
        grad_q[:, 9:] = grad_q[:, 9:] / \
            (physics_guide.grad_ema_q.average.unsqueeze(0) + 1e-12)
        noise = torch.normal(mean=0, std=args.noise_size, size=new_q.shape,
                             device='cuda', dtype=torch.float) * step_size  # * disabled_joint_mask
        new_q = new_q + (noise - 0.5 * grad_q * step_size *
                         step_size) * physics_guide.joint_mask

        # Updating contact point indices (cpi)
        switch_contact = np.random.rand(1) < args.contact_switch
        if switch_contact:
            for i_contact, contact_pool in enumerate(contact_pools):
                update_indices = torch.randint(0, args.n_contact, size=[
                                               args.batch_size], device=args.device)
                update_to = torch.randint(0, contact_pool.shape[0], size=[
                                          args.batch_size], device=args.device)
                update_to = contact_pool[update_to]
                new_cpi[ones, i_contact, update_indices.long()] = update_to

        cpw = cpw - 0.5 * grad_w * step_size * step_size
        new_fc_error, new_sf_dist, new_pntr, new_hprior, new_norm_ali = physics_guide.compute_energy(
            new_cpi, cpw, new_q)
        new_energy_grad = sum_energy(
            new_fc_error, new_sf_dist, new_pntr, new_hprior)
        new_energy = sum_energy(
            new_fc_error, new_sf_dist, new_pntr, new_hprior)
        new_grad_q, new_grad_w = torch.autograd.grad(
            new_energy_grad.sum(), [new_q, cpw], allow_unused=True)
        new_grad_q = new_grad_q * physics_guide.joint_mask  # * disabled_joint_mask
        new_grad_q[:, 3:9] = new_grad_q[:, 3:9] * 10
        new_grad_q[:, 9:] = new_grad_q[:, 9:] * q_grad_weight

        with torch.no_grad():
            alpha = torch.rand(args.batch_size, device=args.device).float()
            accept = alpha < torch.exp((energy - new_energy) / temperature)
            q[accept] = new_q[accept]
            cpi[accept] = new_cpi[accept]
            energy[accept] = new_energy[accept]
            grad_q[accept] = new_grad_q[accept]
            grad_w[accept] = new_grad_w[accept]

            physics_guide.grad_ema_q.apply(grad_q[:, 9:] / q_grad_weight)

            if step % 100 == 99:
                tqdm.write(f"Step { step }, Energy: { energy.mean().detach().cpu().numpy()} "
                           + f"FC: { new_fc_error.mean().detach().cpu().numpy() } "
                           + f"PN: { new_pntr.mean().detach().cpu().numpy() } "
                           + f"SD: { (new_sf_dist / args.n_contact / n_objects).mean().detach().cpu().numpy() } "
                           + f"HP: { new_hprior.mean().detach().cpu().numpy() }")

            if args.viz and step % 1000 == 0:
                os.makedirs(os.path.join(export_dir, str(step)), exist_ok=True)
                hand_model.update_kinematics(q)
                contacts = [hand_model.get_contact_areas(
                    cpi[:, j]).cpu().numpy() for j in range(n_objects)]
                contact_pts = [hand_model.get_contact_points(
                    cpi[:, j], cpw[:, j]).cpu().numpy() for j in range(n_objects)]
                pntr_kpts = hand_model.get_penetration_keypoints().cpu().numpy()
                for i in range(8):
                    go.Figure([plot_mesh(o.get_obj_mesh(i), name=f"object-{j}") for j, o in enumerate(physics_guide.object_models)]
                              + [plot_rect(contacts[j][i, k], name=f"contact-{j}_{k}", color='red') for j in range(
                                  n_objects) for k in range(args.n_contact)]
                              + [plot_mesh(tm.load("data/table.stl"), color='green'), plot_mesh(
                                  hand_model.get_meshes_from_q(None, i, True), 'lightpink', opacity=1.0)]
                              ).write_html(os.path.join(export_dir, str(step), f"{i}.html"))

            if args.log and step % 10 == 9:
                writer.add_scalar("MALA/stepsize", step_size, step)
                writer.add_scalar("MALA/temperature", temperature, step)
                writer.add_scalar("MALA/mc_accept",
                                  accept.float().mean().detach().item(), step)
                writer.add_scalar(
                    "MALA/switch_contact", (switch_contact[0] * accept).float().mean().detach().item(), step)

                writer.add_scalar(
                    "Grasp/enery", energy.mean().detach().cpu().numpy(), step)
                writer.add_scalar(
                    "Grasp/fc_err", new_fc_error.mean().detach().cpu().numpy(), step)
                writer.add_scalar("Grasp/penetration",
                                  new_pntr.mean().detach().cpu().numpy(), step)
                writer.add_scalar(
                    "Grasp/distance", new_sf_dist.mean().detach().cpu().numpy(), step)
                writer.add_scalar(
                    "Grasp/hand_prior", new_hprior.mean().detach().cpu().numpy(), step)

    torch.cuda.empty_cache()

    logger.info("> Refining grasps...")
    args.sf_dist_weight = 20.0
    q = q.detach().clone().requires_grad_(True)
    cpw = cpw.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [{"params": q, "lr": 1e-2}, {"params": cpw, "lr": 1e-1}])

    # Refine steps
    for step in trange(args.max_refine):
        optimizer.zero_grad()
        fc_error, sf_dist, pntr, hprior, norm_ali = physics_guide.compute_energy(
            cpi, cpw, q)
        energy = sum_energy(fc_error, sf_dist, pntr, hprior).mean()
        energy.backward()
        optimizer.step()

        if step % 100 == 99:
            tqdm.write(f"Step { step }, Energy: { energy.mean().detach().cpu().numpy()} "
                       + f"FC: { fc_error.mean().detach().cpu().numpy() } "
                       + f"PN: { pntr.mean().detach().cpu().numpy() } "
                       + f"SD: { (sf_dist / args.n_contact / n_objects).mean().detach().cpu().numpy() } "
                       + f"HP: { hprior.mean().detach().cpu().numpy() }")

        if args.log and step % 10 == 9:
            writer.add_scalar(
                "Grasp/enery", energy.mean().detach().cpu().numpy(), step + args.max_physics)
            writer.add_scalar(
                "Grasp/fc_err", fc_error.mean().detach().cpu().numpy(), step + args.max_physics)
            writer.add_scalar(
                "Grasp/penetration", pntr.mean().detach().cpu().numpy(), step + args.max_physics)
            writer.add_scalar(
                "Grasp/distance", sf_dist.mean().detach().cpu().numpy(), step + args.max_physics)
            writer.add_scalar(
                "Grasp/hand_prior", hprior.mean().detach().cpu().numpy(), step + args.max_physics)

        if args.viz and step % 1000 == 0:
            os.makedirs(os.path.join(export_dir, str(step)), exist_ok=True)
            hand_model.update_kinematics(q)
            contacts = [hand_model.get_contact_areas(
                cpi[:, j]).detach().cpu().numpy() for j in range(n_objects)]
            contact_pts = [hand_model.get_contact_points(
                cpi[:, j], cpw[:, j]).detach().cpu().numpy() for j in range(n_objects)]
            pntr_kpts = hand_model.get_penetration_keypoints().detach().cpu().numpy()
            for i in range(8):
                go.Figure([plot_mesh(o.get_obj_mesh(i), name=f"object-{j}") for j, o in enumerate(physics_guide.object_models)]
                          + [plot_point_cloud(pntr_kpts[i],
                                              name=f"pntr-kpts", surfacecolor='blue')]
                          + [plot_point_cloud(contact_pts[j][i], name=f"contact-{j}") for j in range(n_objects)]
                          + [plot_rect(contacts[j][i, k], name=f"contact-{j}_{k}", color='red') for j in range(
                              n_objects) for k in range(args.n_contact)]
                          + [plot_mesh(tm.load("data/table.stl"), color='green'), plot_mesh(
                              hand_model.get_meshes_from_q(None, i, True), 'lightpink', opacity=1.0)]
                          ).write_html(os.path.join(export_dir, str(step), f"{i}.html"))

    logger.info("> Saving checkpoint...")

    fc_error, sf_dist, pntr, hprior, norm_ali = physics_guide.compute_energy(
        cpi, cpw, q, reduce=False)
    save(args, export_dir, uuids, physics_guide, q, cpi,
         cpw, fc_error, sf_dist, pntr, hprior, norm_ali)


def save(args, export_dir, uuids, physics_guide, q, cpi, cpw, fc_error, sf_dist, pntr, hprior, norm_ali, step=None):
    with torch.no_grad():
        save_dict = {
            "args": args, "uuids": uuids,
            "object_models": args.object_models,
            "q": q, "cpi": cpi, "cpw": cpw,
            "obj_scales": [obj.scale for obj in physics_guide.object_models],
            "obj_poses": [obj.get_poses() for obj in physics_guide.object_models],
            "joint_mask": physics_guide.joint_mask,
            "quantitative": {
                "fc_loss": fc_error,
                "pen": pntr,
                "surf_dist": sf_dist,
                "hand_pri": hprior,
                "norm_ali": norm_ali
            }
        }

        if step is None:
            torch.save(save_dict, os.path.join(export_dir, f"ckpt.pt"))
        else:
            torch.save(save_dict, os.path.join(export_dir, f"ckpt-{step}.pt"))


def main(
    batch_size: int = 1024,
    max_physics: int = 6500,
    max_refine: int = 1500,
    hand_model: str = "allegro",
    n_contact: int = 3,
    object_models: list[str] = typer.Option(
        ["cube", "cube"], help="List of object models"),
    num_obj_pts: int = 256,
    starting_temperature: float = 8.0,
    contact_switch: float = 0.25,
    temperature_decay: float = 0.95,
    stepsize_period: int = 100,
    annealing_period: int = 50,
    contact_group: int = 0,
    langevin_probability: float = 0.85,
    noise_size: float = 0.01,
    fc_error_weight: float = 1.0,
    hprior_weight: float = 10.0,
    pen_weight: float = 10.0,
    sf_dist_weight: float = 10.0,
    hc_pen: bool = False,
    viz: bool = False,
    log: bool = False,
    levitate: bool = False,
    seed: int = 42,
    output_dir: str = "synthesis",
    tag: str = "debug"
):

    # Computation device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    args = Namespace(
        batch_size=batch_size,
        max_physics=max_physics,
        max_refine=max_refine,
        hand_model=hand_model,
        n_contact=n_contact,
        object_models=object_models,
        num_obj_pts=num_obj_pts,
        starting_temperature=starting_temperature,
        contact_switch=contact_switch,
        temperature_decay=temperature_decay,
        stepsize_period=stepsize_period,
        annealing_period=annealing_period,
        contact_group=contact_group,
        langevin_probability=langevin_probability,
        noise_size=noise_size,
        fc_error_weight=fc_error_weight,
        hprior_weight=hprior_weight,
        pen_weight=pen_weight,
        sf_dist_weight=sf_dist_weight,
        hc_pen=hc_pen,
        viz=viz,
        log=log,
        levitate=levitate,
        seed=seed,
        output_dir=output_dir,
        tag=tag,
        device=device
    )

    # Time tag and export directories
    time_tag = datetime.now().strftime('%Y-%m/%d/%H-%M-%S')
    base_dir = f"{ str(output_dir) }/{ hand_model }/{ time_tag }_{ '+'.join(object_models) }-seed_{seed}-{tag}"

    logger.add(os.path.join(base_dir, "log.txt"),
               rotation="10 MB", format="{time} {level} {message}")
    logger.info(f"Logging to { os.path.join(base_dir, 'log.txt') }")

    os.makedirs(base_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    writer = SummaryWriter(logdir=base_dir)

    json.dump(vars(args), open(os.path.join(
        base_dir, "args.json"), 'w'), default=lambda o: str(o))

    synthesis(args, writer, base_dir)


if __name__ == '__main__':
    typer.run(main)
