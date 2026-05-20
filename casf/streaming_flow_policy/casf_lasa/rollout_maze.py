import torch
import numpy as np
from tqdm import tqdm
from torchdyn.core import NeuralODE
from streaming_flow_policy.pusht.dp_state_notebook.all import (
    normalize_data, unnormalize_data, Policy
)
from streaming_flow_policy.lasa.sfpd import VectorFieldWrapper
import matplotlib.pyplot as plt
import os
import collections
from torch import Tensor
from pydrake.all import PiecewisePolynomial

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Optional, Dict
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree

from streaming_flow_policy.pusht.dp_state_notebook.all import (
    normalize_data, unnormalize_data
)


def build_s_corridor_union(demos, w=0.05, grid_res=200, margin=0.1):
    """
    demos: list of arrays (N_i x 2)
    w: corridor half-width (like wall thickness)
    grid_res: resolution of grid for visualization
    margin: padding around the demo space
    """
    # 1. Collect all demo points
    all_points = np.concatenate(demos, axis=0)
    xmin, ymin = all_points.min(0) - margin
    xmax, ymax = all_points.max(0) + margin

    # 2. Create grid
    x = np.linspace(xmin, xmax, grid_res)
    y = np.linspace(ymin, ymax, grid_res)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)

    # 3. Create binary mask of where demos are
    mask = np.zeros(X.shape, dtype=bool)
    for d in demos:
        # nearest grid indices for each trajectory point
        xi = np.searchsorted(x, d[:, 0])
        yi = np.searchsorted(y, d[:, 1])
        xi = np.clip(xi, 0, grid_res - 1)
        yi = np.clip(yi, 0, grid_res - 1)
        mask[yi, xi] = True

    # 4. Compute distance transform from the demo region
    dist_out = distance_transform_edt(~mask) * (x[1] - x[0])  # convert to same scale
    sdf = dist_out - w   # inside if sdf < 0 (within w of any demo)
    
    inside = sdf < 0   # Boolean mask: True = inside corridor
    corridor_pixels = np.stack([X[inside], Y[inside]], axis=-1)

    return X, Y, sdf, mask, corridor_pixels


def build_even_s_corridor(demos, width=0.05, grid_res=300, margin=0.1, smoothness=0.5):
    """
    Build an evenly-thick, sharp S corridor by fitting a unified spline centerline.

    Args:
        demos: list of arrays (N_i, 2)  -- 7 demo trajectories
        width: corridor half-width (in LASA coordinate scale)
        grid_res: grid resolution for SDF
        margin: padding around demo space
        smoothness: smoothing factor for spline fit (0 = interpolate all points)
    Returns:
        X, Y, sdf, centerline, corridor_pixels
    """

    # 1️⃣ Gather all points
    all_points = np.concatenate(demos, axis=0)

    # 2️⃣ Sample each demo evenly along arc length
    resampled = []
    for d in demos:
        t = np.linspace(0, 1, len(d))
        t_new = np.linspace(0, 1, 200)
        x_spline, _ = splprep(d.T, s=smoothness)
        x_eval = np.stack(splev(t_new, x_spline), axis=1)
        resampled.append(x_eval)
    all_resampled = np.concatenate(resampled, axis=0)

    # 3️⃣ Fit a smooth spline through the mean curve (approx centerline)
    # average by phase along S
    mean_traj = np.mean(np.stack(resampled, axis=2), axis=2)
    tck, _ = splprep(mean_traj.T, s=smoothness)
    u = np.linspace(0, 1, 600)
    centerline = np.stack(splev(u, tck), axis=1)

    # 4️⃣ Create workspace grid
    xmin, ymin = all_points.min(0) - margin
    xmax, ymax = all_points.max(0) + margin
    x = np.linspace(xmin, xmax, grid_res)
    y = np.linspace(ymin, ymax, grid_res)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)

    # 5️⃣ Compute signed distance to centerline (KDTree for efficiency)
    tree = KDTree(centerline)
    dist, _ = tree.query(XY)
    sdf = dist.reshape(X.shape) - width     # negative inside

    # --- 6️⃣[optional] Flat start/end caps ---
    v_start = centerline[5] - centerline[0]
    v_start /= np.linalg.norm(v_start) + 1e-8
    v_end = centerline[-1] - centerline[-6]
    v_end /= np.linalg.norm(v_end) + 1e-8

    start_plane = np.dot(XY - centerline[0], v_start)
    end_plane   = np.dot(XY - centerline[-1], v_end)

    outside_start = start_plane < -width
    outside_end   = end_plane >  width
    cap_mask = ~(outside_start | outside_end)

    sdf[~cap_mask.reshape(X.shape)] = np.maximum(sdf[~cap_mask.reshape(X.shape)], 0)
    # --- 6️⃣[optional]  Flat start/end caps ---

    inside = sdf < 0
    corridor_pixels = np.stack([X[inside], Y[inside]], axis=-1)

    return X, Y, sdf, centerline, corridor_pixels

@torch.inference_mode()
def rollout_lasa(policy, dataset_ts, stats, max_steps: int = 3000, 
                 action_horizon=16, obs_horizon=2, device="cuda", 
                 save_path: Optional[str] = "result/lasa_rollout", save_fileName_prefix: Optional[str] = "lasa_rollout_segmented_norm",
                 apply_shaping:str=None, constraint_config:Optional[dict]=None,):

    os.makedirs(save_path, exist_ok=True)
    results = []
    print(f"=== Evaluating LASA rollout ({len(dataset_ts)} demos) ===")

    #########-----------------------------------#########
    corridorX, corridorY, corridorSDF, cooridorCLine, corridorPixels = build_even_s_corridor(dataset_ts.demos_point_norm, width=0.2)
    constraint_config['corridorX'] = corridorX
    constraint_config['corridorY'] = corridorY
    constraint_config['corridorSDF'] = corridorSDF
    constraint_config['cooridorCLine'] = cooridorCLine
    constraint_config['corridorPixels'] = corridorPixels

    #########-----------------------------------#########

    viz_examples = []
    mse_lst, mse_norm_lst = [], []
    final_dist_lst, final_dist_norm_lst = [], []

    for d_idx, demo in enumerate(tqdm(dataset_ts, desc="Rollout LASA")):
        # load normalized observation
        pos_all = demo["pos"]  # (T, 2) # # normalized
        vel_all = demo["vel"]   # (T, 2) # normalized
        obs_start = demo["obs"]  # (2, 4) # normalized
        gt_all = demo["action"]  # (T, 2) # # normalized
        gt_all_real = unnormalize_data(gt_all, stats['action'])
        T = gt_all.shape[0]
        
        obs_dim = obs_start.shape[1]
        print('CHECKING obs_start.shape --> ', obs_start.shape)

        # obs_start = obs_all[:policy.obs_horizon.item()]
        # obs_start[:,-2:] = prev_vs[:policy.obs_horizon.item()]
        # print('debug --> ',obs_start[:,:2], gt_all[:policy.obs_horizon.item()])
        ## obs_start[-1,:2] == gt_all[0]
        
        # initialize rollout buffer
        # obs_deque = obs_start # # normalized
        # obs_deque = collections.deque([obs_all[0]] * obs_horizon, maxlen=obs_horizon)
        obs_deque = collections.deque(obs_start, maxlen=obs_horizon)

        step = 0 

        pred_traj_norm = []
        pred_traj_real = []
        pred_v = []
        
        ## NOTE
        pbar = tqdm(total=max_steps, desc=f"Demo {d_idx+1}", leave=False)
        while step < max_steps:
            # print("ROLLOUT-check obs_deque", obs_deque.shape)
            # assemble observation sequence
            obs_seq = np.stack(obs_deque)  # (obs_horizon, obs_dim)
            nobs = torch.from_numpy(obs_seq).to(device, dtype=torch.float32).unsqueeze(0)  # (1,obs_hor,obs_dim)
            # print('nobs-obs_deque', nobs.shape)
            
            # ---- rollout through the policy ----
            # naction,nvel = policy(nobs, num_actions=action_horizon)  # (1, num_actions, 2) 
            naction,nvel = policy(nobs, num_actions=action_horizon, postShaping=apply_shaping, shapingConfig=constraint_config)  # (1, num_actions, 2)

            naction = naction.detach().cpu().numpy()[0]         # (num_actions, 2)
            nvel = nvel.detach().cpu().numpy()[0]         # (num_actions, 2)
            # convert to real coordinate
            naction_real = unnormalize_data(naction, stats["action"])
            
            # ## ------------------------ ##
            # # dt = 1/(policy.pred_horizon.item()-1) # 1/15
            # # vel_seg = (naction[1:, :] - naction[:-1, :]) / dt # (7,2)
            # traj_times_full = np.linspace(0, 1, policy.pred_horizon.item())
            # traj_times_seg = traj_times_full[:action_horizon]
            # traj_seg = PiecewisePolynomial.FirstOrderHold(traj_times_seg, naction.T)
            # x_seg = [traj_seg.value(t).T for t in traj_times_seg]  # (8,2)
            # vel_seg = [traj_seg.EvalDerivative(t).T for t in traj_times_seg]  # (8,2)
            
            # x_seg = np.squeeze(np.asarray(x_seg))
            # vel_seg = np.squeeze(np.asarray(vel_seg))
            # # print('DEBUG** x_seg, vel_seg', x_seg.shape, vel_seg.shape) # x_seg, vel_seg (8, 2) (8, 2)
            # x_seg_real = unnormalize_data(x_seg, stats["action"])
            # ## ------------------------ ##

            # print("mean |vel_seg|:", np.linalg.norm(vel_seg, axis=-1).mean())
            # print("mean |nvel|:", np.linalg.norm(nvel, axis=-1).mean())
            # print(error)
            
            # append predicted segment
            if len(pred_traj_norm) == 0:
                pred_traj_norm.extend(list(naction)) # (8,2)
                pred_traj_real.extend(list(naction_real)) 
                pred_v.extend(list(nvel)) # (7,2)
                # pred_v.extend(vel_all[step:step+action_horizon])  # append last vel to keep alignment
            else:
                pred_traj_norm.extend(list(naction[1:])) # (7,2)
                pred_traj_real.extend(list(naction_real[1:])) # (7,2)
                pred_v.extend(list(nvel[1:])) # (7,2)
                # pred_v.extend(vel_all[step:step+action_horizon-1])

            next_obs = np.concatenate((naction, nvel), axis=-1) # (8,4)
            # overlap = 2  # how many points to keep
            # next_obs = next_obs[-overlap:] # (2,4)
            # obs_deque = next_obs
            for nexObs in next_obs:
                obs_deque.append(nexObs)
            # print("ROLLOUT-check obs_deque", obs_deque.shape)
                
            # fig, axes = plt.subplots(1, 1, figsize=(3,3), sharex=False)
            # # axes.scatter(naction[:,0], naction[:,1], c='r', marker='o',s=1, label='LASA traj')
            # axes.plot(naction[:,0], naction[:,1], 'k--', alpha=0.5, label='LASA traj')
            # axes.quiver(naction[:,0], naction[:,1], vel_all[step:step+action_horizon,0], vel_all[step:step+action_horizon,1],
            #         color='red', alpha=0.4, label='naction - velAll', scale=1)
            # axes.set_title("Velocity comparison");axes.set_xlabel("Normalized time (t ∈ [0,1])");axes.legend()
            # plt.tight_layout()
            # plt.savefig("./naction_velAll_check.png", dpi=150)
            # print(f"Saved LASA dataset visualization to: ./naction_velAll_check.png")

            ## the first action should be the last observed position
            # advance step (overlap one obs horizon)
            step += (action_horizon - 1) # move by (H-1), since last point overlaps
            
            pbar.update(action_horizon - 1)
        pbar.close()

        pred_traj_norm = np.array(pred_traj_norm)
        pred_traj_real = np.array(pred_traj_real)
        pred_v = np.array(pred_v)
        
        results.append({
            "gt_real": gt_all_real[:len(pred_traj_real)],
            "pred_real": pred_traj_real,
            "gt_norm": gt_all[:len(pred_traj_norm)],
            "pred_norm": pred_traj_norm
        })

        mse = np.mean((pred_traj_real[:len(gt_all)] - gt_all_real[:len(pred_traj_real)]) ** 2)
        final_dist = np.linalg.norm(pred_traj_real[-1] - gt_all_real[:len(pred_traj_real)][-1])
        mse_norm = np.mean((pred_traj_norm[:len(gt_all)] - gt_all[:len(pred_traj_norm)]) ** 2)
        final_dist_norm = np.linalg.norm(pred_traj_norm[:len(gt_all)][-1] - gt_all[:len(pred_traj_norm)][-1])

        mse_lst.append(mse)
        final_dist_lst.append(final_dist)
        mse_norm_lst.append(mse_norm)
        final_dist_norm_lst.append(final_dist_norm)

        print(f"Demo {d_idx+1}: MSE={mse:.6f}, FinalDist={final_dist:.6f} | "
              f"MSE_NORM={mse_norm:.6f}, FinalDist_NORM={final_dist_norm:.6f}")

        print('check pred_traj_norm.shape, pred_v.shape', pred_traj_norm.shape, pred_v.shape)
        viz_examples.append((constraint_config, gt_all, pred_traj_norm, pred_v, mse_norm, final_dist_norm))

    # ---- Summary ----
    results = {
        "MSE": float(np.mean(mse_lst)),
        "MSE_NORM": float(np.mean(mse_norm_lst)),
        "FinalDist": float(np.mean(final_dist_lst)),
        "FinalDist_NORM": float(np.mean(final_dist_norm_lst)),
    }

    print(f"✅ Rollout done | MSE={results['MSE']:.4e}, "
          f"FinalDist={results['FinalDist']:.4f}, "
          f"MSE_NORM={results['MSE_NORM']:.4e}, "
          f"FinalDist_NORM={results['FinalDist_NORM']:.4f}")

    # ---- Visualization ----
    if viz_examples:
        fig, axes = plt.subplots(len(viz_examples), 1, figsize=(3,3*len(viz_examples)), sharex=False)
        # ensure axes is always iterable
        if len(viz_examples) == 1:
            axes = [axes]
            
        ax_ind=0
        for ax, (mazeConfig, gt, pred, predv, mse, dist) in zip(axes, viz_examples):
            # mazeConfig['corridorX']; mazeConfig['corridorY']; mazeConfig['corridorSDF']; mazeConfig['cooridorCLine']; mazeConfig['corridorPixels'] 

            ax.contourf(mazeConfig['corridorX'], mazeConfig['corridorY'], mazeConfig['corridorSDF'], 
                            levels=[-1e9, 0, 1e9],   # below 0 = inside corridor, above 0 = outside
                            colors=["#F5D48F", "white"],   # inside (yellow), outside (white)
                            alpha=1)
            
            ax.plot(gt[:, 0], gt[:, 1], "g-", lw=2, label="GT")
            ax.plot(pred[:, 0], pred[:, 1], "r--", lw=1.5, label="Pred")

            # Mark start (star) and end (black circle) points for both GT and prediction
            ax.scatter(gt[0, 0], gt[0, 1], marker="*", color="green", s=80, label="GT Start")
            ax.scatter(gt[-1, 0], gt[-1, 1], marker="o", color="green", s=60, label="GT End")
            ax.scatter(pred[0, 0], pred[0, 1], marker="*", color="red", s=80, label="Pred Start")
            ax.scatter(pred[-1, 0], pred[-1, 1], marker="o", color="red", s=60, label="Pred End")
            ax.quiver(pred[::20, 0], pred[::20, 1], predv[::20,0], predv[::20,1],
                    color='red', alpha=0.5, label='pos - predV', scale=1)

            ax.legend()
            ax.set_title(f"demo{ax_ind}: MSE={mse:.4f}\nFinalDist={dist:.4f}")
            ax_ind += 1

        plt.tight_layout()
        save_file_norm = save_fileName_prefix+'_norm.png'
        savefile = os.path.join(save_path, save_file_norm)
        plt.savefig(savefile, dpi=300)
        plt.close(fig)
        print(f"📈 Saved rollout visualization to {savefile}")

    return results
