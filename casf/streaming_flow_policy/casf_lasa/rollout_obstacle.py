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

from streaming_flow_policy.pusht.dp_state_notebook.all import (
    normalize_data, unnormalize_data
)
from torchdiffeq import odeint

@torch.inference_mode()
def rollout_lasa(policy, dataset_ts, stats, max_steps: int = 3000, 
                 action_horizon=16, obs_horizon=2, device="cuda", 
                 save_path: Optional[str] = "result/lasa_rollout", save_fileName_prefix: Optional[str] = "lasa_rollout_segmented_norm",
                 apply_shaping:str=None, constraint_config:Optional[dict]=None,):
    os.makedirs(save_path, exist_ok=True)
    results = []
    print(f"=== Evaluating LASA rollout ({len(dataset_ts)} demos) ===")

    #########-----------------------------------#########
    ## defining the real-obstacle position
    xmin_norm, ymin_norm = -1,-1
    xmax_norm, ymax_norm = 1,1
    max_span_norm = max((xmax_norm-xmin_norm), (ymax_norm-ymin_norm)) # 2
    
    # defining normalized-obstacle position
    workspace_norm_size   = np.array([max_span_norm, max_span_norm]) # [2,2]
    center_scale=np.array(constraint_config['center_scale'], dtype=np.float32) # [0.25,0.25]
    center_norm = np.array([
        xmin_norm + center_scale[0] * workspace_norm_size[0],   # 25% from the left edge
        ymax_norm - center_scale[1] * workspace_norm_size[1],   # 25% down from the top
    ])
    radius_norm = constraint_config['radius_scale'] * np.mean(workspace_norm_size)  # radius_scale=0.15 defines radius proportional to workspace size
    constraint_config['center_norm'] = center_norm
    constraint_config['radius_norm'] = radius_norm

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
        # print('CHECKING obs_start.shape --> ', obs_start.shape)

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
            # print('nobs-obs_seq', obs_seq.shape) # (2,4)
            nobs = torch.from_numpy(obs_seq).to(device, dtype=torch.float32).unsqueeze(0)  # (1,obs_hor,obs_dim)
            # print('nobs-obs_deque', nobs.shape) # (1,2,4)
            
            # ---- rollout through the policy ----
            # print('check in rollout_lasa nobs.shape',nobs.shape)
            naction,nvel = policy(nobs, num_actions=action_horizon, postShaping=apply_shaping, shapingConfig=constraint_config)  # (1, num_actions, 2)
            naction = naction.detach().cpu().numpy()[0]         # (num_actions, 2)
            nvel = nvel.detach().cpu().numpy()[0]         # (num_actions, 2)
            # convert to real coordinate
            naction_real = unnormalize_data(naction, stats["action"])

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
            # next_obs = naction
            
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

        viz_examples.append((center_norm, radius_norm, gt_all, pred_traj_norm, pred_v, mse_norm, final_dist_norm))

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
        for ax, (cNorm, rNorm, gt, pred, predv, mse, dist) in zip(axes, viz_examples):

            # Before drawing
            # ensure all matplotlib-safe
            if torch.is_tensor(cNorm):
                cNorm = cNorm.detach().cpu().numpy()
            if torch.is_tensor(rNorm):
                rNorm = float(rNorm.detach().cpu().item())
            circ_norm = plt.Circle(cNorm, rNorm, color='orange', alpha=0.3)
            ax.add_patch(circ_norm)
            
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
        save_file_norm = save_fileName_prefix+f'_norm_postShaping{apply_shaping}.png'
        savefile = os.path.join(save_path, save_file_norm)
        plt.savefig(savefile, dpi=300)
        plt.close(fig)
        print(f"📈 Saved rollout visualization to {savefile}")

    return results
