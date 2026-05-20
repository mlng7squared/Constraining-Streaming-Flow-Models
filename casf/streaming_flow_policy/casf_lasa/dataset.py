import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
import glob
import scipy.io as sio
from typing import Dict, Optional, Callable
from pydrake.all import PiecewisePolynomial
import matplotlib.pyplot as plt
CHECK_VISU = False
# ----------------------------
# helpers (copied from PushT)
# ----------------------------
def create_sample_indices(episode_ends, sequence_length, pad_before=0, pad_after=0):
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx   = min(idx+sequence_length, episode_length) + start_idx
            start_offset     = buffer_start_idx - (idx+start_idx)
            end_offset       = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx   = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx
            ])
    return np.array(indices)

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = {}
    for key, arr in train_data.items():
        sample = arr[buffer_start_idx:buffer_end_idx]  # (S,K)
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros((sequence_length,) + arr.shape[1:], dtype=arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data: np.ndarray):
    """
    Args:
        data (np.ndarray, shape=(..., K)): Data to compute statistics over.
    """
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data: np.ndarray, stats: Dict[str, np.ndarray]):
    """
    Args:
        data (np.ndarray, shape=(..., K)): Data to normalize.
        stats (Dict[str, np.ndarray]): Statistics to use for normalization.
    """
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata: np.ndarray, stats: Dict[str, np.ndarray]):
    """
    Args:
        ndata (np.ndarray, shape=(..., K)): Normalized data to un0normalize.
        stats (Dict[str, np.ndarray]): Statistics to use for un-normalization.
    """
    # unnormalize to [0, 1]
    ndata = (ndata + 1) / 2
    # unnormalize to original range
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# plotting utilities
def plot_trajectory(ax, trajectory: np.ndarray, color='red', **plot_kwargs):
    """
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on.
        trajectory (np.ndarray, shape=(T, 2)): Trajectory to plot.
    """
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, **plot_kwargs)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], marker='o', color=color, label='start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], marker='x', color=color, label='end')
    return ax

def plot_velocity(ax, trajectory: np.ndarray, velocity: np.ndarray, scale: float=0.1, color='blue', **quiver_kwargs):
    """
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on.
        trajectory (np.ndarray, shape=(T, 2)): Trajectory to plot.
        velocity (np.ndarray, shape=(T, 2)): Velocity to plot.
        scale (float): Scale for quiver arrows.
    """
    if trajectory.dim != 2 or velocity.dim != 2:
        raise ValueError("trajectory and velocity must be 2D arrays.")
    ax.quiver(
        trajectory[:, 0], trajectory[:, 1],
        velocity[:, 0], velocity[:, 1],
        angles='xy', scale_units='xy', scale=1/scale,
        color=color,
        **quiver_kwargs
    )
    return ax
    
# ----------------------------
class MatSequenceDatasetTask(Dataset):
    def __init__(self, dataset_dir: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 obs_dim: int,
                 action_horizon: int,
                 task: str = 'Sshape',
                 split: str="train",
                 split_config: dict=None,
                 seed: int=42,
                 stats: dict=None,
                 demos_point_norm: list=None):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.pred_horizon=pred_horizon

        # --- ensure task is a list ---
        if isinstance(task, str):
            task_list = [task]
        elif isinstance(task, (list, tuple)):
            task_list = list(task)
        else:
            raise ValueError(f"Invalid task argument: expected str or list[str], got {type(task)}")

        # split based on demos
        if split_config is not None:
            num_train_demos = split_config["train_demos"]
            num_test_demos  = split_config["test_demos"]
            assert num_train_demos + num_test_demos <= 7, "Expected max 7 demos per task"
        else:
            num_train_demos = 6
            num_test_demos = 1
        
        # ---- load only demos of the specified task(s) ----
        all_pos, all_vel, episode_ends, demo_bounds, offset = [], [], [], [], 0
        for t_idx, t in enumerate(task_list):
            task_mat_path = os.path.join(dataset_dir, t + ".mat")
            if not os.path.isfile(task_mat_path):
                raise FileNotFoundError(f"Task file not found: {task_mat_path}")

            print(f"Loading demos from: {task_mat_path}")
            data = sio.loadmat(task_mat_path, squeeze_me=True, struct_as_record=False)
            demos = np.ravel(data["demos"])
            
            # Split demos into train/test subsets
            if split in ["train"]:
                demos_to_use = demos[:num_train_demos]
            elif split in ["test"]:
                demos_to_use  = demos[-num_test_demos:]
            else:  # "all"
                demos_to_use = demos

            for demo in demos_to_use:
                pos = demo.pos.T.astype(np.float32)
                vel = demo.vel.T.astype(np.float32)
                self.T = pos.shape[0]
                all_pos.append(pos)
                all_vel.append(vel)
                s, e = offset, offset + self.T
                offset = e
                episode_ends.append(e)
                demo_bounds.append((s, e, t_idx))
            
        all_pos = np.concatenate(all_pos, axis=0)
        all_vel = np.concatenate(all_vel, axis=0)
        # obs = np.concatenate([all_pos, all_vel], axis=-1) if use_vel_in_obs else all_pos
        actions = all_pos
        self.dataset = {"pos": all_pos, "vel": all_vel, "action": actions}
        self.episode_ends = np.array(episode_ends, dtype=np.int64)
        self.demo_bounds = demo_bounds

        # --- generate training indices ---
        if split == "train":
            all_indices = create_sample_indices(
                episode_ends=self.episode_ends,
                sequence_length=pred_horizon,
                pad_before=obs_horizon-1,
                pad_after=action_horizon-1
            )
            self.indices = all_indices
        elif split == "test":
            # full demos directly used → one entry per demo
            all_indices = create_sample_indices(
                episode_ends=self.episode_ends,
                sequence_length=self.T,
                pad_before=0,
                pad_after=0
            )
            self.indices = all_indices
        else:
            self.indices = None  # not used for "all"
            
        # --- stats + normalization ---
        if stats is None:
            if split == "train":
                # compute stats ONLY from train data
                train_mask = np.zeros(len(self.dataset["pos"]), dtype=bool)
                for (buf_s, buf_e, _, _) in self.indices:
                    train_mask[buf_s:buf_e] = True
                # self.stats = {k: get_data_stats(v[train_mask]) for k,v in self.dataset.items()}
                self.stats = {}
                for k,v in self.dataset.items():
                    if k in ['pos','action']:
                        self.stats[k] = get_data_stats(v[train_mask])
                xmin,ymin = self.stats['pos']['min']
                xmax,ymax = self.stats['pos']['max']
                vel_scale = 2.0 / np.array([xmax - xmin, ymax - ymin])
                self.stats['vel'] = vel_scale
            else:
                raise ValueError("Stats must be provided for non-train splits to avoid leakage.")
        else:
            self.stats = stats

        # normalization on the dataset (must do normalization!)
        # self.normalized_data = {k: normalize_data(v, self.stats[k]) for k, v in self.dataset.items()}
        self.normalized_data = {}
        for k,v in self.dataset.items():
            if k in ['pos','action']:  
                self.normalized_data[k] = normalize_data(v, self.stats[k])
            elif k == 'vel':
                self.normalized_data[k] = v * self.stats[k]  # keep geometry consistent
                # vel_norm = vel * vel_scale   
            else:
                raise ValueError(f"Unexpected key in dataset: {k}")

        # --- generate demos_point_norm ---
        if demos_point_norm is None:
            if split == "train":
                episode_start = 0
                self.demos_point_norm = []
                for episode_end in self.episode_ends:
                    demo_point_norm = self.normalized_data['pos'][episode_start:episode_end]
                    self.demos_point_norm.append(demo_point_norm)
                    episode_start = episode_end
                ## self.demos_point_norm is a list of #episodes arrays, each array contains training 
            else:
                raise ValueError("demos_point_norm must be provided for non-train splits to get LASA maze region")
        else:
            self.demos_point_norm = demos_point_norm
            
        ## constructing obs 
        # sample["obs"] = sample["obs"][:self.obs_horizon, :]
        # obs_each = np.concatenate([self.normalized_data["pos"], self.normalized_data["vel"]], axis=-1)
        
        # obs_all = []
        # dt = 1/(self.pred_horizon-1)
        # start_i = 0
        # print(self.episode_ends) # [1000 2000 3000 4000 5000 6000]
        # for end_i in self.episode_ends:
        #     # print('check start-end', start_i, end_i)
        #     pos_seg = self.normalized_data["pos"][start_i:end_i] # (1000,2) 
        #     vel_seg = (pos_seg[1:, :] - pos_seg[:-1, :]) / dt # (999, 2)
        #     # print('check--', pos_seg.shape, vel_seg.shape)
        #     # Repeat the last velocity once to match position length
        #     vel_seg = np.concatenate([vel_seg, vel_seg[-1:, :]], axis=0)  # (1000, 2)
        #     obs_eachE = np.concatenate([pos_seg, vel_seg], axis=-1) # (1000,4)
        #     # print('check---', vel_seg.shape, obs_eachE.shape) # (1000, 2) (1000, 4)
        #     obs_all.append(obs_eachE)
        #     if not np.all(obs_eachE[:-1, :2] == pos_seg[:-1,:]):   
        #         print(error)
        #     start_i = end_i
        # obs_all = np.concatenate(obs_all, axis=0) # (21000, 4)
        # # print('check----', obs_all.shape) # (6000, 4)
        # self.normalized_data.update({"obs": obs_all}) # (6000, 4)
        
        if CHECK_VISU:
            visual_episode = self.episode_ends[0]
            fig, axes = plt.subplots(1, 1, figsize=(3,3), sharex=False)
            # axes.scatter(naction[:,0], naction[:,1], c='r', marker='o',s=1, label='LASA traj')
            axes.plot(self.normalized_data['pos'][:visual_episode,0], self.normalized_data['pos'][:visual_episode,1], 'k--', alpha=0.5, label='LASA traj')
            axes.plot(self.normalized_data['action'][:visual_episode,0], self.normalized_data['action'][:visual_episode,1], 'k--', alpha=0.5, label='LASA traj')
            axes.quiver(self.normalized_data['pos'][:visual_episode,0], self.normalized_data['pos'][:visual_episode,1], self.normalized_data['obs'][:visual_episode,2], self.normalized_data['obs'][:visual_episode,3],
                    color='red', alpha=0.5, label='pos - obsV', scale=1)
            axes.quiver(self.normalized_data['pos'][:visual_episode,0], self.normalized_data['pos'][:visual_episode,1], self.normalized_data['vel'][:visual_episode,0], self.normalized_data['vel'][:visual_episode,1],
                    color='blue', alpha=0.4, label='pos - velNorm', scale=1)
            axes.set_title("Velocity comparison");axes.set_xlabel("Normalized time (t ∈ [0,1])");axes.legend()
            plt.tight_layout()
            plt.savefig("./obsv_velNorm.png", dpi=150)
            print(f"Saved LASA dataset visualization to: ./obsv_velNorm.png")

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.obs_dim = obs_dim
        self.split = split
    
    def __getitem__(self, idx):
        '''
        self.normalized_data:
            pos -- [6000,2]->[16,2] --> 5 episodes's full-traj pos in (-1,1)
            vel -- [6000,2]->[16,2] --> 5 episodes's full-traj vel scaled based on pos -- aligned with pos
            action -- [6000,2]->[16,2] --> 5 episodes's full-traj next-pos -- pos[1:] == action[:-1]
            
            obs -- [6000,2]->[16,2] --> --> 5 episodes's full-traj reconstructed pos-vel -- obs[:-1] aligned with pos[:-1]
                each obs-vel indicates its corresponding obs-pos's next action --> i.e., the correcponding action
                (e.g., obs-pos[i] taking obs-vel[i] can get to action[i])

            after TransformTraining:
            x -- [1,2]
            v -- [1,2]
            t -- [1,]
        '''
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        seq_len = self.pred_horizon if self.split == 'train' else self.T
        sample = sample_sequence(
            train_data=self.normalized_data,
            sequence_length=seq_len,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        if self.obs_dim==4:
            trajTime = np.linspace(0, 1, self.pred_horizon)
            trajPos = sample['pos'][:self.pred_horizon,:2]  # (16,2)
            traj_segment: PiecewisePolynomial = PiecewisePolynomial.FirstOrderHold(
                trajTime, trajPos.T)
            vs = [traj_segment.EvalDerivative(t).T for t in trajTime]  # list of (1, ACTION_DIM)
            vs = np.concatenate(vs, axis=0)  # (PRED_HORIZON, ACTION_DIM) # (16,2)
            xs = [traj_segment.value(t).T for t in trajTime]  # list of (1, ACTION_DIM)
            xs = np.concatenate(xs, axis=0)  # (PRED_HORIZON, ACTION_DIM) # (16,2)

            obs_each = np.concatenate([xs, vs], axis=-1)
            sample.update({"obs":obs_each})
            sample["obs"] = sample["obs"][:self.obs_horizon]
        if self.obs_dim==2:
            trajTime = np.linspace(0, 1, self.pred_horizon)
            trajPos = sample['pos'][:self.pred_horizon,:2]  # (16,2)
            traj_segment: PiecewisePolynomial = PiecewisePolynomial.FirstOrderHold(
                trajTime, trajPos.T)
            vs = [traj_segment.EvalDerivative(t).T for t in trajTime]  # list of (1, ACTION_DIM)
            vs = np.concatenate(vs, axis=0)  # (PRED_HORIZON, ACTION_DIM) # (16,2)
            xs = [traj_segment.value(t).T for t in trajTime]  # list of (1, ACTION_DIM)
            xs = np.concatenate(xs, axis=0)  # (PRED_HORIZON, ACTION_DIM) # (16,2)
            obs_each = xs
            sample.update({"obs":obs_each})
            sample["obs"] = sample["obs"][:self.obs_horizon]
            
        return sample

    def __len__(self):
        if self.split == "train":
            return len(self.indices)
        elif self.split == "test":
            return len(self.indices)
        else:
            return 0

class MatSequenceDatasetWithNextObsAsAction_Task(MatSequenceDatasetTask):
    """
    A .mat dataset where actions = next obs (like PushTStateDatasetWithNextObsAsAction).

    Inherits everything from MatSequenceDataset but overrides how 'action'
    is constructed: shifted positions instead of raw positions.
    """

    def __init__(self, *args, transform_datum_fn: Optional[Callable] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform_datum_fn = transform_datum_fn
        # Recompute actions using "next pos"

        # --- Train split: recompute actions as "next pos" ---
        actions = []
        start_idx = 0
        for end_idx in self.episode_ends:
            pos_ep = self.dataset["pos"][start_idx:end_idx, :2]  # (T, 2)
            actions_ep = np.concatenate(
                [pos_ep[1:], pos_ep[[-1]]], axis=0
            )  # shift forward, duplicate last
            actions.append(actions_ep)
            start_idx = end_idx
        actions = np.concatenate(actions, axis=0)

        # Replace in dataset
        self.dataset["action"] = actions.astype(np.float32)
        self.normalized_data["action"] = normalize_data(
            self.dataset["action"], self.stats["action"]
        )

        
    def __getitem__(self, *args, **kwargs):
        datum = super().__getitem__(*args, **kwargs)  # get {'obs','action'}
        if self.transform_datum_fn is not None:
            datum = self.transform_datum_fn(datum)  # transform into {'obs','x','v','t',...}
        return datum

def TransformTrainingDatum(datum: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    # pos, vel, action, obs = datum['pos'], datum['vel'], datum['action'] # (2, 5); (16, 2) 
    obs, action = datum['obs'], datum['action'] # (2, 4); (16, 2) 
    OBS_HORIZON, OBS_DIM = obs.shape # (2, 4)
    PRED_HORIZON, ACTION_DIM = action.shape 
    # assert PRED_HORIZON == self.pred_horizon.item()
    assert OBS_HORIZON == 2  # logic currently only works for history of length 2
    assert PRED_HORIZON == 16

    # TODO (Sid): Set the first action correctly when creating the dataset.
    # print('in Transformtraning action[0] == obs[-1, :2]? --> ', action[0] == obs[-1, :2])
    if not np.all(action[0] == obs[-1, :2]):
        print('11111111111111111111111111')
        action = action.copy()
        action[0] = obs[-1, :2]

    # Create a trajectory from the action sequence.
    traj_times = np.linspace(0, 1, PRED_HORIZON)  # (PRED_HORIZON,) # (16,)
    traj_positions = action  # (PRED_HORIZON, ACTION_DIM) # (16,2)
    traj: PiecewisePolynomial = PiecewisePolynomial.FirstOrderHold(
        traj_times, traj_positions.T,
    )

    time = np.float32(np.random.rand())  # (,)  in [0,1) -- as np.random.rand() → a uniform random number ≥ 0 and < 1 (float64)
    x = traj.value(time).T  # (1, ACTION_DIM)
    v = traj.EvalDerivative(time).T  # (1, ACTION_DIM)

    # Add noise to position
    sigma=0.1  # fixed noise level
    x = x + sigma * np.random.randn(*x.shape)  # (1, ACTION_DIM)
    x = x.astype(np.float32)  # (1, ACTION_DIM)

    return {
        'obs': obs,  # (OBS_HORIZON, OBS_DIM) # (2,4)
        'action': action, 
        'x': x.astype(np.float32),  # (1, ACTION_DIM) # (1,2)
        'v': v.astype(np.float32),  # (1, ACTION_DIM) # (1,2)
        't': time,  # (,)
    }
    
if __name__ == "__main__":
    # from torch.utils.data import DataLoader
    # single-task
    ds_single = MatSequenceDatasetWithNextObsAsAction_Task(
        dataset_dir="/home/droplab/Monica/robotics_policy/sfp_monica/external/lasa/DataSet",
        pred_horizon=16,
        obs_horizon=2,
        obs_dim=2,
        action_horizon=8,
        task="Sshape",
        split="train",
        split_config={"train_demos": 6, "test_demos": 1},
        transform_datum_fn=TransformTrainingDatum
    )

    ds_single_ts = MatSequenceDatasetWithNextObsAsAction_Task(
        dataset_dir="/home/droplab/Monica/robotics_policy/sfp_monica/external/lasa/DataSet",
        pred_horizon=16,
        obs_horizon=2,
        obs_dim=2,
        action_horizon=8,
        task="Sshape",
        split="test",
        split_config={"train_demos": 6, "test_demos": 1},
        stats=ds_single.stats,
    )

    train_loader = torch.utils.data.DataLoader(
        ds_single,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        # persistent_workers=True
    )
    print('single-train-set length: ', len(train_loader))  
    print('single-test-set length: ', len(ds_single_ts))
    

    demo = ds_single_ts[0]
    print('single-task test demo:', demo['action'].shape, demo['pos'].shape, demo['vel'].shape)

    full_traj,full_action,full_vel,full_v,full_x = [],[],[],[],[]
    for bat_id, nbatch in enumerate(train_loader): 
        print(f'check {bat_id} nbatch-dict --> ', nbatch.keys()) # dict_keys(['obs', 'action', 'x', 'v', 't'])
        print(f'check {bat_id} nbatch-dict --> ', nbatch['obs'].shape, nbatch['x'].shape, nbatch['v'].shape, nbatch['t'].shape ) 
        assert nbatch['obs'].shape[0] == 1  # batch size 1
        
        nobs = nbatch['obs'][0] # [2, 4]
        pos = nobs[:,:2].detach().cpu().numpy()
        vel = nobs[1][-2:].detach().cpu().numpy() # get latest pos's (bs,2)
        action = nbatch['action'][0].detach().cpu().numpy() # [16,2]
        nx = nbatch['x'][0].detach().cpu().numpy()
        nv = nbatch['v'][0].detach().cpu().numpy()

        # print('check training datum pos, vel, action, x, v:', pos.shape, vel.shape, action.shape, nx.shape, nv.shape)

        full_traj.append([pos])
        full_action.append([action])
        full_vel.append([vel])
        full_v.append(nv)  
        full_x.append(nx)  
    
    full_traj = np.concatenate(full_traj, axis=0)
    full_action = np.concatenate(full_action, axis=0)
    full_vel = np.concatenate(full_vel, axis=0)
    full_v = np.concatenate(full_v, axis=0)
    full_x = np.concatenate(full_x, axis=0)
    print('check full_traj, full_action, full_vel, full_x, full_v:', full_traj.shape, full_action.shape, full_vel.shape, full_x.shape, full_v.shape)

    # # === Plotting ===
    # fig, axes = plt.subplots(2, 1, figsize=(3,6), sharex=False)
    
    # plot_trajectory(axes[0], full_traj[:, :2], color='red', label='trajectory')
    # plot_velocity(axes[0], full_traj[:, :2], full_vel, scale=0.1, color='blue', label='velocity (from obs)')

    # plot_trajectory(axes[1], full_traj[:, :2], color='red', label='trajectory')
    # plot_trajectory(axes[1], full_action[:, :2], color='red', label='trajectory')
