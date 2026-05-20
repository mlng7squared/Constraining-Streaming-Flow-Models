from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch import Tensor
import collections
from tqdm.auto import tqdm


from streaming_flow_policy.pusht.dp_state_notebook.all import (
    normalize_data, unnormalize_data, Policy, PushTEnv
)

def Rollout(
        env: PushTEnv,
        policy: Policy,
        stats: Dict,
        max_steps: int = 200,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        device: str = 'cuda',
        policy_kwargs: Optional[Dict] = None,
    ) -> Tuple[float, List[np.ndarray]]:

    # get first observation
    obs, info = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    # (obs_horizon, obs_dim) == (2, 5)
    ## as input context to the policy

    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    policy_kwargs = policy_kwargs or {}
    ## integration_steps_per_action = 6
    policy_kwargs["num_actions"] = 1 + action_horizon # 9 = 8 + 1
    ## Model will predict 1 + action_horizon future positions or velocities

    with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon (2) number of observations
            ## shape: (obs_horizon, obs_dim)
            obs_seq = np.stack(obs_deque)
            # normalize observation
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                naction: Tensor = policy(nobs, **policy_kwargs)
                ## (1, NUM_ACTIONS, ACTION_DIM) = (1,9,2)
                ## num_actions = 9 (1 + action_horizon)
                ## action_dim = 2 (e.g., x, y target position for agent)

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            # (pred_horizon, action_dim) = (9, 2)
            action_pred = unnormalize_data(naction, stats=stats['action'])
            # (9, 2)

            # only take action_horizon number of actions
            start = obs_horizon - 1 # 2-1=1
            end = start + action_horizon # 1+8=9
            action = action_pred[start:end,:]
            ## (9, 2)[1:9,:] --> (8,2)
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            ## For each predicted step:
                # Apply action (e.g., target position)
                # Observe new state and reward
                # Append to history and rendering list
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i]) # action[i]: shape (2,)
                ## Reward at each step = fraction of goal covered (scaled to [0, 1])
                ## Score = best reward ever achieved in that episode
                
                # save observations
                obs_deque.append(obs) # adds new obs to context
                # and reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    score = max(rewards)
    ## score: maximum reward during rollout (e.g., best block-goal overlap)
    ## The final score is the maximum reward achieved during the rollout
    ## "What was the best target coverage achieved during the episode?"
    return score, imgs



####### TODO-2

# ---------- inverse-metric post-shaping (NumPy) ----------
def _sdf_circle_np(x: np.ndarray, center: np.ndarray, radius: float, eps: float=1e-6):
    """
    x: (2,), center: (2,)
    returns phi (scalar), n (2,) outward unit normal
    """
    rel = x - center
    dist = np.sqrt(np.clip((rel**2).sum(), eps, None))
    phi  = dist - radius
    n    = rel / dist
    return phi, n

def _metric_weight_np(phi: float, alpha: float=5.0, p: float=2.0, eps: float=1e-2, w_max: float=1e3):
    # Optional: only use distance outside obstacle
    phi_eff = max(phi, 0.0)
    w_raw   = alpha / max(phi_eff + eps, eps)**p
    return min(w_raw, w_max)

def _shape_step_inverse_metric(
    x_cur: np.ndarray,           # (2,)
    a_proposed: np.ndarray,      # (2,)
    center: np.ndarray,          # (2,)
    radius: float,
    alpha: float=5.0, p: float=2.0, eps: float=1e-2, w_max: float=1e3,
    inward_only: bool=True
) -> np.ndarray:
    """
    Returns a shaped next position: x_cur + S * (a_proposed - x_cur),
    where S = (I + w n n^T)^(-1) = I - (w/(1+w)) n n^T.
    If inward_only=True, apply S only when the step moves inward (n·delta < 0).
    """
    delta = a_proposed - x_cur  # proposed displacement
    if np.allclose(delta, 0.0):
        return a_proposed.copy()

    phi, n = _sdf_circle_np(x_cur, center, radius)
    w = _metric_weight_np(phi, alpha=alpha, p=p, eps=eps, w_max=w_max)
    if w <= 0.0:
        return a_proposed.copy()

    # inward check: moving along -n decreases distance (n·delta < 0)
    ndot = float(n @ delta)
    if inward_only and ndot >= 0.0:
        return a_proposed.copy()

    coef = w / (1.0 + w)          # scalar in (0,1)
    # S * delta = delta - coef * (n n^T) delta = delta - coef * (n·delta) n
    delta_shaped = delta - coef * ndot * n
    return x_cur + delta_shaped


# ---------- modified rollout with post-multiplier ----------
def Rollout_post(
        env,
        policy,
        stats: Dict,
        max_steps: int = 200,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        device: str = 'cuda',
        policy_kwargs: Optional[Dict] = None,
        # --- metric post-processing controls ---
        use_metric_post: bool = True,
        obstacle_center: Tuple[float, float] = (0.0, -2.0),
        obstacle_radius: float = 1.0,
        alpha: float = 5.0,
        p: float = 2.0,
        eps: float = 1e-2,
        w_max: float = 1e3,
        inward_only: bool = True,
    ) -> Tuple[float, List[np.ndarray]]:

    # get first observation
    obs, info = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = []
    done = False
    step_idx = 0

    policy_kwargs = dict(policy_kwargs or {})
    policy_kwargs["num_actions"] = 1 + action_horizon

    center_np = np.array(obstacle_center, dtype=np.float32)

    with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
        while not done:
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(obs_deque)                        # (2, OBS_DIM)
            # normalize observation
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            # device transfer
            nobs_t = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            # infer action sequence (pred_horizon x action_dim) in normalized space → unnormalized
            with torch.no_grad():
                naction_t: torch.Tensor = policy(nobs_t, **policy_kwargs)  # (1, pred_horizon, action_dim)
            naction = naction_t.detach().to('cpu').numpy()[0]              # (pred_horizon, action_dim)
            action_pred = unnormalize_data(naction, stats=stats['action']) # (pred_horizon, action_dim)

            # only take action_horizon number of actions, aligned with "present"
            start = obs_horizon - 1
            end   = start + action_horizon
            action_seq = action_pred[start:end, :]                         # (H, A)

            # execute without replanning
            for i in range(len(action_seq)):
                # current position from latest observation (assumes first 2 dims are planar pos)
                x_cur = np.asarray(obs_deque[-1][:2], dtype=np.float32)    # (2,)
                a_prop = np.asarray(action_seq[i][:2], dtype=np.float32)   # (2,)

                if use_metric_post:
                    a_shaped_xy = _shape_step_inverse_metric(
                        x_cur=x_cur,
                        a_proposed=a_prop,
                        center=center_np,
                        radius=obstacle_radius,
                        alpha=alpha, p=p, eps=eps, w_max=w_max,
                        inward_only=inward_only
                    )
                else:
                    a_shaped_xy = a_prop

                # write back into the full action vector (if more than 2 dims, pass-through others)
                a_to_step = action_seq[i].copy()
                a_to_step[:2] = a_shaped_xy

                # env step
                obs, reward, done, _, info = env.step(a_to_step)

                # save obs/reward/vis
                obs_deque.append(obs)
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                # progress
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=float(reward))
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    score = max(rewards) if rewards else 0.0
    return score, imgs