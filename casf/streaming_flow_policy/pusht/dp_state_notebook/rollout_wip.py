'''
During rollout: shape the proposed motion using a state-dependent inverse metric 
𝑆(𝑥)=(𝐼+𝑤 𝑛𝑛⊤)−1, where:𝑥 = current position of the agent, 𝑛 = unit normal pointing away from obstacle (e.g., wall),
𝑤 = a weight that increases as you approach the obstacle.

'''

from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch import Tensor
import collections
from tqdm.auto import tqdm


from streaming_flow_policy.pusht.dp_state_notebook.all import (
    normalize_data, unnormalize_data, Policy, PushTEnv
)

# --------------------------------------------
# Inverse metric post-processing for PushT walls
# --------------------------------------------
def sdf_box_np(x: np.ndarray,
               x_min: float = 5.0, y_min: float = 5.0,
               x_max: float = 506.0, y_max: float = 506.0):
    """Signed distance & normal to nearest wall (interior of box)."""
    x0, y0 = float(x[0]), float(x[1])
    dl, dr = x0 - x_min, x_max - x0
    db, dt = y0 - y_min, y_max - y0
    dists = np.array([dl, dr, db, dt], dtype=np.float32)
    idx = int(np.argmin(dists))  # closest wall
    phi = float(dists[idx])
    if idx == 0:   n = np.array([+1.0,  0.0], dtype=np.float32)  # from left wall
    elif idx == 1: n = np.array([-1.0,  0.0], dtype=np.float32)  # from right wall
    elif idx == 2: n = np.array([ 0.0, +1.0], dtype=np.float32)  # from bottom
    else:          n = np.array([ 0.0, -1.0], dtype=np.float32)  # from top
    return phi, n

def metric_weight(phi: float, alpha: float = 5.0, p: float = 2.0,
                  eps: float = 1e-2, w_max: float = 1e3):
    w_raw = alpha / max(phi + eps, eps)**p
    return min(w_raw, w_max)

def shape_step_inverse_metric_box(x_cur: np.ndarray, a_prop: np.ndarray,
                                   box_bounds=(5.0, 5.0, 506.0, 506.0),
                                   alpha=5.0, p=2.0, eps=1e-2, w_max=1e3,
                                   inward_only=True) -> np.ndarray:
    """Apply S(x) to delta = a_prop - x_cur, returning x_cur + shaped_delta."""
    
    delta = a_prop - x_cur
    if np.allclose(delta, 0.0):
        return a_prop.copy()

    phi, n = sdf_box_np(x_cur, *box_bounds)
    w = metric_weight(phi, alpha=alpha, p=p, eps=eps, w_max=w_max)
    ndot = float(n @ delta)

    if inward_only and ndot >= 0.0:
        return a_prop.copy()  # moving away from wall → allow

    coef = w / (1.0 + w)
    delta_shaped = delta - coef * ndot * n
    return x_cur + delta_shaped


def Rollout_WIP(
        env: PushTEnv,
        policy: Policy,
        stats: Dict,
        max_steps: int = 200,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        device: str = 'cuda',
        policy_kwargs: Optional[Dict] = None,

        # M(x) constraint settings
        use_metric_post: bool = True,
        box_bounds: Tuple[float, float, float, float] = (5.0, 5.0, 506.0, 506.0),
        ## Bounds of the environment walls: (x_min, y_min, x_max, y_max)
        alpha: float = 5.0,
        ## alpha-Strength of penalty near the wall -- TUNE! TODO!
        p: float = 2.0,
        ## p-How fast the penalty grows as you approach the wall (decay shape)
        ## Fix to 2 (inverse-square), or 1
        eps: float = 1e-2,
        ## Stability offset to prevent division by zero (smooth near wall)
        ## Fix to 1e-2 (units of pixels)
        w_max: float = 1e3,
        ## Maximum penalty weight (clipping upper bound)
        ## Fix to 1e3 or 1e2
        inward_only: bool = True,
        
    ) -> Tuple[float, List[np.ndarray]]:

    # get first observation
    obs, info = env.reset()

    # obs_deque will always keep a queue of last 2 steps of observations
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    # (obs_horizon, obs_dim) == (2, 5)
    ## as input context to the policy
    
    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    policy_kwargs = policy_kwargs or {}
    policy_kwargs["num_actions"] = 1 + action_horizon
    
    with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
        ## max_steps is the maximum number of environment steps allowed during the rollout.
        
        while not done:
            B = 1
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(obs_deque)
            # normalize observation -- which into [0,1] and then [-1,1]
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                naction: Tensor = policy(nobs, **policy_kwargs)

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])
            # unnormalize --> Scale from [-1, 1] to [0, 1], then Recover original range

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            ## (9, 2)[1:9,:] --> (8,2)
            # (action_horizon, action_dim)
            ## action_horizon: the number of future actions you extract from the predicted trajectory at each policy call.
            
            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                
                #TODO
                x_cur = np.asarray(obs_deque[-1][:2], dtype=np.float64)  # x_cur: current agent position (2,)
                ## (2, 5)[-1] --> (5,)[:2] --> (2,)
                a_prop = np.asarray(action[i][:2], dtype=np.float64) # a_prop: proposed next position from the policy (2,)
                ## (8,2)[i] --> (2)[:2] --> (2,)
                ## NOTE-with inverse of the M(x) --> a_prop==x_cur+S(x)(a_prop-x_cur)
                ## the inverse of M(x) will penalize moving into obstacles (like walls)
                ### Replace the original action with this shaped version - Send the shaped action to env.step(...)

                
                if use_metric_post:
                    # NOTE
                    a_shaped = shape_step_inverse_metric_box(
                        x_cur=x_cur,
                        a_prop=a_prop,
                        box_bounds=box_bounds,
                        alpha=alpha, p=p, eps=eps, w_max=w_max,
                        inward_only=inward_only
                    )
                else:
                    a_shaped = a_prop
                    
                action_step = action[i].copy()
                action_step[:2] = a_shaped
                
                #TODO
                
                # stepping env
                obs, reward, done, _, info = env.step(action_step) # the modified action[i]
                ## Each call to env.step(...) moves the simulation forward one frame.
                
                # save observations
                obs_deque.append(obs)
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
    return score, imgs
