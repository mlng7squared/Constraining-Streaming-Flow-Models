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

    
    #########-----------------------------------#########
    if 'shapingConfig' in policy_kwargs:
        if 'center_scale' in policy_kwargs['shapingConfig']:
            ## defining the real-obstacle position
            xmin_norm, ymin_norm = -1,-1
            xmax_norm, ymax_norm = 1,1
            max_span_norm = max((xmax_norm-xmin_norm), (ymax_norm-ymin_norm)) # 2
            
            # defining normalized-obstacle position
            workspace_norm_size   = np.array([max_span_norm, max_span_norm]) # [2,2]
            center_scale=np.array(policy_kwargs['shapingConfig']['center_scale'], dtype=np.float32) # [0.25,0.25]
            center_norm = np.array([
                xmin_norm + center_scale[0] * workspace_norm_size[0],   # 25% from the left edge
                ymax_norm - center_scale[1] * workspace_norm_size[1],   # 25% down from the top
            ])
            radius_norm = policy_kwargs['shapingConfig']['radius_scale'] * np.mean(workspace_norm_size)  # radius_scale=0.15 defines radius proportional to workspace size
            policy_kwargs['shapingConfig']['center_norm'] = center_norm
            policy_kwargs['shapingConfig']['radius_norm'] = radius_norm
    #########-----------------------------------#########
    
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
                # naction: Tensor = policy(nobs, **policy_kwargs)
                naction = policy(nobs, **policy_kwargs)
                ## (1, NUM_ACTIONS, ACTION_DIM) = (1,9,2)
                ## num_actions = 9 (1 + action_horizon)
                ## action_dim = 2 (e.g., x, y target position for agent)

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            # nvel = nvel.detach().cpu().numpy()[0]         # (num_actions, 2)
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
    return score, imgs, step_idx

