from typing import Dict, Optional
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import matplotlib.pyplot as plt

from pydrake.all import PiecewisePolynomial
from torchdyn.core import NeuralODE
from torchdiffeq import odeint

from streaming_flow_policy.pusht.dp_state_notebook.base_policy import Policy


class StreamingFlowPolicyDeterministic (Policy):
    def __init__(self,
                 velocity_net: nn.Module,
                 action_dim: int,
                 pred_horizon: int = 16,
                 obs_horizon:int = 2,
                 sigma: float = 0.0,
                 device: torch.device = 'cuda',
        ):
        """
        Args:
            velocity_net (nn.Module): velocity network
            action_dim (int): action dimension
            pred_horizon (int): prediction horizon
            sigma (float): standard deviation of the Gaussian noise
            device (torch.device): device
        """
        super().__init__()
        self.velocity_net = velocity_net
        self.action_dim = action_dim
        self.device = device

        # Register pred_horizon and sigma as buffers if provided
        self.register_buffer('pred_horizon', torch.tensor(pred_horizon, dtype=torch.int32))
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))
        self.pred_horizon: Tensor; self.sigma: Tensor

    def TransformTrainingDatum(self, datum: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Args:
            datum (Dict[str, np.ndarray]):
                'obs' (np.ndarray, shape=(OBS_HORIZON, OBS_DIM), dtype=np.float32)
                'action' (np.ndarray, shape=(PRED_HORIZON, ACTION_DIM), dtype=np.float32)

        Returns:
            Dict[str, np.ndarray]:
                'obs' (np.ndarray, shape=(OBS_HORIZON, OBS_DIM), dtype=np.float32)
                'x' (np.ndarray, shape=(ACTION_DIM,), dtype=np.float32): position
                'v' (np.ndarray, shape=(ACTION_DIM,), dtype=np.float32): velocity
                't' (np.ndarray, shape=(,), dtype=np.float32): time
        """
        obs, action = datum['obs'], datum['action'] # (2, 5); (16, 2) 
        OBS_HORIZON, OBS_DIM = obs.shape # (2, 5)
        PRED_HORIZON, ACTION_DIM = action.shape 
        assert PRED_HORIZON == self.pred_horizon.item()
        assert OBS_HORIZON == 2  # logic currently only works for history of length 2

        # Ensure that the first action matches the last observation
        # This may not happen when the sequence starts from the beginning of an
        # episode and both obs and action are duplicated. Then, hack this by
        # setting the first action to the last observation. However, the right
        # thing to do is:
        # TODO (Sid): Set the first action correctly when creating the dataset.
        # print('a'*10)
        # print(action[0] == obs[0,:2]) #FALSE
        # print('b'*10)
        # print(action[0] == obs[1,:2]) #TRUE
        # print('c'*10)
        if not np.all(action[0] == obs[-1, :2]):
            # print('11111111111111111111111111')
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
        x = x + self.sigma.item() * np.random.randn(*x.shape)  # (1, ACTION_DIM)
        x = x.astype(np.float32)  # (1, ACTION_DIM)

        ## x and v are a single position–velocity pair sampled from the continuous demo trajectory built from the 16 action waypoints.
        dt = 1.0/(PRED_HORIZON-1)
        t_next = min(time + dt, 1.0) # clamp
        x_next = traj.value(t_next).T.astype(np.float32)
        return {
            'obs': obs,  # (OBS_HORIZON, OBS_DIM) # (2,5)
            'action': action, 
            'x': x.astype(np.float32),  # (1, ACTION_DIM) # (1,2)
            'v': v.astype(np.float32),  # (1, ACTION_DIM) # (1,2)
            't': time,  # (,)
            'dt': np.float32(dt),
            'x_next': x_next,
        }

    
    @torch.enable_grad()
    def Loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            batch (Dict[str, Tensor]):
                'obs' (Tensor, shape=(B, OBS_HORIZON, OBS_DIM))
                'x' (Tensor, shape=(B, 1, ACTION_DIM))
                'v' (Tensor, shape=(B, 1, ACTION_DIM))
                't' (Tensor, shape=(B,))

        Returns:
            Tensor (shape=(,), dtype=torch.float32): loss
        """
        # device transfer
        obs = batch['obs'].to(self.device)  # (B, OBS_HORIZON, OBS_DIM)  # torch.Size([1024, 2, 4])
        x = batch['x'].to(self.device)  # (B, 1, ACTION_DIM)
        v = batch['v'].to(self.device)  # (B, 1, ACTION_DIM)
        t = batch['t'].to(self.device)  # (B,)
        B = obs.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs.flatten(start_dim=1).to(x.dtype)  # (B, OBS_HORIZON * OBS_DIM) # torch.Size([1024, 8])
        
        # predict the velocity
        v_pred = self.velocity_net(
            sample=x, timestep=t, global_cond=obs_cond
        )  # (B, 1, ACTION_DIM)
        
        # dt = batch['dt'].to(self.device)  # (B,)
        # x_next_gt = batch['x_next'].to(self.device)  # (B,1,2)
        # x_next_pred = x + dt.view(-1,1,1) * v_pred # Eucler
        # # x_mid = x + 0.5 * dt.view(-1,1,1) * v_pred
        # # t_mid = t + 0.5 * dt
        # # k2 = self.velocity_net(sample=x_mid, timestep=t_mid, global_cond=obs_cond)
        # # x_next_pred = x + dt.view(-1,1,1) * k2
        # loss_pos = nn.functional.mse_loss(x_next_pred, x_next_gt)
        
        # L2 loss
        loss_mse = nn.functional.mse_loss(v_pred, v)  # (,)
        loss_cos = 1 - torch.nn.functional.cosine_similarity(v_pred, v, dim=-1).mean()

        loss = loss_mse
        # loss = loss_mse + 0.1* loss_pos

        return loss

    @torch.inference_mode()
    def __call__(self,
                 nobs: Tensor,
                 num_actions: Optional[int] = None,
                 integration_steps_per_action: int = 6,
                 postShaping:str = None,
                 shapingConfig:Optional[dict]=None,
    ) -> Tensor:
        """
        Args:
            nobs (Tensor, shape=(OBS_HORIZON, OBS_DIM)): normalized observations
            num_actions (Optional[int]): number of actions to predict
            integration_steps_per_action (int): number of integration steps per action

        Returns:
            Tensor (shape=(1, NUM_ACTIONS, ACTION_DIM)): predicted actions
        """
        # print('aaaa-check nobs.shape', nobs.shape)
        # obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)  # (1, OBS_HORIZON * OBS_DIM)
        obs_cond = nobs.flatten(start_dim=1)   # (B, OBS_HORIZON*OBS_DIM) (B,2*4) ## (B,2*2)=B,4
        # print('bbbb-check obs_cond.shape', obs_cond.shape)
        x = nobs[:, -1, :2]  # (B, ACTION_DIM)
        # print('DEBUG --> ', )

        # Integration time steps.
        num_actions = num_actions or self.pred_horizon.item()
        assert 1 <= num_actions <= self.pred_horizon.item()
        num_future_actions = num_actions - 1
        t_max = num_future_actions / (self.pred_horizon.item() - 1) # 7/15
        total_integration_steps = 1 + num_future_actions * integration_steps_per_action
        t_span = torch.linspace(0, t_max, total_integration_steps, device=x.device)

        ## odeint from torchdiffeq 
        # if postShaping == 'None' or postShaping == 'obstacle':
        #     vf = ShapedVectorFieldWrapper_CASF(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
        # elif postShaping == 'wall':
        #     vf = ShapedVectorFieldWrapperWall_CASF(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
        # elif postShaping == 'maze':
        #     vf = ShapedVectorFieldWrapperCorridor_CASF(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
        # else:  # postShaping=None
        #     vf = VectorFieldWrapper(self.velocity_net, obs_cond)
        # pred_traj = odeint(vf, x, t_span, rtol=1e-4, atol=1e-4)
        # # pred_traj = odeint(field_safe, x0, t_eval, method="dopri5", rtol=1e-5, atol=1e-6, options=dict(max_num_steps=10000))
        # pred_traj = pred_traj.permute(1, 0, 2)  # (T,2) # torch.Size([1,43,2])
        
        # --- torchdyn --- # 
        # ode_solver = NeuralODE(
        #     vector_field=VectorFieldWrapper(self.velocity_net, obs_cond),
        #     solver="euler", # "dopri5",
        #     sensitivity="adjoint",
        #     atol=1e-6,
        #     rtol=1e-6,
        # )
        # traj = ode_solver.trajectory(x=x0, t_span=t_span)  # (K, 2)
        # # print('cccc-check traj.shape', traj.shape)
        # traj = traj.permute(1, 0, 2) ## # (B, len_t/K, ACTION_DIM)
        # # print('dddd-check traj.shape', traj.shape)
        # --- torchdyn --- # 

        # --- self-written rollout [Eucler!] --- # TODO: try midpoint/dopri5
        Δt = t_span[1] - t_span[0]
        traj = [x]
        traj_shaped = [x]
        vel_traj,vel_traj_shaped = [],[]
        # --- rollout ---
        x_shaped = x
        for i in range(total_integration_steps):
            t = t_span[i]
            # query velocity net (vector field)
            if postShaping == 'obstacle':
                if len(shapingConfig['center_norm']) > 1:
                    # print('multiple obstacle... ')
                    vf = VectorFieldWrapper(self.velocity_net, obs_cond)
                    vf_shaped = ShapedVectorFieldWrapper_multi_CASF(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
                else:
                    vf = VectorFieldWrapper(self.velocity_net, obs_cond)
                    vf_shaped = ShapedVectorFieldWrapper_CASF(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
                shaping = True
            elif postShaping == 'obstacle-cbf':
                vf = VectorFieldWrapper(self.velocity_net, obs_cond)
                vf_shaped = ShapedVectorFieldWrapper_CBF(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
                shaping = True
            elif postShaping == 'obstacle-hardB':
                vf = VectorFieldWrapper(self.velocity_net, obs_cond)
                vf_shaped = ShapedVectorFieldWrapper_HARD(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
                shaping = True
            elif postShaping == 'wall':
                print('DEBUG->postShaping==wall -- NOT EXPECTED')
                vf = VectorFieldWrapper(self.velocity_net, obs_cond)
                vf_shaped = ShapedVectorFieldWrapperWall_CASF(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
                shaping = True
            elif postShaping == 'maze':
                print('DEBUG->postShaping==maze -- NOT EXPECTED')
                vf = VectorFieldWrapper(self.velocity_net, obs_cond)
                vf_shaped = ShapedVectorFieldWrapperCorridor_CASF(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
                shaping = True
            else:  # postShaping=None 
                vf = VectorFieldWrapper(self.velocity_net, obs_cond)
                vf_shaped = None
                shaping = False
            
            # EULER
            v = vf(t,x)  # (B, 2)
            x_next = x + Δt * v
            
            vel_traj.append(v)
            traj.append(x_next)
        
            if shaping:
                v_shaped = vf_shaped(t,x)
                x_next_shaped = x + Δt * v_shaped
                vel_traj_shaped.append(v_shaped)
                traj_shaped.append(x_next_shaped)
                x = x_next_shaped
            else:
                x = x_next

        # stack
        traj = torch.stack(traj, dim=1)       # (B, T+1, 2)
        traj = traj[:,:-1,:]
        vel_traj = torch.stack(vel_traj, dim=1)  # (B, T, 2)
        if shaping:
            traj_shaped = torch.stack(traj_shaped, dim=1)
            traj_shaped = traj_shaped[:,:-1,:]
            vel_traj_shaped = torch.stack(vel_traj_shaped, dim=1)  # (B, T-1, 2)
            # --- self-written rollout [Eucler!] --- # TODO: try midpoint

        select_action_indices = torch.arange(
            0,
            total_integration_steps,
            integration_steps_per_action, device=traj.device
        )
        ## naction = traj[select_action_indices]  # (NUM_ACTIONS, 2)
        ## naction = naction.unsqueeze(0)  # (1, NUM_ACTIONS, 2)
        # print('traj.shape', traj.shape) # [1, 43, 2]
        naction = traj[:, select_action_indices, :] #  [1, 43, 2]
        nvel = vel_traj[:, select_action_indices, :]
        if shaping:
            naction_shaped = traj_shaped[:, select_action_indices, :]
            nvel_shaped = vel_traj_shaped[:, select_action_indices, :]
            return naction, nvel
        return naction, nvel


class VectorFieldWrapper (nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model: nn.Module, obs_cond: Tensor):
        super().__init__()
        self.model = model
        self.obs_cond = obs_cond

    def forward(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Args:
            t (Tensor, shape=(,), dtype=float): time
            x (Tensor, shape=(ACTION_DIM,), dtype=float): position

        Returns:
            Tensor (shape=(ACTION_DIM,), dtype=float): velocity
        """
        B = x.shape[0]
        x = x.unsqueeze(1) 
        # x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, ACTION_DIM)
        t_expand = t.repeat(B)  
        v: Tensor = self.model(
            sample=x,
            timestep=t_expand,
            global_cond=self.obs_cond,
        )  # (1, 1, ACTION_DIM)
        # v = v.flatten()  # (ACTION_DIM,)
        v = v.squeeze(1)  # (B, ACTION_DIM)
        return v


### ------- postShaping: obstacle ------- ###
def shape_velocity_batch_metric_CASF(
    a: torch.Tensor,           # (B, 2) current position/action
    v: torch.Tensor,           # (B, 2) raw velocity
    center: torch.Tensor,      # (2,) obstacle center
    radius: float,             # scalar
    alpha=10.0,
    beta=1.0,
    w_scale=50.0,
    eps=1e-6,
) -> torch.Tensor:
    """
    Implements the CASF-style metric-weighted velocity shaping:
    v_tilde = M(a)^(-1) v_theta(a)
    where M(a) = I + w(d) [α n n^T + β (I - n n^T)].
    """
    dtype = a.dtype
    device = a.device

    # 🔧 Force dtype alignment for safety
    center = torch.as_tensor(center, device=device, dtype=dtype)
    radius = torch.as_tensor(radius, device=device, dtype=dtype)
    
    # Relative geometry
    rel = a - center[None, :]                    # (B,2)
    dist = torch.sqrt((rel**2).sum(dim=-1) + eps)  # (B,)
    n = rel / (dist[:, None] + eps)                      # (B,2)
    phi = dist - radius                          # (B,) signed distance
    # phi = torch.linalg.norm(a - center[None, :], dim=-1, keepdim=True) - radius ### signed distance
    # n = (a - center[None, :]) / (torch.linalg.norm((a - center[None, :]), dim=-1, keepdim=True) + 1e-8)

    # TODO: Smooth influence: Influence weight w(d): e.g. Gaussian or polynomial decay
    # w = torch.exp(-w_scale * phi.clamp_min(0.0))  # (B,)
    # shaping weight w that →1 at boundary, →0 away
    w = torch.exp(-w_scale * torch.abs(phi))   # w≈1 at φ=0, →0 inside/outside
    # w = torch.exp(-w_scale * torch.abs(phi))
    # w = 1.0 / (1.0 + torch.exp(20*(phi - 0.05)))
    # w = torch.exp(-w_scale * phi.clamp_min(0.0))  # (B,1)
    # w = 1.0 / (1.0 + (phi / 0.1)**2)  # slower falloff
    # w = (1.0 / (torch.clamp(phi, min=eps)))**w_scale ## the metric blow up at the boundary
            ## p > 1 makes blow-up sharper -- 2.0 or 3.0
    
        
    # Metric construction per point
    I = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(a.shape[0], -1, -1) # (B,2,2)
    nnT = torch.einsum('bi,bj->bij', n, n)                                     # (B,2,2)
    M = I + w[:, None, None] * (alpha * nnT + beta * (I - nnT))                # (B,2,2)
    ## alpha * nnT --> penalises motion in the direction of collision
    ## beta * (I-nnT) --> penalises tangential motion (sideways)
        ## vectors are tangent to the circle boundary 
        ## i.e. if you move in those directions, you “slide around” the obstacle instead of going in or out
    ### alpha ≫ beta → so normal direction is much more expensive than tangent
    ### w controls how strong this effect is depending on signed distance
    ### φ = 0 → on boundary → shaping max
    ### φ far > 0 or < 0 → shaping fades out
    # Invert the metric for each batch element (soft projection)
    M_inv = torch.linalg.inv(M)                                                # (B,2,2)
    ## use M⁻¹ to “pull” the velocity into safe tangent direction
    
    # ######-barrier-######
    # # Decompose velocity into normal/tangent
    # v_n = (v * n).sum(-1, keepdim=True) * n                   # (B,2)
    # v_t = v - v_n
    # # One-sided rectification (no inward flow):
    # # keep only outward normal (v·n)+, kill inward (v·n)-
    # v_n_out = torch.relu((v * n).sum(-1, keepdim=True)) * n   # (B,2)
    # # Blend: pass tangential freely, pass outward normal through metric,
    # # inward normal = 0 (cannot enter).
    # v_corr = v_t + v_n_out
    # ######-barrier-######
    
    v_shaped = torch.bmm(M_inv, v.unsqueeze(-1)).squeeze(-1)                   # (B,2)
    # v_shaped = torch.linalg.solve(M, v.unsqueeze(-1)).squeeze(-1)


    # ######-cbf-barrier residual: r = -kappa*h - ∇h·v -######
    # grad_h_dot_v = (n * v_shaped).sum(-1)                    # (B,)
    # r = -alpha*phi - grad_h_dot_v                       # (B,)
    # # Only correct when residual is positive (violation)
    # lam = torch.relu(r)                               # (B,)
    # v_shaped = v_shaped + lam[:,None] * n
    # ######-barrier residual: r = -kappa*h - ∇h·v -######
    
    # # hard barrier inside
    # if_inside = (phi <= 0)
    # v_shaped[if_inside] = (n[if_inside] * torch.norm(v[if_inside], dim=-1, keepdim=True))
    # # one-sided: only shape when inward
    # inward = ((v * n).sum(-1, keepdim=True) < 0).float()
    # v_shaped = inward * v_shaped + (1 - inward) * v
    
    v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
    v_shaped = v_shaped / (v_shaped.norm(dim=-1, keepdim=True) + 1e-6) * v_norm
    return v_shaped


def shape_velocity_batch_hardBarrier(
    a: torch.Tensor,           # (B, 2) current position/action
    v: torch.Tensor,           # (B, 2) raw velocity
    center: torch.Tensor,      # (2,) obstacle center
    radius: float,             # scalar
    alpha=1.0,
    beta=0.0,
    w_scale=0.0,
    eps=0.0,
) -> torch.Tensor:

    dtype = a.dtype
    device = a.device

    # 🔧 Force dtype alignment for safety
    center = torch.as_tensor(center, device=device, dtype=dtype)
    radius = torch.as_tensor(radius, device=device, dtype=dtype)
    
    # Relative geometry
    rel = a - center[None, :]                    # (B,2)
    dist = torch.sqrt((rel**2).sum(dim=-1) + eps)  # (B,)
    n = rel / (dist[:, None] + eps)                      # (B,2)
    phi = dist - radius                          # (B,) signed distance
    # phi = torch.linalg.norm(a - center[None, :], dim=-1, keepdim=True) - radius ### signed distance
    # n = (a - center[None, :]) / (torch.linalg.norm((a - center[None, :]), dim=-1, keepdim=True) + 1e-8)
    # --- 2. Check Velocity Condition ---
    
    # Project velocity onto the normal vector to find the inward/outward component
    # This is the dot product of v and n
    v_dot_n = (v * n).sum(dim=-1, keepdim=True)        # (B, 1)

    # --- 3. Identify Unsafe Conditions ---
    
    # Condition 1: Is the agent at or inside the boundary?
    is_at_or_inside = (phi <= 0)                      # (B,)
    
    # Condition 2: Is the velocity pointing into the obstacle?
    # (v_dot_n < 0 means v and n are pointing in opposite directions)
    is_moving_inward = (v_dot_n.squeeze() <= 0)       # (B,)
    
    # Combine conditions: We must project if (at/inside AND moving in)
    needs_projection = (is_at_or_inside & is_moving_inward) # (B,)

    # --- 4. Apply Hard Projection ---
    
    # Calculate the part of the velocity that is tangent to the boundary
    # v_tangent = v - v_normal
    # v_normal = (v_dot_n * n)
    v_tangent = v - (v_dot_n * n)                     # (B, 2)
    
    # Use torch.where to select the velocity:
    # IF needs_projection is True -> use v_tangent
    # ELSE (agent is safe)       -> use v (nominal velocity)
    v_safe = torch.where(
        needs_projection[:, None],  # (B, 1) - Unsqueeze to broadcast with (B, 2)
        v_tangent,                  # (B, 2) - Projected (safe) velocity
        v                           # (B, 2) - Original (nominal) velocity
    )

    return v_safe

    # # --- Outside obstacle: keep nominal ---
    # outside = (phi >= 0).float()[:, None]
    # # --- Near boundary or inside: strong correction ---
    # # Classic barrier: repel = K * n / φ
    # repel = alpha * n / (phi[:, None] + eps)
    # # Inside obstacle → push out even harder
    # inside = (phi < 0).float()[:, None]
    # repel_inside = alpha * 10.0 * n       # optional: stronger inside
    # # Combine
    # v_shaped = (
    #     outside * v +
    #     (1 - outside) * (v + repel * (1 - inside) + repel_inside * inside)
    # )
    # v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
    # v_shaped = v_shaped / (v_shaped.norm(dim=-1, keepdim=True) + 1e-6) * v_norm

    # return v_shaped



class ShapedVectorFieldWrapper_HARD(VectorFieldWrapper):
    """
    CASF-style metric-weighted streaming flow.
    Applies continuous, differentiable constraint shaping:
        ṽ(a) = M(a)^(-1) vθ(a)
            center, radius,
            alpha=10.0, beta=1.0, w_scale=50.0,
            eps=1e-6
    """

    def __init__(self, model: nn.Module, obs_cond: torch.Tensor, shapingConfig:Optional[dict]=None,
                 ):
        super().__init__(model, obs_cond)
        device = obs_cond.device

        self.center = torch.as_tensor(shapingConfig['center_norm'], dtype=torch.float32, device=device)
        self.radius = float(shapingConfig['radius_norm'])
        self.alpha = shapingConfig['alpha']
        self.beta = shapingConfig['beta']
        self.w_scale = shapingConfig['w_scale']
        self.eps = shapingConfig['eps']

    def forward(self, t: torch.Tensor, x_norm: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 1. Predict nominal normalized velocity
        v_norm = super().forward(t, x_norm)                # (B,2)
        # v_norm_shaped = shape_velocity_batch_metric_CASF(
        #     x_norm, v_norm, self.center, self.radius,
        #     alpha=self.alpha, beta=self.beta, w_scale=self.w_scale, eps=self.eps
        # )
        v_norm_shaped = shape_velocity_batch_hardBarrier(
            x_norm, v_norm, self.center, self.radius,
            alpha=self.alpha, beta=self.beta, w_scale=self.w_scale, eps=self.eps
        )

        return v_norm_shaped


def shape_velocity_batch_CBF(
    a: torch.Tensor,           # (B, 2) current position/action
    v: torch.Tensor,           # (B, 2) raw velocity
    center: torch.Tensor,      # (2,) obstacle center
    radius: float,             # scalar
    alpha=5.0,
    beta=0.0,
    w_scale=0.0,
    eps=1e-6,
) -> torch.Tensor:
    """Project velocities onto the CBF-safe half-space.

    This is the closed-form solution of the per-point QP

        min_v_safe 0.5 ||v_safe - v||^2
        s.t.       n(a)^T v_safe >= -alpha * h(a)

    with h(a) = ||a - center|| - radius. For a single circular obstacle in 2D
    the feasible set is one half-space, so no iterative QP solver is needed.
    """

    dtype = a.dtype
    device = a.device

    center = torch.as_tensor(center, device=device, dtype=dtype)
    radius = torch.as_tensor(radius, device=device, dtype=dtype)

    rel = a - center[None, :]
    dist = torch.sqrt((rel * rel).sum(dim=-1, keepdim=True).clamp_min(eps))
    n = rel / dist.clamp_min(eps)
    h = dist - radius

    lower = -float(alpha) * h
    lhs = (v * n).sum(dim=-1, keepdim=True)
    deficit = (lower - lhs).clamp_min(0.0)
    denom = (n * n).sum(dim=-1, keepdim=True).clamp_min(eps)
    return v + (deficit / denom) * n

class ShapedVectorFieldWrapper_CBF(VectorFieldWrapper):
    """
    CASF-style metric-weighted streaming flow.
    Applies continuous, differentiable constraint shaping:
        ṽ(a) = M(a)^(-1) vθ(a)
            center, radius,
            alpha=10.0, beta=1.0, w_scale=50.0,
            eps=1e-6
    """

    def __init__(self, model: nn.Module, obs_cond: torch.Tensor, shapingConfig:Optional[dict]=None,
                 ):
        super().__init__(model, obs_cond)
        device = obs_cond.device

        self.center = torch.as_tensor(shapingConfig['center_norm'], dtype=torch.float32, device=device)
        self.radius = float(shapingConfig['radius_norm'])
        self.alpha = shapingConfig['alpha']
        self.beta = shapingConfig['beta']
        self.w_scale = shapingConfig['w_scale']
        self.eps = shapingConfig['eps']

    def forward(self, t: torch.Tensor, x_norm: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 1. Predict nominal normalized velocity
        v_norm = super().forward(t, x_norm)                # (B,2)
        # v_norm_shaped = shape_velocity_batch_metric_CASF(
        #     x_norm, v_norm, self.center, self.radius,
        #     alpha=self.alpha, beta=self.beta, w_scale=self.w_scale, eps=self.eps
        # )
        v_norm_shaped = shape_velocity_batch_CBF(
            x_norm, v_norm, self.center, self.radius,
            alpha=self.alpha, beta=self.beta, w_scale=self.w_scale, eps=self.eps
        )

        return v_norm_shaped



### ------- postShaping: obstacle ------- ###
def shape_velocity_batch_metric_rander(
    a: torch.Tensor,           # (B, 2) current position/action
    v: torch.Tensor,           # (B, 2) raw velocity
    center: torch.Tensor,      # (2,) obstacle center
    radius: float,             # scalar
    alpha=10.0,
    beta=1.0,
    w_scale=50.0,
    eps=1e-6,
    drift_gain: float = 0.03,   # strength of Randers drift β(x)
    # blend: float = 0.1          # how much to trust (M^{-1}v + β) near wall (0..1)

) -> torch.Tensor:
    """
    Implements the CASF-style metric-weighted velocity shaping:
    v_tilde = M(a)^(-1) v_theta(a)
    where M(a) = I + w(d) [α n n^T + β (I - n n^T)].
    """
    dtype = a.dtype
    device = a.device

    # 🔧 Force dtype alignment for safety
    center = torch.as_tensor(center, device=device, dtype=dtype)
    radius = torch.as_tensor(radius, device=device, dtype=dtype)
    
    # Relative geometry
    rel = a - center[None, :]                    # (B,2)
    dist = torch.sqrt((rel**2).sum(dim=-1) + eps)  # (B,)
    n = rel / (dist[:, None] + eps)                      # (B,2)
    phi = dist - radius                          # (B,) signed distance
    # phi = torch.linalg.norm(a - center[None, :], dim=-1, keepdim=True) - radius ### signed distance
    # n = (a - center[None, :]) / (torch.linalg.norm((a - center[None, :]), dim=-1, keepdim=True) + 1e-8)

    # TODO: Smooth influence: Influence weight w(d): e.g. Gaussian or polynomial decay
    # w = torch.exp(-w_scale * phi.clamp_min(0.0))  # (B,)
    # shaping weight w that →1 at boundary, →0 away
    # w = torch.exp(-w_scale * torch.relu(phi))   # w≈1 at φ=0, →0 inside/outside
    # w = torch.exp(-w_scale * torch.abs(phi))
    # w = 1.0 / (1.0 + torch.exp(20*(phi - 0.05)))
    # w = torch.exp(-w_scale * phi.clamp_min(0.0))  # (B,1)
    # w = 1.0 / (1.0 + (phi / 0.1)**2)  # slower falloff
    # w = (1.0 / (torch.clamp(phi, min=eps)))**w_scale ## the metric blow up at the boundary
            ## p > 1 makes blow-up sharper -- 2.0 or 3.0
    
    w = torch.exp(-w_scale * torch.relu(phi))
    # # Barrier weight for near-boundary emphasis
    # w  = 1.0 / (torch.clamp(phi, min=eps) ** w_scale)                              # (B,)
    # w = torch.clamp(w, max=1e4) 
    # wb = torch.clamp(w / (w + 1.0), 0.0, 1.0)                                # (B,) smooth 0..1

        
    # Metric construction per point
    I = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(a.shape[0], -1, -1) # (B,2,2)
    nnT = torch.einsum('bi,bj->bij', n, n)                                     # (B,2,2)
    # M = I + w[:, None, None] * (alpha * nnT + beta * (I - nnT))                # (B,2,2)
    Msym = I + w[:,None, None]*(alpha*nnT + beta*(I-nnT))
    # M = M + 1e-6 * torch.eye(2, device=a.device)[None]
    ## alpha * nnT --> penalises motion in the direction of collision
    ## beta * (I-nnT) --> penalises tangential motion (sideways)
        ## vectors are tangent to the circle boundary 
        ## i.e. if you move in those directions, you “slide around” the obstacle instead of going in or out
    ### alpha ≫ beta → so normal direction is much more expensive than tangent
    ### w controls how strong this effect is depending on signed distance
    ### φ = 0 → on boundary → shaping max
    ### φ far > 0 or < 0 → shaping fades out
    # Invert the metric for each batch element (soft projection)

    # ---- Randers skew  ----
    # tangent direction
    # tangent = torch.stack([-n[:,1], n[:,0]], dim=-1)   # rot90(n)
    t1 = torch.stack([-n[:,1], n[:,0]], dim=-1)   # rot90(n)
    side = torch.sign((v * t1).sum(-1, keepdim=True)) 
    tangent = side * t1   # rot90(n)
    # swirl magnitude
    b = drift_gain * w                                 # (B,1)
    # construct skew matrix S(n) = b * [0 -1; 1 0] projected into tangent frame
    # simplest: swirl along tangent direction only
    # S = torch.zeros_like(Msym)
    # S[:,0,1] = -b.squeeze(-1) * tangent[:,0] * tangent[:,1]
    # S[:,1,0] =  b.squeeze(-1) * tangent[:,0] * tangent[:,1]
    tnT = torch.einsum('bi,bj->bij', tangent, n)
    S = b[:,None,None] * (tnT - tnT.transpose(1,2))
    # full Randers metric
    M = Msym + S
    M = M + 1e-6*I
    
    M_inv = torch.linalg.inv(M)                                                # (B,2,2)
    ## use M⁻¹ to “pull” the velocity into safe tangent direction
    # Riemannian projection
    v_shaped = torch.bmm(M_inv, v.unsqueeze(-1)).squeeze(-1)                   # (B,2)
    # v_shaped = torch.linalg.solve(M, v.unsqueeze(-1)).squeeze(-1)

    # # Randers drift β(x): tangential “current” around obstacle
    # vdot = (v * n).sum(-1, keepdim=True)                                     # (B,1)
    # beta_vec = drift_gain * w.unsqueeze(-1) * _rot90(n) * vdot.abs()          # (B,2)

    # # Blend with original velocity so far-away behaviour is unchanged
    # wb = wb.unsqueeze(-1) * blend                                            # (B,1)
    # v_shaped = (1.0 - wb) * v + wb * (v_shaped + beta_vec)

    v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
    v_shaped = v_shaped / (v_shaped.norm(dim=-1, keepdim=True) + 1e-6) * v_norm
    return v_shaped

class ShapedVectorFieldWrapper_CASF(VectorFieldWrapper):
    """
    CASF-style metric-weighted streaming flow.
    Applies continuous, differentiable constraint shaping:
        ṽ(a) = M(a)^(-1) vθ(a)
            center, radius,
            alpha=10.0, beta=1.0, w_scale=50.0,
            eps=1e-6
    """

    def __init__(self, model: nn.Module, obs_cond: torch.Tensor, shapingConfig:Optional[dict]=None,
                 ):
        super().__init__(model, obs_cond)
        device = obs_cond.device

        self.center = torch.as_tensor(shapingConfig['center_norm'], dtype=torch.float32, device=device)
        self.radius = float(shapingConfig['radius_norm'])
        self.alpha = shapingConfig['alpha']
        self.beta = shapingConfig['beta']
        self.w_scale = shapingConfig['w_scale']
        self.eps = shapingConfig['eps']
        self.drift_gain = shapingConfig['drift_gain']

    def forward(self, t: torch.Tensor, x_norm: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 1. Predict nominal normalized velocity
        v_norm = super().forward(t, x_norm)                # (B,2)
        # v_norm_shaped = shape_velocity_batch_metric_CASF(
        #     x_norm, v_norm, self.center, self.radius,
        #     alpha=self.alpha, beta=self.beta, w_scale=self.w_scale, eps=self.eps
        # )
        v_norm_shaped = shape_velocity_batch_metric_rander(
            x_norm, v_norm, self.center, self.radius,
            alpha=self.alpha, beta=self.beta, w_scale=self.w_scale, eps=self.eps,drift_gain=self.drift_gain
        )

        return v_norm_shaped

### ------- postShaping: obstacle ------- ###

### gpt version ### 
def shape_velocity_multi_obstacle_rander(
    a: torch.Tensor,                 # (B,2)
    v: torch.Tensor,                 # (B,2)
    centers: torch.Tensor,           # (K,2)
    radii: torch.Tensor,             # (K,)
    alpha=10.0,
    beta = 1.0,
    w_scale=30.0,
    drift_gain=0.3,
    eps=1e-8,
):
    """
    Multi-Obstacle CASF + Randers metric shaping.
    Smooth combination of obstacles using a soft-min over SDFs.
    """

    device, dtype = a.device, a.dtype
    B = a.shape[0]
    K = centers.shape[0]

    # Expand
    a_exp      = a[:,None,:]                  # (B,1,2)
    centers    = centers[None,:,:]            # (1,K,2)
    radii      = radii[None,:]                # (1,K)

    # ---------------------------------------------------------
    # 1. Compute SDF to each obstacle
    # ---------------------------------------------------------
    rel = a_exp - centers                     # (B,K,2)
    dist = torch.sqrt((rel**2).sum(-1) + eps) # (B,K)
    phi  = dist - radii                       # (B,K)

    # normals per obstacle
    n_i = rel / (dist[:,:,None] + eps)        # (B,K,2)

    # ---------------------------------------------------------
    # 2. Combine obstacles with **soft-min** (smooth)
    # ---------------------------------------------------------
    # soft-min using logsumexp trick:
    # min(phi_1,...,phi_K) ~ -1/γ * log(sum(exp(-γ phi_k)))
    gamma = 50.0
    softmin_phi = - (1.0/gamma) * torch.logsumexp(-gamma * phi, dim=1)   # (B,)

    # weights for blending normals (smooth signed-distance weights)
    w_raw = torch.exp(-w_scale * phi.clamp_min(0.0))     # (B,K)
    w_weights = w_raw / (w_raw.sum(dim=1, keepdim=True) + eps)   # (B,K)

    # blended normal
    n = (w_weights[:,:,None] * n_i).sum(dim=1)           # (B,2)
    n = n / (n.norm(dim=-1, keepdim=True) + eps)

    phi_c = softmin_phi                                  # (B,)

    # ---------------------------------------------------------
    # 3. Metric weight w(φ)
    # ---------------------------------------------------------
    w = torch.exp(-w_scale * phi_c.clamp_min(0.0))       # (B,)

    # ---------------------------------------------------------
    # 4. Build metric M = I + w [ α n nᵀ + β (I − n nᵀ) ]
    # ---------------------------------------------------------
    I = torch.eye(2, device=device, dtype=dtype)[None].expand(B,-1,-1)
    nnT = torch.einsum('bi,bj->bij', n, n)

    Msym = I + w[:,None,None]*(alpha*nnT + beta*(I - nnT))

    # ---------------------------------------------------------
    # 5. Randers skew (drift)
    # ---------------------------------------------------------
    # tangent = rot90(n)
    tangent = torch.stack([-n[:,1], n[:,0]], dim=-1)

    # choose swirl direction based on v·tangent
    side = torch.sign((v * tangent).sum(-1, keepdim=True)) 
    tangent = tangent * side

    b = drift_gain * w                                   # (B,)
    tnT = torch.einsum('bi,bj->bij', tangent, n)
    S = b[:,None,None] * (tnT - tnT.transpose(1,2))

    M = Msym + S + 1e-6 * I

    # ---------------------------------------------------------
    # 6. Solve M ṽ = v
    # ---------------------------------------------------------
    v_shaped = torch.linalg.solve(M, v.unsqueeze(-1)).squeeze(-1)

    # Preserve magnitude
    v_norm = v.norm(dim=-1, keepdim=True) + eps
    v_shaped = v_shaped / (v_shaped.norm(dim=-1, keepdim=True) + eps) * v_norm
    
    return v_shaped

def shape_velocity_batch_metric_rander_multi(
    a: torch.Tensor,           # (B, 2) current position
    v: torch.Tensor,           # (B, 2) nominal velocity
    centers: torch.Tensor,     # (N, 2) obstacle centers
    radii: torch.Tensor,       # (N,) obstacle radii
    alpha=10.0,
    beta=1.0,
    w_scale=50.0,
    eps=1e-6,
    drift_gain=0.03
) -> torch.Tensor:
    """
    Multi-obstacle CASF Shaping using Randers Metric.
    Computes metric based on the closest obstacle.
    """
    device = a.device
    B = a.shape[0]
    
    # 1. Compute signed distances (phi) and normals (n) for all obstacles
    # a: (B, 2) -> (1, B, 2)
    # centers: (N, 2) -> (N, 1, 2)
    rel = a.unsqueeze(0) - centers.unsqueeze(1) # (N, B, 2)
    dists = torch.norm(rel, dim=-1) # (N, B)
    
    # phi = dist - radius
    # radii: (N,) -> (N, 1)
    phis = dists - radii.unsqueeze(1) # (N, B)

    # Normals: (N, B, 2)
    normals = rel / (dists.unsqueeze(-1) + eps) 

    # 2. Find the closest obstacle (minimum phi)
    # We only shape based on the obstacle we are most likely to hit
    min_phis, closest_indices = torch.min(phis, dim=0) # (B,)
    
    # Gather the normals of the closest obstacles
    # closest_indices is (B,), we need to select from normals (N, B, 2)
    # Expand indices to (1, B, 2) for gathering
    idx_expanded = closest_indices.view(1, B, 1).expand(1, B, 2)
    n_closest = torch.gather(normals, 0, idx_expanded).squeeze(0) # (B, 2)

    # 3. Compute Weight (w) based on min_phi
    # w -> 1 at boundary, 0 away
    w = torch.exp(-w_scale * torch.relu(min_phis)) # (B,)
    
    # 4. Construct Metric Matrix M
    I = torch.eye(2, device=device, dtype=a.dtype).unsqueeze(0).expand(B, -1, -1)
    nnT = torch.einsum('bi,bj->bij', n_closest, n_closest) # (B, 2, 2)
    
    # Symmetric component (Riemannian)
    Msym = I + w[:, None, None] * (alpha * nnT + beta * (I - nnT))

    # 5. Randers Skew (Drift)
    # Tangent vector t = [-ny, nx] * sign(v dot t)
    t_raw = torch.stack([-n_closest[:,1], n_closest[:,0]], dim=-1)
    side = torch.sign((v * t_raw).sum(-1, keepdim=True))
    tangent = side * t_raw
    
    tnT = torch.einsum('bi,bj->bij', tangent, n_closest)
    
    # Skew strength
    b = drift_gain * w
    S = b[:, None, None] * (tnT - tnT.transpose(1, 2))
    
    # Total Metric
    M = Msym + S + 1e-6 * I
    
    # 6. Apply Shaping: v_tilde = M^-1 * v
    M_inv = torch.linalg.inv(M)
    v_shaped = torch.bmm(M_inv, v.unsqueeze(-1)).squeeze(-1)
    
    # 7. Preserve Magnitude (Normalization)
    v_norm_orig = torch.norm(v, dim=-1, keepdim=True) + 1e-8
    v_shaped = v_shaped / (v_shaped.norm(dim=-1, keepdim=True) + 1e-6) * v_norm_orig
    
    return v_shaped

class ShapedVectorFieldWrapper_multi_CASF(VectorFieldWrapper):
    """
    Wrapper handling Multi-Obstacle configuration.
    """
    def __init__(self, model: nn.Module, obs_cond: torch.Tensor, shapingConfig: Optional[dict] = None):
        super().__init__(model, obs_cond)
        device = obs_cond.device

        # Parse Centers: Expecting (N, 2)
        centers = shapingConfig['center_norm']
        if not torch.is_tensor(centers):
            centers = torch.tensor(centers, dtype=torch.float32, device=device)
        if centers.ndim == 1:
            centers = centers.unsqueeze(0) # Handle single obstacle case
        self.centers = centers

        # Parse Radii: Expecting (N,) or scalar
        radii = shapingConfig['radius_norm']
        if not torch.is_tensor(radii):
            radii = torch.tensor(radii, dtype=torch.float32, device=device)
        if radii.ndim == 0:
            # If scalar, repeat for all centers
            radii = radii.expand(self.centers.shape[0])
        self.radii = radii

        self.alpha = shapingConfig.get('alpha', 10.0)
        self.beta = shapingConfig.get('beta', 0.0)
        self.w_scale = shapingConfig.get('w_scale', 50.0)
        self.eps = shapingConfig.get('eps', 1e-6)
        self.drift_gain = shapingConfig.get('drift_gain', 0.1)

    def forward(self, t: torch.Tensor, x_norm: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Predict nominal velocity
        v_norm = super().forward(t, x_norm)
        
        # Apply Multi-Obstacle Shaping
        v_norm_shaped = shape_velocity_batch_metric_rander_multi(
            x_norm, v_norm, 
            self.centers, self.radii,
            alpha=self.alpha, beta=self.beta, 
            w_scale=self.w_scale, eps=self.eps, 
            drift_gain=self.drift_gain
        )
        return v_norm_shaped


###########################################################################

### ------- postShaping: wall ------- ###

def shape_velocity_batch_metric_wall_CASF(
    a: torch.Tensor,             # (B, 2) real positions/actions
    v: torch.Tensor,             # (B, 2) raw velocity field
    box_bounds: tuple,           # (x_min, y_min, x_max, y_max)
    alpha=10.0,
    beta=1.0,
    w_scale=1.0,
    eps=1e-6,
) -> torch.Tensor:
    """
    CASF-style metric-weight shaping for wall (box) constraints.
    v_tilde = M(a)^(-1) v_theta(a)
    Each point computes distance and normal to the *nearest wall*,
    then applies the local metric M(a) and its inverse.
    """

    x_min, y_min, x_max, y_max = box_bounds
    B = a.shape[0]
    device = a.device

    # Distances to each wall (positive inside the box)
    dl = a[:, 0] - x_min     # left
    dr = x_max - a[:, 0]     # right
    db = a[:, 1] - y_min     # bottom
    dt = y_max - a[:, 1]     # top

    dists = torch.stack([dl, dr, db, dt], dim=1)          # (B, 4)
    phi, idx = torch.min(dists, dim=1)                    # nearest wall & distance
    phi = phi.clamp_min(0.0)                              # ensure positive

    # Outward normals for each wall
    n_table = torch.tensor([
        [+1.0,  0.0],   # left wall normal
        [-1.0,  0.0],   # right wall normal
        [ 0.0, +1.0],   # bottom wall normal
        [ 0.0, -1.0],   # top wall normal
    ], device=device, dtype=torch.float32)
    n = n_table[idx]                                      # (B,2) outward normal to nearest wall

    # Smooth influence weight
    # Option 1: exponential decay from wall (Gaussian-like)
    # w = torch.exp(-w_scale * phi)    
    # w = torch.exp(-w_scale * phi ** 2)# (B,)NOTE!
    # Option 2 (alternative): polynomial decay (commented)
    # w = alpha / torch.clamp(phi + eps, min=eps).pow(w_scale) # p
    # Option 3 (alternative): --- band-limited polynomial weight ---
    # d0 = 0.10 * min(x_max - x_min, y_max - y_min)   # 10% band
    d0 = 0.01 * min(x_max - x_min, y_max - y_min)
    s  = torch.clamp(d0 - phi, min=0.0) / d0        # [0,1]
    w  = (s ** 2) * w_scale                         # p=2 (default), w_scale as overall gain
    w  = torch.clamp(w, max=1e3)                  # optional

    # Metric definition
    I = torch.eye(2, device=device).unsqueeze(0).expand(B, -1, -1)
    nnT = torch.einsum('bi,bj->bij', n, n)
    M = I + w[:, None, None] * (alpha * nnT + beta * (I - nnT))  # (B,2,2)

    # Invert the metric
    M_inv = torch.linalg.inv(M)                          # (B,2,2)

    # Apply metric-weighted projection
    v_shaped = torch.bmm(M_inv, v.unsqueeze(-1)).squeeze(-1)  # (B,2)
    return v_shaped

class ShapedVectorFieldWrapperWall_CASF(VectorFieldWrapper):
    """
    CASF-style wall (box) constraint shaping.
    Applies M(a)^{-1} * vθ(a) using the local metric near walls.
    """

    def __init__(self, model: nn.Module, obs_cond: torch.Tensor,
                 shapingConfig:Optional[dict]=None):
        super().__init__(model, obs_cond)
        device = obs_cond.device

        # Convert bounds and limits
        self.box_bounds = tuple(float(b) for b in shapingConfig['box_bounds'])

        self.alpha = shapingConfig['alpha']
        self.beta = shapingConfig['beta']
        self.w_scale = shapingConfig['w_scale']
        self.eps = shapingConfig['eps']

    def forward(self, t: torch.Tensor, x_norm: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 1. Predict nominal velocity (normalized)
        v_norm = super().forward(t, x_norm)          # (B,2)

        # 2. Apply CASF metric-weight shaping near walls
        v_norm_shaped = shape_velocity_batch_metric_wall_CASF(
            x_norm, v_norm, self.box_bounds,
            alpha=self.alpha, beta=self.beta,
            w_scale=self.w_scale, eps=self.eps
        )

        return v_norm_shaped

### ------- postShaping: wall ------- ###

###########################################################################

### ------- postShaping: maze ------- ###
import torch.nn.functional as F
@torch.inference_mode()
def query_sdf_and_grad(a, X, Y, sdf):
    """Interpolate φ(x) and ∇φ(x) for each query a=(x,y)."""
    device = a.device
    H, W = sdf.shape
    sdf_t = torch.as_tensor(sdf, dtype=torch.float32, device=device)[None,None,:,:]
    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(Y.min()), float(Y.max())

    gx = 2 * (a[:,0] - x_min)/(x_max - x_min) - 1
    gy = 2 * (a[:,1] - y_min)/(y_max - y_min) - 1
    grid = torch.stack([gx, gy], dim=-1).view(1,a.shape[0],1,2)

    phi = F.grid_sample(sdf_t, grid, align_corners=True, mode='bilinear').view(-1)

    dx = float(X[0,1]-X[0,0]); dy = float(Y[1,0]-Y[0,0])
    sdf_torch = torch.as_tensor(sdf, dtype=torch.float32, device=device)
    sdf_dx = (sdf_torch[:,2:]-sdf_torch[:,:-2])/(2*dx)
    sdf_dy = (sdf_torch[2:,:]-sdf_torch[:-2,:])/(2*dy)
    sdf_dx = F.pad(sdf_dx[None,None,:,:], (1,1,0,0), mode='replicate')
    sdf_dy = F.pad(sdf_dy[None,None,:,:], (0,0,1,1), mode='replicate')
    grad_x = F.grid_sample(sdf_dx, grid, align_corners=True, mode='bilinear').view(-1)
    grad_y = F.grid_sample(sdf_dy, grid, align_corners=True, mode='bilinear').view(-1)
    grad = torch.stack([grad_x,grad_y],dim=-1)
    n = F.normalize(grad, dim=-1, eps=1e-6)
    return phi, n

def shape_velocity_batch_metric_corridor_CASF(
    a, v, X, Y, sdf,
    alpha=10.0, beta=1.0, w_scale=50.0, eps=1e-6):
    """
    CASF-style velocity shaping w.r.t. corridor boundary defined by SDF.
    Keeps flow inside by inflating metric near walls.
    """
    phi, n = query_sdf_and_grad(a, X, Y, sdf)   # (B,), (B,2)
    phi = phi.clamp_min(0.0)                    # only outside distance counts
    w = torch.exp(-w_scale * phi)               # decays fast away from wall

    I = torch.eye(2, device=a.device).unsqueeze(0).expand(a.size(0), -1, -1)
    nnT = torch.einsum('bi,bj->bij', n, n)
    M = I + w[:,None,None]*(alpha*nnT + beta*(I - nnT))
    M_inv = torch.linalg.inv(M)
    v_shaped = torch.bmm(M_inv, v.unsqueeze(-1)).squeeze(-1)
    return v_shaped

class ShapedVectorFieldWrapperCorridor_CASF(VectorFieldWrapper):
    """
    CASF-style shaping for S-shaped corridor constraint.
    Applies metric-weighted velocity shaping based on precomputed SDF.
    """
    def __init__(self, model, obs_cond, shapingConfig):
        super().__init__(model, obs_cond)
        device = obs_cond.device

        self.X = shapingConfig['corridorX']
        self.Y = shapingConfig['corridorY']
        self.sdf = shapingConfig['corridorSDF']
        self.alpha = shapingConfig['alpha']
        self.beta = shapingConfig['beta']
        self.w_scale = shapingConfig['w_scale']
        self.eps = shapingConfig['eps']

    def forward(self, t, x_norm, *args, **kwargs):
        v_norm = super().forward(t, x_norm)
        v_norm_shaped = shape_velocity_batch_metric_corridor_CASF(
            x_norm, v_norm, self.X, self.Y, self.sdf,
            alpha=self.alpha, beta=self.beta, w_scale=self.w_scale, eps=self.eps
        )
        return v_norm_shaped
### ------- postShaping: maze ------- ###


'''

    w = torch.exp(-w_scale * torch.relu(phi))

    # Metric construction per point
    I = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(a.shape[0], -1, -1) # (B,2,2)
    nnT = torch.einsum('bi,bj->bij', n, n)                                     # (B,2,2)
    Msym = I + w[:,None, None]*(alpha*nnT + beta*(I-nnT))

    # ---- Randers skew  ----
    t1 = torch.stack([-n[:,1], n[:,0]], dim=-1)   # rot90(n)
    side = torch.sign((v * t1).sum(-1, keepdim=True)) 
    tangent = side * t1   # rot90(n)
    b = drift_gain * w                                 # (B,1)
    tnT = torch.einsum('bi,bj->bij', tangent, n)
    S = b[:,None,None] * (tnT - tnT.transpose(1,2))
    # full Randers metric
    M = Msym + S
    M = M + 1e-6*I
    
    M_inv = torch.linalg.inv(M)                                                # (B,2,2)
    # Riemannian projection
    v_shaped = torch.bmm(M_inv, v.unsqueeze(-1)).squeeze(-1)                   # (B,2)
    v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
    v_shaped = v_shaped / (v_shaped.norm(dim=-1, keepdim=True) + 1e-6) * v_norm
'''
