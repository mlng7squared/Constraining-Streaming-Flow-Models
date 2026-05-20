from typing import Dict, Optional
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from pydrake.all import PiecewisePolynomial
from torchdyn.core import NeuralODE

from streaming_flow_policy.pusht.dp_state_notebook.base_policy import Policy


class StreamingFlowPolicyDeterministic (Policy):
    def __init__(self,
                 velocity_net: nn.Module,
                 action_dim: int,
                 pred_horizon: int = 16,
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
        if not np.all(action[0] == obs[-1, :2]):
            # print('DEBUGGGG')
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
        
        return {
            'obs': obs,  # (OBS_HORIZON, OBS_DIM) # (2,5)
            'action': action, 
            'x': x.astype(np.float32),  # (1, ACTION_DIM) # (1,2)
            'v': v.astype(np.float32),  # (1, ACTION_DIM) # (1,2)
            't': time,  # (,)
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
        obs = batch['obs'].to(self.device)  # (B, OBS_HORIZON, OBS_DIM) 
        x = batch['x'].to(self.device)  # (B, 1, ACTION_DIM)
        v = batch['v'].to(self.device)  # (B, 1, ACTION_DIM)
        t = batch['t'].to(self.device)  # (B,)
        B = obs.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs.flatten(start_dim=1)  # (B, OBS_HORIZON * OBS_DIM) (B, 16*2)

        # predict the velocity
        v_pred = self.velocity_net(
            sample=x, timestep=t, global_cond=obs_cond
        )  # (B, 1, ACTION_DIM)

        # L2 loss
        loss_mse = nn.functional.mse_loss(v_pred, v)  # (,)
        loss_cos = 1 - torch.nn.functional.cosine_similarity(v_pred, v, dim=-1).mean()

        loss = loss_mse
        # loss = loss_mse + 0.5 * loss_cos

        return loss

    @torch.inference_mode()
    def __call__(self,
                 nobs: Tensor,
                 num_actions: Optional[int] = None,
                 integration_steps_per_action: int = 6,
                 postShaping: str = None,
                 shapingConfig: Dict = None
    ) -> Tensor:
        """
        Args:
            nobs (Tensor, shape=(OBS_HORIZON, OBS_DIM)): normalized observations
            num_actions (Optional[int]): number of actions to predict
            integration_steps_per_action (int): number of integration steps per action

        Returns:
            Tensor (shape=(1, NUM_ACTIONS, ACTION_DIM)): predicted actions
        """
        obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)  # (1, OBS_HORIZON * OBS_DIM)
        # obs_cond = nobs.flatten(start_dim=1)   # (B, OBS_HORIZON*OBS_DIM) (B,2*4)

        # if postShaping == 'obstacle':
        #     vf = ShapedVectorFieldWrapper_CASF(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
        # elif postShaping == 'wall':
        #     vf = ShapedVectorFieldWrapper_casfWALL(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
        # else:  # postShaping=None 
        #     print('NO SHAPING')
        #     vf = VectorFieldWrapper(self.velocity_net, obs_cond)
        # ode_solver = NeuralODE(
        #     vector_field=vf,
        #     solver="euler", # "dopri5",
        #     sensitivity="adjoint",
        #     atol=1e-6,
        #     rtol=1e-6,
        # )

        # Integration time steps.
        num_actions = num_actions or self.pred_horizon.item()
        assert 1 <= num_actions <= self.pred_horizon.item()
        num_future_actions = num_actions - 1
        t_max = num_future_actions / (self.pred_horizon.item() - 1)
        total_integration_steps = 1 + num_future_actions * integration_steps_per_action
        t_span = torch.linspace(0, t_max, total_integration_steps)


        x = nobs[-1, :2]  # (2,)
        # traj = ode_solver.trajectory(x=x, t_span=t_span)  # (K, 2)

        # --- self-written rollout [Eucler!] --- # TODO: try midpoint/dopri5
        Δt = t_span[1] - t_span[0]
        traj = [x]
        traj_shaped = [x]
        vel_traj,vel_traj_shaped = [],[]
        # --- rollout ---
        for i in range(total_integration_steps):
            t = t_span[i].to(x.device)
            # query velocity net (vector field)
            if postShaping == 'obstacle':
                vf = VectorFieldWrapper(self.velocity_net, obs_cond)
                vf_shaped = ShapedVectorFieldWrapper_CASF(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
                shaping = True
            elif postShaping == 'wall':
                vf = VectorFieldWrapper(self.velocity_net, obs_cond)
                vf_shaped = ShapedVectorFieldWrapper_casfWALL(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
                shaping = True
            elif postShaping == 'wall-cbf':
                vf = VectorFieldWrapper(self.velocity_net, obs_cond)
                vf_shaped = ShapedVectorFieldWrapper_cbfWALL(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
                shaping = True
            elif postShaping == 'wall-hardBarrier':
                vf = VectorFieldWrapper(self.velocity_net, obs_cond)
                vf_shaped = ShapedVectorFieldWrapper_hardWALL(self.velocity_net, obs_cond, shapingConfig=shapingConfig)
                shaping = True
            elif postShaping == 'maze':
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
            
            # # MIDPOINT/RK2
            # k1 = vf(t, x)
            # x_mid = x + 0.5 * Δt * k1
            # k2 = vf(t + 0.5*Δt, x_mid)
            # v  = k2
            # x_next = x + Δt * v            
            
            # # RK4
            # k1 = vf(t, x)
            # k2 = vf(t + 0.5*Δt, x + 0.5*Δt*k1)
            # k3 = vf(t + 0.5*Δt, x + 0.5*Δt*k2)
            # k4 = vf(t + 1.0*Δt, x + 1.0*Δt*k3)
            # v  = (k1 + 2*k2 + 2*k3 + k4) / 6.0
            # x_next = x + Δt * v
            
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
        traj = torch.stack(traj, dim=0)       # (B, T+1, 2)
        traj = traj[:-1,:]
        vel_traj = torch.stack(vel_traj, dim=0)  # (B, T, 2)
        if shaping:
            traj_shaped = torch.stack(traj_shaped, dim=0)
            traj = traj_shaped[:-1,:]
            vel_traj = torch.stack(vel_traj_shaped, dim=0)  # (B, T-1, 2)
        # --- self-written rollout [Eucler!] --- # TODO: try midpoint

        select_action_indices = np.arange(
            0,
            total_integration_steps,
            integration_steps_per_action,
        )
        naction = traj[select_action_indices]  # (NUM_ACTIONS, 2)
        naction = naction.unsqueeze(0)  # (1, NUM_ACTIONS, 2)
        return naction

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
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, ACTION_DIM)
        v: Tensor = self.model(
            sample=x,
            timestep=t.repeat(x.shape[0]),
            global_cond=self.obs_cond,
        )  # (1, 1, ACTION_DIM)
        v = v.flatten()  # (ACTION_DIM,)
        return v


### ------- postShaping: obstacle ------- ###
def shape_velocity_metric_rander_single(
    a: torch.Tensor,      # (2,)
    v: torch.Tensor,      # (2,)
    center: torch.Tensor, # (2,)
    radius: float,
    alpha=10.0,
    beta=1.0,
    w_scale=50.0,
    eps=1e-6,
    drift_gain=0.03,
) -> torch.Tensor:
    """
    Non-batched CASF Randers shaping for 2-D.
    a, v, center are all shape (2,)
    returns shaped velocity (2,)
    """
    device = a.device
    dtype  = a.dtype

    # ensure small correct shapes
    a      = a.to(device=device, dtype=dtype)
    v      = v.to(device=device, dtype=dtype)
    center = center.to(device=device, dtype=dtype)

    rel  = a - center                     # (2,)
    dist = torch.sqrt((rel*rel).sum() + eps)
    n    = rel / (dist + eps)             # (2,)
    phi  = dist - radius                  # scalar

    # weighting
    w = torch.exp(-w_scale * torch.relu(phi))

    I   = torch.eye(2, device=device, dtype=dtype)
    nnT = torch.outer(n, n)
    Msym = I + w * (alpha*nnT + beta*(I - nnT))

    # Randers skew
    t1 = torch.stack([-n[1], n[0]])         # 90deg rotated
    side = torch.sign((v*t1).sum())         # scalar
    # side = tanh(kappa * (v * t1).sum(-1, keepdim=True)) # scalar
    tangent = side * t1

    tnT = torch.outer(tangent, n)
    S = drift_gain * w * (tnT - tnT.T)

    M = Msym + S + 1e-6*I
    M_inv = torch.linalg.inv(M)

    v_shaped = M_inv @ v                    # (2,)
    v_shaped = v_shaped / (v_shaped.norm()+1e-6) * (v.norm()+1e-6)
    return v_shaped

def shape_velocity_hardB_wall(
    a: torch.Tensor,      # (2,)
    v: torch.Tensor,      # (2,)
    alpha=1.0,
    eps=1e-6,
) -> torch.Tensor:
    device = a.device
    dtype  = a.dtype
    
    a = a.to(device,dtype)
    v = v.to(device,dtype)
    # ----------------------------- wall SDF ---------------------------------
    # room bounds (PushT)
    low  = -1
    high = 1

    phi_left   =  a[0] - low
    phi_right  = -(a[0] - high)
    phi_bottom =  a[1] - low
    phi_top    = -(a[1] - high)

    phi_all = torch.stack([phi_left,phi_right,phi_bottom,phi_top])
    idx = phi_all.argmin()
    phi = phi_all[idx]

    normals = torch.tensor([
        [ +1, 0],   # left wall   (-1, y)
        [ -1, 0],   # right wall  (+1, y)
        [ 0, +1],   # bottom wall (x, -1)
        [ 0, -1],   # top wall    (x, +1)
    ], device=device, dtype=dtype)

    n = normals[idx]


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
    v_shaped = torch.where(
        needs_projection,  # (B, 1) - Unsqueeze to broadcast with (B, 2)
        v_tangent,                  # (B, 2) - Projected (safe) velocity
        v                           # (B, 2) - Original (nominal) velocity
    )

    v_shaped = v_shaped / (v_shaped.norm()+1e-6) * (v.norm()+1e-6)
    return v_shaped

    
def shape_velocity_cbf_wall(
    a: torch.Tensor,      # (2,)
    v: torch.Tensor,      # (2,)
    alpha=10.0,
    eps=1e-6,
) -> torch.Tensor:
    device = a.device
    dtype  = a.dtype


    a = a.to(device,dtype)
    v = v.to(device,dtype)
    # ----------------------------- wall SDF ---------------------------------
    # room bounds (PushT)
    low  = -1
    high = 1

    phi_left   =  a[0] - low
    phi_right  = -(a[0] - high)
    phi_bottom =  a[1] - low
    phi_top    = -(a[1] - high)

    phi_all = torch.stack([phi_left,phi_right,phi_bottom,phi_top])
    idx = phi_all.argmin()
    phi = phi_all[idx]

    normals = torch.tensor([
        [ +1, 0],   # left wall   (-1, y)
        [ -1, 0],   # right wall  (+1, y)
        [ 0, +1],   # bottom wall (x, -1)
        [ 0, -1],   # top wall    (x, +1)
    ], device=device, dtype=dtype)

    n = normals[idx]


    ######-cbf-barrier residual: r = -kappa*h - ∇h·v -######
    # grad_h_dot_v = (n * v).sum(-1)                    # (B,)
    grad_h_dot_v = torch.sum(n * v, dim=-1, keepdim=True)
    
    r = -alpha*phi - grad_h_dot_v                       # (B,)
    
    # Only correct when residual is positive (violation)
    lam = torch.relu(r)/ (torch.sum(n*n, dim=-1, keepdim=True) + eps) 
    v_shaped = v + lam * n
    
    v_shaped = v_shaped / (v_shaped.norm()+1e-6) * (v.norm()+1e-6)
    return v_shaped

def shape_velocity_metric_rander_wall(
    a: torch.Tensor,      # (2,)
    v: torch.Tensor,      # (2,)
    alpha=10.0,
    beta=1.0,
    w_scale=50.0,
    eps=1e-6,
    drift_gain=0.03,
) -> torch.Tensor:
    device = a.device
    dtype  = a.dtype

    a = a.to(device,dtype)
    v = v.to(device,dtype)
    # ----------------------------- wall SDF ---------------------------------
    # room bounds (PushT)
    low  = -1
    high = 1

    phi_left   =  a[0] - low
    phi_right  = -(a[0] - high)
    phi_bottom =  a[1] - low
    phi_top    = -(a[1] - high)

    phi_all = torch.stack([phi_left,phi_right,phi_bottom,phi_top])
    idx = phi_all.argmin()
    phi = phi_all[idx]

    normals = torch.tensor([
        [ +1, 0],   # left wall   (-1, y)
        [ -1, 0],   # right wall  (+1, y)
        [ 0, +1],   # bottom wall (x, -1)
        [ 0, -1],   # top wall    (x, +1)
    ], device=device, dtype=dtype)

    n = normals[idx]

    # weighting
    w = torch.exp(-w_scale * torch.relu(phi))

    # symmetric metric core
    I   = torch.eye(2, device=device, dtype=dtype)
    nnT = torch.outer(n, n)
    Msym = I + w * (alpha*nnT + beta*(I - nnT))

    # Randers skew
    t1 = torch.stack([-n[1], n[0]])       # 90deg rotated
    side = torch.sign((v*t1).sum())       # scalar
    tangent = side * t1
    tnT = torch.outer(tangent, n)

    S = drift_gain * w * (tnT - tnT.T)

    M = Msym + S + 1e-6*I
    M_inv = torch.linalg.inv(M)

    v_shaped = M_inv @ v
    v_shaped = v_shaped / (v_shaped.norm()+1e-6) * (v.norm()+1e-6)
    return v_shaped


class ShapedVectorFieldWrapper_CASF(VectorFieldWrapper):
    def __init__(self, model, obs_cond, shapingConfig):
        super().__init__(model, obs_cond)
        d = obs_cond.device
        self.center = torch.as_tensor(shapingConfig["center_norm"], device=d, dtype=torch.float32)
        self.radius = float(shapingConfig["radius_norm"])
        self.alpha  = shapingConfig["alpha"]
        self.beta   = shapingConfig["beta"]
        self.w_scale= shapingConfig["w_scale"]
        self.eps    = shapingConfig["eps"]
        self.drift_gain = shapingConfig["drift_gain"]

    def forward(self, t, x_norm, *args, **kwargs):
        v = super().forward(t, x_norm)   # (2,)    <-- because VF wrapper flatten() returned (2,)
        v_s = shape_velocity_metric_rander_single(
            x_norm, v,
            self.center, self.radius,
            alpha=self.alpha, beta=self.beta, w_scale=self.w_scale,
            eps=self.eps, drift_gain=self.drift_gain
        )
        return v_s

class ShapedVectorFieldWrapper_hardWALL(VectorFieldWrapper):
    def __init__(self, model, obs_cond, shapingConfig):
        super().__init__(model, obs_cond)
        d = obs_cond.device
        self.alpha  = shapingConfig["alpha"]
        self.eps    = shapingConfig["eps"]

    def forward(self, t, x_norm, *args, **kwargs):
        v = super().forward(t, x_norm)   # (2,)    <-- because VF wrapper flatten() returned (2,)
        v_s = shape_velocity_hardB_wall(
            x_norm, v,
            alpha=self.alpha,
            eps=self.eps,
        )
        return v_s
    
class ShapedVectorFieldWrapper_cbfWALL(VectorFieldWrapper):
    def __init__(self, model, obs_cond, shapingConfig):
        super().__init__(model, obs_cond)
        d = obs_cond.device
        self.alpha  = shapingConfig["alpha"]
        self.eps    = shapingConfig["eps"]

    def forward(self, t, x_norm, *args, **kwargs):
        v = super().forward(t, x_norm)   # (2,)    <-- because VF wrapper flatten() returned (2,)
        v_s = shape_velocity_cbf_wall(
            x_norm, v,
            alpha=self.alpha,
            eps=self.eps,
        )
        return v_s
    
class ShapedVectorFieldWrapper_casfWALL(VectorFieldWrapper):
    def __init__(self, model, obs_cond, shapingConfig):
        super().__init__(model, obs_cond)
        d = obs_cond.device
        self.alpha  = shapingConfig["alpha"]
        self.beta   = shapingConfig["beta"]
        self.w_scale= shapingConfig["w_scale"]
        self.eps    = shapingConfig["eps"]
        self.drift_gain = shapingConfig["drift_gain"]

    def forward(self, t, x_norm, *args, **kwargs):
        v = super().forward(t, x_norm)   # (2,)    <-- because VF wrapper flatten() returned (2,)
        v_s = shape_velocity_metric_rander_wall(
            x_norm, v,
            alpha=self.alpha, beta=self.beta, w_scale=self.w_scale,
            eps=self.eps, drift_gain=self.drift_gain
        )
        return v_s


### BATCH-awared
# class VectorFieldWrapper (nn.Module):
#     """Wraps model to torchdyn compatible format."""
#     def __init__(self, model: nn.Module, obs_cond: Tensor):
#         super().__init__()
#         self.model = model
#         self.obs_cond = obs_cond

#     def forward(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
#         """
#         Args:
#             t (Tensor, shape=(,), dtype=float): time
#             x (Tensor, shape=(ACTION_DIM,), dtype=float): position

#         Returns:
#             Tensor (shape=(ACTION_DIM,), dtype=float): velocity
#         """
#         B = x.shape[0]
#         x = x.unsqueeze(1) 
#         # x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, ACTION_DIM)
#         t_expand = t.repeat(B)  
#         v: Tensor = self.model(
#             sample=x,
#             timestep=t_expand,
#             global_cond=self.obs_cond,
#         )  # (1, 1, ACTION_DIM)
#         # v = v.flatten()  # (ACTION_DIM,)
#         v = v.squeeze(1)  # (B, ACTION_DIM)
#         return v
