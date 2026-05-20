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

    w = torch.exp(-w_scale * torch.relu(phi))
        
    # Metric construction per point
    I = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(a.shape[0], -1, -1) # (B,2,2)
    nnT = torch.einsum('bi,bj->bij', n, n)                                     # (B,2,2)
    Msym = I + w[:,None, None]*(alpha*nnT + beta*(I-nnT))
    
    # ---- Randers skew  ----
    # tangent direction
    t1 = torch.stack([-n[:,1], n[:,0]], dim=-1)   # rot90(n)
    side = torch.sign((v * t1).sum(-1, keepdim=True)) 
    tangent = side * t1   # rot90(n)
    # swirl magnitude
    b = drift_gain * w                                 # (B,1)
    tnT = torch.einsum('bi,bj->bij', tangent, n)
    S = b[:,None,None] * (tnT - tnT.transpose(1,2))
    
    # full Randers metric
    M = Msym + S
    M = M + 1e-6*I
    
    M_inv = torch.linalg.inv(M)                                                # (B,2,2)
    v_shaped = torch.bmm(M_inv, v.unsqueeze(-1)).squeeze(-1)                   # (B,2)

    v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
    v_shaped = v_shaped / (v_shaped.norm(dim=-1, keepdim=True) + 1e-6) * v_norm
    return v_shaped
