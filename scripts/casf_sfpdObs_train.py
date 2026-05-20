"""
All functions in this file have been copied from the Diffusion Policy repo, in
particular, this notebook (diffusion_policy_state_pusht_demo.ipynb):
https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B
"""
# diffusion policy import
from typing import Dict, Callable
from pathlib import Path
import sys
import numpy as np
import torch
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def install_pydrake_fallback_if_needed():
    try:
        import pydrake.all  # noqa: F401
        return
    except Exception:
        pass

    import types

    class PiecewisePolynomial:
        def __init__(self, times, values):
            self.times = np.asarray(times, dtype=np.float64)
            self.values = np.asarray(values, dtype=np.float64)
            if self.values.ndim != 2:
                raise ValueError("PiecewisePolynomial values must be 2-D")
            if self.values.shape[1] != self.times.shape[0]:
                raise ValueError("PiecewisePolynomial expects values shaped (dim, num_times)")

        @classmethod
        def FirstOrderHold(cls, times, values):
            return cls(times, values)

        def value(self, t):
            t_float = float(np.asarray(t))
            out = [np.interp(t_float, self.times, self.values[dim]) for dim in range(self.values.shape[0])]
            return np.asarray(out, dtype=np.float64)[:, None]

        def EvalDerivative(self, t):
            t_float = float(np.asarray(t))
            idx = int(np.searchsorted(self.times, t_float, side="right") - 1)
            idx = max(0, min(idx, len(self.times) - 2))
            dt = max(float(self.times[idx + 1] - self.times[idx]), 1e-12)
            deriv = (self.values[:, idx + 1] - self.values[:, idx]) / dt
            return np.asarray(deriv, dtype=np.float64)[:, None]

    pydrake_module = types.ModuleType("pydrake")
    pydrake_all_module = types.ModuleType("pydrake.all")
    pydrake_all_module.PiecewisePolynomial = PiecewisePolynomial
    pydrake_module.all = pydrake_all_module
    sys.modules.setdefault("pydrake", pydrake_module)
    sys.modules.setdefault("pydrake.all", pydrake_all_module)


def install_legacy_policy_fallback_if_needed():
    import types

    module_name = "streaming_flow_policy.pusht.dp_state_notebook.base_policy"
    if module_name in sys.modules:
        return

    try:
        from streaming_flow_policy.pusht.dp_state_notebook.base_policy import Policy  # noqa: F401
        return
    except Exception:
        pass

    class Policy(torch.nn.Module):
        pass

    streaming_flow_policy_module = types.ModuleType("streaming_flow_policy")
    pusht_module = types.ModuleType("streaming_flow_policy.pusht")
    dp_state_module = types.ModuleType("streaming_flow_policy.pusht.dp_state_notebook")
    base_policy_module = types.ModuleType(module_name)
    base_policy_module.Policy = Policy

    sys.modules.setdefault("streaming_flow_policy", streaming_flow_policy_module)
    sys.modules.setdefault("streaming_flow_policy.pusht", pusht_module)
    sys.modules.setdefault("streaming_flow_policy.pusht.dp_state_notebook", dp_state_module)
    sys.modules.setdefault(module_name, base_policy_module)


install_pydrake_fallback_if_needed()
install_legacy_policy_fallback_if_needed()

from casf.streaming_flow_policy.casf_lasa.dataset import MatSequenceDatasetTask, MatSequenceDatasetWithNextObsAsAction_Task
from casf.network import ConditionalUnet1D
from casf.streaming_flow_policy.casf_lasa.sfpd import StreamingFlowPolicyDeterministic

import argparse
import os

# --------------------------
# Argument Parser
# --------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train Streaming Flow Policy on LASA dataset")

    # core model & training parameters
    parser.add_argument("--pred_horizon", type=int, default=16, help="Prediction horizon length")
    parser.add_argument("--obs_horizon", type=int, default=2, help="Number of observation steps")
    parser.add_argument("--action_horizon", type=int, default=8, help="Number of actions predicted per step")
    parser.add_argument("--obs_dim", type=int, default=4, help="Observation dimension")
    parser.add_argument("--action_dim", type=int, default=2, help="Action dimension")
    parser.add_argument("--sigma", type=float, default=0.2, help="Noise level for x during training")

    # training schedule
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val_every", type=int, default=200, help="Validate every N epochs")

    # dataset/task
    parser.add_argument("--taskName", type=str, default="Sshape", help="Task name from LASA dataset")
    parser.add_argument("--train_demos", type=int, default=6, help="Number of training demos")
    parser.add_argument("--test_demos", type=int, default=0, help="Number of test demos")
    parser.add_argument("--dataset_path", type=str, default="/home/droplab/Monica/robotics_policy/sfp_monica/external/lasa/DataSet", # "/home/jieting/Projects/casf/sfp_monica/external/lasa/DataSet", # 
                        help="Path to LASA dataset folder")

    # evaluation control
    parser.add_argument("--eval", type=bool, default=False, help="Run evaluation during training")

    # experiment naming / save paths
    parser.add_argument("--exp_prefix", type=str, default="lasa_1task", help="Prefix for experiment naming")
    parser.add_argument("--models_dir", type=str, default="./models", help="Model save directory")
    parser.add_argument("--results_dir", type=str, default="./results_lasa", help="Validation results save directory")

    args = parser.parse_args()
    return args

"""
|o|o|                             observations: 2
| |a|a|a|a|a|a|a|a|               actions executed: 8
|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
"""

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

# =============================================================================
# Parameters
# =============================================================================

args = get_args()
dataset_path = Path(args.dataset_path).expanduser()
if dataset_path.suffix == ".mat":
    if args.taskName != dataset_path.stem:
        raise ValueError(
            f"--dataset_path points to {dataset_path.name}, but --taskName is {args.taskName}. "
            "Use the matching task name or pass the parent LASA directory."
        )
    dataset_path = dataset_path.parent
args.dataset_path = str(dataset_path.resolve())
args.models_dir = str(Path(args.models_dir).expanduser().resolve())
os.makedirs(args.models_dir, exist_ok=True)

print("========== CONFIGURATION ==========")
for k, v in vars(args).items():
    print(f"{k:20s}: {v}")
print("===================================")

pred_horizon = args.pred_horizon
obs_horizon = args.obs_horizon
action_horizon = args.action_horizon
obs_dim = args.obs_dim
action_dim = args.action_dim
sigma = args.sigma
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
val_every = args.val_every
taskName = args.taskName
EVAL = args.eval

expName = f"CASF_{args.exp_prefix}{taskName}_sfpdObs_{num_epochs}ep_lr{learning_rate}_obsDim{obs_dim}_demo{args.train_demos}-{args.test_demos}_norm"
save_path = os.path.join(args.models_dir, f"{expName}.pth")
# save_path_bestmse = os.path.join(args.models_dir, f"{expName}_BESTmse.pth")
# =============================================================================

# create network object
velocity_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon,
    fc_timesteps=1,
)

# device transfer
device = torch.device('cuda')
_ = velocity_net.to(device)

policy = StreamingFlowPolicyDeterministic(
    velocity_net=velocity_net,
    action_dim=action_dim,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    sigma=sigma,
    device=device,
)

# create dataset from file 
train_ds = MatSequenceDatasetWithNextObsAsAction_Task(args.dataset_path, 
                                                 pred_horizon=policy.pred_horizon.item(), 
                                                 obs_horizon=obs_horizon, action_horizon=action_horizon, obs_dim=obs_dim,
                                                 task=taskName,
                                                 split='train', split_config={"train_demos": args.train_demos, "test_demos": args.test_demos}, 
                                                 transform_datum_fn=policy.TransformTrainingDatum)
if args.test_demos > 0:
    val_ds = MatSequenceDatasetWithNextObsAsAction_Task(args.dataset_path, 
                                                pred_horizon=policy.pred_horizon.item(),
                                                obs_horizon=obs_horizon, action_horizon=action_horizon, obs_dim=obs_dim,
                                                task=taskName,
                                                split='test', split_config={"train_demos": args.train_demos, "test_demos": args.test_demos},
                                                stats=train_ds.stats,   # <-- reuse stats #NOTE
                                                demos_point_norm = train_ds.demos_point_norm
                                                )

stats = train_ds.stats
# create dataloader
train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=batch_size,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process after each epoch
    persistent_workers=True
)
if args.test_demos > 0:
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True
    )

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(parameters=velocity_net.parameters(), power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=velocity_net.parameters(),
    lr=learning_rate, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_loader) * num_epochs
)

val_every = num_epochs # run validation every N epochs
with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(train_loader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                nbatch = dict_apply(nbatch, lambda x: x.to(device, non_blocking=True))
                # L2 loss
                loss = policy.Loss(nbatch)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(velocity_net.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))
        # ========= eval for this epoch ==========
        if EVAL:
            print('NOTE: EVAL=True')
            policy.eval()
            val_result_vis_path = f'./R_casfLASA/{expName}'
            val_result_vis_file_prefix = f'valFullTrajectory_visual_{epoch_idx}ep'
            # run validation
            if (epoch_idx+1) % val_every == 0:
                print("Running validation...")
                with torch.no_grad():
                    from casf.streaming_flow_policy.casf_lasa.rollout import rollout_lasa

                    results = rollout_lasa(policy, val_ds, stats=stats, device="cuda", 
                                           action_horizon=action_horizon, obs_horizon=obs_horizon,
                                           save_path=val_result_vis_path, save_fileName_prefix=val_result_vis_file_prefix)
                    
                    val_mse = results["MSE"]
                    val_mseNorm = results["MSE_NORM"]
                    val_FD = results["FinalDist"]
                    val_FDNorm = results["FinalDist_NORM"]
                    
                    print(f"Validation MSE at {epoch_idx}: {val_mse}")
                    print(f"Validation MSE_Norm at {epoch_idx}: {val_mseNorm}")
                    print(f"Validation distance at {epoch_idx}: {val_FD}")
                    print(f"Validation distance_norm at {epoch_idx}: {val_FDNorm}")
                print('--'*10,f'finishing validation at ep{epoch_idx}','--'*10)
        # ========= eval end for this epoch ==========
        policy.train()
                
# Weights of the EMA model
# is used for inference
ema_velocity_net = policy.velocity_net
ema.copy_to(ema_velocity_net.parameters())

# Save model
torch.save(policy.state_dict(), save_path)
print(f"Saved model to {save_path}.")
