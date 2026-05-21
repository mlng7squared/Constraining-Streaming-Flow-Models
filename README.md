# Constraining Streaming Flow Models

Compact reproduction for CASF on the LASA `Sshape` task. The checkpoint is
trained with demos `0..5`; evaluation in `notebooks/casf_lasa.ipynb` runs SFP
and CASF on the held-out test demo `6`, reports `MaskedFD`, `MPD`, and `IntV`,
and saves the rollout plot.

## Environment

Use a conda environment for the LASA scripts and notebook:

```bash
cd Constraining-Streaming-Flow-Models
conda create -n casf-lasa python=3.10 -y
conda activate casf-lasa
pip install -r requirements.txt
python -m ipykernel install --user --name casf-lasa --display-name "Python (casf-lasa)"
```

## Data

The default LASA file is local to this repo:

```text
data/lasa/Sshape.mat
```

## Train SFP Checkpoint

Train on the first six LASA demos and reserve one demo for test:

```bash
python scripts/casf_sfpdObs_train.py \
  --taskName Sshape \
  --action_horizon 8 \
  --epochs 1000 \
  --lr 1e-4 \
  --batch_size 1024 \
  --obs_dim 4 \
  --obs_horizon 2 \
  --pred_horizon 16 \
  --sigma 0.1 \
  --dataset_path data/lasa/Sshape.mat \
  --exp_prefix "lasaTask_ah8_" \
  --train_demos 6 \
  --test_demos 1
```

The expected checkpoint name is:

```text
models/CASF_lasaTask_ah8_Sshape_sfpdObs_1000ep_lr0.0001_obsDim4_demo6-1_norm.pth
```

## Evaluate SFP vs CASF

Open `notebooks/casf_lasa.ipynb` with the `casf-lasa` kernel and run all cells.
The notebook evaluates only the held-out test demo and show quantitative and qualitative results comparing SFP and CASF.