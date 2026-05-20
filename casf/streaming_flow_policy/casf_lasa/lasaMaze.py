import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation, binary_fill_holes
import scipy.io as sio
import os
from typing import Dict, Optional, Callable
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree


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
#####

def build_s_corridor_union00(demos, w=0.05, grid_res=200, margin=0.1):
    """
    demos: list of arrays (N_i x 2)
    w: corridor half-width (like wall thickness)
    grid_res: resolution of grid for visualization
    margin: padding around the demo space
    """
    # 1. Collect all demo points
    all_points = np.concatenate(demos, axis=0)
    xmin, ymin = all_points.min(0) - margin
    xmax, ymax = all_points.max(0) + margin

    # 2. Create grid
    x = np.linspace(xmin, xmax, grid_res)
    y = np.linspace(ymin, ymax, grid_res)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)

    # 3. Create binary mask of where demos are
    mask = np.zeros(X.shape, dtype=bool)
    for d in demos:
        # nearest grid indices for each trajectory point
        xi = np.searchsorted(x, d[:, 0])
        yi = np.searchsorted(y, d[:, 1])
        xi = np.clip(xi, 0, grid_res - 1)
        yi = np.clip(yi, 0, grid_res - 1)
        mask[yi, xi] = True

    # 4. Compute distance transform from the demo region
    dist_out = distance_transform_edt(~mask) * (x[1] - x[0])  # convert to same scale
    sdf = dist_out - w   # inside if sdf < 0 (within w of any demo)
    
    inside = sdf < 0   # Boolean mask: True = inside corridor
    corridor_pixels = np.stack([X[inside], Y[inside]], axis=-1)

    return X, Y, sdf, mask, corridor_pixels

def build_s_corridor_union(
    demos, grid_res=200, margin=0.1,
    smooth_sigma=1.0, density_thresh=0.0,
    inflate=0.0
):
    """
    Build one connected S corridor and shrink/expand it evenly using morphological operations.
    """
    # 1. Collect points and grid
    all_points = np.concatenate(demos, axis=0)
    xmin, ymin = all_points.min(0) - margin
    xmax, ymax = all_points.max(0) + margin
    x = np.linspace(xmin, xmax, grid_res)
    y = np.linspace(ymin, ymax, grid_res)
    X, Y = np.meshgrid(x, y)
    pixel_size = x[1] - x[0]

    # 2. Build smoothed density map
    density = np.zeros(X.shape, dtype=np.float32)
    for d in demos:
        xi = np.searchsorted(x, d[:, 0])
        yi = np.searchsorted(y, d[:, 1])
        xi = np.clip(xi, 0, grid_res - 1)
        yi = np.clip(yi, 0, grid_res - 1)
        density[yi, xi] += 1
    from scipy.ndimage import gaussian_filter
    density = gaussian_filter(density, sigma=smooth_sigma)

    # 3. Threshold to get a single region
    region = density > density_thresh * density.max()
    region = binary_fill_holes(region)

    # 4. Morphological inflation (erosion/dilation in pixel space)
    
    corridor_halfwidth = np.mean(distance_transform_edt(region)) * pixel_size
    n_pixels = max(int(abs(inflate) / corridor_halfwidth * 10), 1)
    if inflate > 0:
        region = binary_erosion(region, iterations=n_pixels)
    elif inflate < 0:
        region = binary_dilation(region, iterations=n_pixels)

    # 5. Compute signed distance field after reshaping
    sdf_out = distance_transform_edt(~region) * pixel_size
    sdf_in = distance_transform_edt(region) * pixel_size
    sdf = sdf_out - sdf_in

    inside = sdf < 0
    corridor_pixels = np.stack([X[inside], Y[inside]], axis=-1)
    return X, Y, sdf, region, corridor_pixels

def build_even_s_corridor(demos, width=0.05, grid_res=300, margin=0.1, smoothness=0.5):
    """
    Build an evenly-thick, sharp S corridor by fitting a unified spline centerline.

    Args:
        demos: list of arrays (N_i, 2)  -- 7 demo trajectories
        width: corridor half-width (in LASA coordinate scale)
        grid_res: grid resolution for SDF
        margin: padding around demo space
        smoothness: smoothing factor for spline fit (0 = interpolate all points)
    Returns:
        X, Y, sdf, centerline, corridor_pixels
    """

    # 1️⃣ Gather all points
    all_points = np.concatenate(demos, axis=0)

    # 2️⃣ Sample each demo evenly along arc length
    resampled = []
    for d in demos:
        t = np.linspace(0, 1, len(d))
        t_new = np.linspace(0, 1, 200)
        x_spline, _ = splprep(d.T, s=smoothness)
        x_eval = np.stack(splev(t_new, x_spline), axis=1)
        resampled.append(x_eval)
    all_resampled = np.concatenate(resampled, axis=0)

    # 3️⃣ Fit a smooth spline through the mean curve (approx centerline)
    # average by phase along S
    mean_traj = np.mean(np.stack(resampled, axis=2), axis=2)
    tck, _ = splprep(mean_traj.T, s=smoothness)
    u = np.linspace(0, 1, 600)
    centerline = np.stack(splev(u, tck), axis=1)

    # 4️⃣ Create workspace grid
    xmin, ymin = all_points.min(0) - margin
    xmax, ymax = all_points.max(0) + margin
    x = np.linspace(xmin, xmax, grid_res)
    y = np.linspace(ymin, ymax, grid_res)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)

    # 5️⃣ Compute signed distance to centerline (KDTree for efficiency)
    tree = KDTree(centerline)
    dist, _ = tree.query(XY)
    sdf = dist.reshape(X.shape) - width     # negative inside

    # --- 6️⃣[optional] Flat start/end caps ---
    v_start = centerline[5] - centerline[0]
    v_start /= np.linalg.norm(v_start) + 1e-8
    v_end = centerline[-1] - centerline[-6]
    v_end /= np.linalg.norm(v_end) + 1e-8

    start_plane = np.dot(XY - centerline[0], v_start)
    end_plane   = np.dot(XY - centerline[-1], v_end)

    outside_start = start_plane < -width
    outside_end   = end_plane >  width
    cap_mask = ~(outside_start | outside_end)

    sdf[~cap_mask.reshape(X.shape)] = np.maximum(sdf[~cap_mask.reshape(X.shape)], 0)
    # --- 6️⃣[optional]  Flat start/end caps ---

    inside = sdf < 0
    corridor_pixels = np.stack([X[inside], Y[inside]], axis=-1)

    return X, Y, sdf, centerline, corridor_pixels

if __name__ == "__main__":
    dataset_dir = "/home/droplab/Monica/robotics_policy/sfp_monica/external/lasa/DataSet"
    task = 'Sshape'
    task_mat_path = os.path.join(dataset_dir, task + ".mat")
    print(f"Loading demos from: {task_mat_path}")
    data = sio.loadmat(task_mat_path, squeeze_me=True, struct_as_record=False)
    demos = np.ravel(data["demos"])
    demos_point = []
    for demo in demos:
        pos = demo.pos.T.astype(np.float32)
        demos_point.append(np.asarray(pos))
    
    demos_point_array = np.concatenate(demos_point, axis=0)
    pos_stats = get_data_stats(demos_point_array)
    demos_point_norm = []
    for dePoint in demos_point:
        dePoint_norm = normalize_data(dePoint, pos_stats)
        demos_point_norm.append(dePoint_norm)
    # X, Y, sdf, region, corridor_pixels = build_s_corridor_union(demos_point_norm, inflate=0.02,smooth_sigma=1.0)
    X  , Y, sdf, centerline, corridor_pixels = build_even_s_corridor(demos_point_norm, width=0.2)


    plt.figure(figsize=(6,6))
    plt.contourf(X, Y, sdf, 
                    levels=[-1e9, 0, 1e9],   # below 0 = inside corridor, above 0 = outside
                    colors=["#F5D48F", "white"],   # inside (yellow), outside (white)
                    alpha=1)
    # plt.imshow(mask, origin='lower',
    #         extent=[X.min(), X.max(), Y.min(), Y.max()],
    #         cmap='viridis', alpha=1)
    # for d in demos_point_norm:
    #     plt.plot(d[:,0], d[:,1], 'g', lw=1.2)
    # plt.scatter(corridor_pixels[:,0], corridor_pixels[:,1], s=5, color='gold', alpha=0.7, label='Inside corridor')
    # plt.plot(centerline[:,0], centerline[:,1], 'k--', lw=2, label="Centerline")
    for d in demos_point_norm:
        plt.plot(d[:,0], d[:,1], 'g-', lw=0.8)
    plt.plot(centerline[:,0], centerline[:,1], 'r--', lw=1.2, label="Centerline")
    plt.title("Evenly thick, sharp S corridor")

    plt.legend()
    plt.axis('equal')
    plt.title("S-shaped corridor")
    plt.savefig("./lasaMaze_check.png", dpi=150)
    print('image saved to -- ./lasaMaze_check.png')
