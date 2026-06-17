"""
Task 1: Bundle Adjustment from Scratch using PyTorch
=====================================================
Recover 3D points, camera poses (R, T), and focal length from 2D observations.

Key design:
- Focal length `f` is shared across all 50 cameras (single scalar parameter).
- Each camera has 6-DOF extrinsics: 3 Euler angles for rotation, 3 for translation.
- All 20000 3D points are optimized directly.
- Only visible points contribute to the reprojection loss.

GPU-accelerated with PyTorch + CUDA.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time


# ==============================================================================
#  Geometry utilities
# ==============================================================================

def euler_angles_to_matrix(euler_angles, convention="XYZ"):
    """
    Convert Euler angles to rotation matrices.

    This replicates pytorch3d.transforms.euler_angles_to_matrix so we avoid
    the pytorch3d dependency on Windows.

    Args:
        euler_angles: (*, 3) tensor of angles in radians.
        convention: rotation order (only "XYZ" supported here).

    Returns:
        R: (*, 3, 3) rotation matrix.
    """
    if convention != "XYZ":
        raise NotImplementedError("Only XYZ convention is implemented.")

    # Separate angles
    rx = euler_angles[..., 0]  # rotation about X
    ry = euler_angles[..., 1]  # rotation about Y
    rz = euler_angles[..., 2]  # rotation about Z

    cos_rx, sin_rx = torch.cos(rx), torch.sin(rx)
    cos_ry, sin_ry = torch.cos(ry), torch.sin(ry)
    cos_rz, sin_rz = torch.cos(rz), torch.sin(rz)

    # Rx
    zeros = torch.zeros_like(rx)
    ones = torch.ones_like(rx)
    Rx = torch.stack([
        torch.stack([ ones,     zeros,      zeros     ], dim=-1),
        torch.stack([ zeros,    cos_rx,    -sin_rx    ], dim=-1),
        torch.stack([ zeros,    sin_rx,     cos_rx    ], dim=-1),
    ], dim=-2)  # (*, 3, 3)

    # Ry
    Ry = torch.stack([
        torch.stack([ cos_ry,   zeros,      sin_ry    ], dim=-1),
        torch.stack([ zeros,    ones,       zeros     ], dim=-1),
        torch.stack([-sin_ry,   zeros,      cos_ry    ], dim=-1),
    ], dim=-2)

    # Rz
    Rz = torch.stack([
        torch.stack([ cos_rz,  -sin_rz,     zeros     ], dim=-1),
        torch.stack([ sin_rz,   cos_rz,     zeros     ], dim=-1),
        torch.stack([ zeros,    zeros,      ones      ], dim=-1),
    ], dim=-2)

    # R = Rz @ Ry @ Rx  (extrinsic rotation)
    return Rz @ Ry @ Rx


def project_points(points_3d, R, T, f, cx, cy):
    """
    Project 3D points to 2D pixel coordinates.

    Args:
        points_3d: (N, 3) — 3D world coordinates.
        R:         (V, 3, 3) — per-view rotation matrices.
        T:         (V, 3) — per-view translation vectors.
        f:         scalar — focal length (shared).
        cx, cy:    float — principal point (image center).

    Returns:
        u, v: (V, N) — projected pixel coordinates.
    """
    # Transform to camera coordinates: Xc = R @ P + T
    # points_3d: (N, 3), R: (V, 3, 3), T: (V, 3)
    Xc = (R @ points_3d.T) + T.unsqueeze(-1)  # (V, 3, N)

    Xc_x = Xc[:, 0, :]  # (V, N)
    Xc_y = Xc[:, 1, :]
    Xc_z = Xc[:, 2, :]  # (V, N)

    # Project: u = -f * Xc / Zc + cx,  v = f * Yc / Zc + cy
    u = -f * Xc_x / Xc_z + cx
    v =  f * Xc_y / Xc_z + cy

    return u, v


# ==============================================================================
#  Bundle Adjustment
# ==============================================================================

def run_bundle_adjustment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ------------------------------------------------------------------
    #  1. Load data
    # ------------------------------------------------------------------
    data_dir = Path("data")
    points2d_data = np.load(data_dir / "points2d.npz")
    colors = np.load(data_dir / "points3d_colors.npy")  # (20000, 3)

    NUM_VIEWS = 50
    NUM_POINTS = 20000
    IMG_W, IMG_H = 1024, 1024
    cx, cy = IMG_W / 2.0, IMG_H / 2.0  # 512, 512

    # Build observation tensors: shape (V, N, 3) -> [x, y, visibility]
    obs_list = [points2d_data[f"view_{v:03d}"] for v in range(NUM_VIEWS)]
    observations = torch.tensor(np.stack(obs_list, axis=0), dtype=torch.float32, device=device)
    # Extract x, y, visibility
    obs_xy = observations[..., :2]    # (V, N, 2)
    obs_vis = observations[..., 2]    # (V, N)  — 1.0 = visible, 0.0 = occluded
    vis_mask = obs_vis > 0.5          # boolean mask

    print(f"[INFO] Loaded {NUM_VIEWS} views, {NUM_POINTS} points")
    print(f"[INFO] Visible observations: {vis_mask.sum().item()} / {NUM_VIEWS * NUM_POINTS}")

    # ------------------------------------------------------------------
    #  2. Initialize optimizable parameters
    # ------------------------------------------------------------------
    # Focal length: initialize assuming ~60° FoV  (f = H / (2*tan(fov/2)))
    # For 1024 px height and 60° FoV: f ≈ 1024 / (2 * tan(30°)) ≈ 887
    f_init = IMG_H / (2 * np.tan(np.deg2rad(60.0 / 2)))
    f = nn.Parameter(torch.tensor(f_init, dtype=torch.float32, device=device))
    print(f"[INFO] Initial focal length: {f.item():.1f}")

    # Euler angles: initialize near zero (cameras face +Z direction)
    euler_angles = nn.Parameter(torch.zeros(NUM_VIEWS, 3, dtype=torch.float32, device=device))

    # Translation: initialize at [0, 0, -d] with d in [2, 3]
    # Using small random perturbation for variety
    torch.manual_seed(42)
    d_init = 2.5
    T = nn.Parameter(torch.zeros(NUM_VIEWS, 3, dtype=torch.float32, device=device))
    with torch.no_grad():
        T[:, 0] = torch.randn(NUM_VIEWS, device=device) * 0.1  # small x offset
        T[:, 1] = torch.randn(NUM_VIEWS, device=device) * 0.1  # small y offset
        T[:, 2] = -d_init + torch.randn(NUM_VIEWS, device=device) * 0.1

    # 3D points: initialize near origin with small random spread
    points_3d = nn.Parameter(
        torch.randn(NUM_POINTS, 3, dtype=torch.float32, device=device) * 0.5
    )

    num_params = 1 + NUM_VIEWS * 3 + NUM_VIEWS * 3 + NUM_POINTS * 3
    print(f"[INFO] Total parameters: {num_params}")

    # ------------------------------------------------------------------
    #  3. Optimizer
    # ------------------------------------------------------------------
    # Use different learning rates for different parameter groups
    optimizer = torch.optim.Adam([
        {"params": [f],               "lr": 10.0},
        {"params": [euler_angles],    "lr": 0.005},
        {"params": [T],               "lr": 0.01},
        {"params": [points_3d],       "lr": 0.01},
    ])

    # Learning rate scheduler: reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, verbose=True
    )

    # ------------------------------------------------------------------
    #  4. Optimization loop
    # ------------------------------------------------------------------
    NUM_ITER = 2000
    loss_history = []

    print(f"\n[INFO] Starting optimization ({NUM_ITER} iterations)...")
    t_start = time.time()

    for iteration in range(NUM_ITER):
        optimizer.zero_grad()

        # Build rotation matrices from Euler angles
        R = euler_angles_to_matrix(euler_angles, convention="XYZ")  # (V, 3, 3)

        # Project 3D points to 2D
        u_pred, v_pred = project_points(points_3d, R, T, f, cx, cy)

        # Stack predictions: (V, N, 2)
        pred_xy = torch.stack([u_pred, v_pred], dim=-1)

        # Reprojection error: ||pred - obs||
        diff = pred_xy - obs_xy  # (V, N, 2)
        errors = torch.norm(diff, dim=-1)  # (V, N)

        # Only count visible points
        masked_errors = errors * vis_mask.float()

        # Mean loss over visible observations
        loss = masked_errors.sum() / vis_mask.sum().float()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([f, euler_angles, T, points_3d], max_norm=10.0)
        optimizer.step()
        scheduler.step(loss)

        loss_history.append(loss.item())

        if (iteration + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  Iter {iteration+1:5d}/{NUM_ITER} | "
                  f"Loss: {loss.item():.6f} | "
                  f"f: {f.item():.1f} | "
                  f"Time: {elapsed:.1f}s")

    t_total = time.time() - t_start
    print(f"\n[INFO] Optimization finished in {t_total:.1f}s")
    print(f"[INFO] Final loss: {loss_history[-1]:.6f}")
    print(f"[INFO] Optimized focal length: {f.item():.2f}")

    # ------------------------------------------------------------------
    #  5. Plot loss curve
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(loss_history, linewidth=0.5, color="steelblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reprojection Error (pixels)")
    ax.set_title("Bundle Adjustment — Loss Curve")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("loss_curve.png", dpi=150)
    print("[INFO] Loss curve saved to loss_curve.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    #  6. Save reconstructed 3D point cloud as colored OBJ
    # ------------------------------------------------------------------
    points_np = points_3d.detach().cpu().numpy()  # (N, 3)
    colors_np = colors  # (N, 3), already in [0, 1]

    with open("reconstructed.ply", "w") as fo:
        fo.write("ply\n")
        fo.write("format ascii 1.0\n")
        fo.write(f"element vertex {NUM_POINTS}\n")
        fo.write("property float x\n")
        fo.write("property float y\n")
        fo.write("property float z\n")
        fo.write("property uchar red\n")
        fo.write("property uchar green\n")
        fo.write("property uchar blue\n")
        fo.write("end_header\n")
        for i in range(NUM_POINTS):
            x, y, z = points_np[i]
            r, g, b = (np.clip(colors_np[i], 0, 1) * 255).astype(np.uint8)
            fo.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    # Also save as OBJ (with colors as vertex comments, since standard OBJ
    # doesn't directly support per-vertex colors in a universal way)
    with open("reconstructed.obj", "w") as fo:
        fo.write("# Reconstructed 3D point cloud from Bundle Adjustment\n")
        fo.write(f"# {NUM_POINTS} vertices\n")
        for i in range(NUM_POINTS):
            x, y, z = points_np[i]
            r, g, b = colors_np[i]
            fo.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

    print("[INFO] Point cloud saved to reconstructed.ply and reconstructed.obj")

    # ------------------------------------------------------------------
    #  7. Save optimized camera parameters
    # ------------------------------------------------------------------
    np.savez(
        "optimized_cameras.npz",
        f=f.detach().cpu().numpy(),
        euler_angles=euler_angles.detach().cpu().numpy(),
        T=T.detach().cpu().numpy(),
        R=euler_angles_to_matrix(euler_angles, convention="XYZ").detach().cpu().numpy(),
    )
    print("[INFO] Camera parameters saved to optimized_cameras.npz")

    print("\n[INFO] Task 1 complete!")
    return loss_history, f.item()


if __name__ == "__main__":
    run_bundle_adjustment()
