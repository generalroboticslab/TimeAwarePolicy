"""Matplotlib helpers for visualizing task observations and policy actions."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from core.common.tf_numpy import tf_combine

# ---------- Math helpers ----------
def quat_to_rot(q):
    """Quaternion [x,y,z,w] -> 3x3 rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [    2*(xy + wz),  1 - 2*(xx + zz),        2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ])
    return R

def make_box_vertices(center, R, size_xyz):
    """
    Create 8 vertices of a box centered at 'center' with dims size_xyz = [sx, sy, sz]
    oriented by rotation R (3x3). Returns (8,3) array.
    """
    sx, sy, sz = size_xyz
    corners_local = np.array([
        [-sx/2, -sy/2, -sz/2], [ sx/2, -sy/2, -sz/2], [ sx/2,  sy/2, -sz/2], [-sx/2,  sy/2, -sz/2],
        [-sx/2, -sy/2,  sz/2], [ sx/2, -sy/2,  sz/2], [ sx/2,  sy/2,  sz/2], [-sx/2,  sy/2,  sz/2],
    ])
    return (R @ corners_local.T).T + center

def draw_box(ax, center, quat, size, facecolor=(0.6, 0.6, 0.9, 0.3), edgecolor=None, linewidth=1.0, isotropic=True):
    """Draw an oriented cube/box. size can be scalar (cube) or (sx,sy,sz)."""
    R = quat_to_rot(quat)
    size_xyz = (np.array([size, size, size]) if np.isscalar(size) else np.asarray(size))
    V = make_box_vertices(center, R, size_xyz)
    faces = [
        [V[0], V[1], V[2], V[3]],  # bottom
        [V[4], V[5], V[6], V[7]],  # top
        [V[0], V[1], V[5], V[4]],  # side
        [V[2], V[3], V[7], V[6]],  # side
        [V[1], V[2], V[6], V[5]],  # side
        [V[4], V[7], V[3], V[0]],  # side
    ]
    poly = Poly3DCollection(faces, facecolors=facecolor, edgecolors=edgecolor, linewidths=linewidth)
    ax.add_collection3d(poly)

def draw_frame(ax, origin, R, length=0.05, lw=2.0, alpha=0.9):
    """Draw a small triad frame at origin with rotation R."""
    x_axis = origin + length * R[:, 0]
    y_axis = origin + length * R[:, 1]
    z_axis = origin + length * R[:, 2]
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r', lw=lw, alpha=alpha)
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g', lw=lw, alpha=alpha)
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b', lw=lw, alpha=alpha)

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    mid_x = np.mean(x_limits); mid_y = np.mean(y_limits)
    ax.set_xlim3d([mid_x - max_range/2, mid_x + max_range/2])
    ax.set_ylim3d([mid_y - max_range/2, mid_y + max_range/2])
    ax.set_zlim3d([0., max_range])

    # Critical line: enforce equal visual aspect
    ax.set_box_aspect((1, 1, 1))


def misc_axes_settings(ax):
    ax.grid(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.line.set_color((1,1,1,0))
    ax.yaxis.line.set_color((1,1,1,0))
    ax.zaxis.line.set_color((1,1,1,0))


# ---------- Gripper ----------
def draw_parallel_gripper(ax,
                          eef_pos,
                          eef_quat,
                          jaw_width=0.08, finger_len=0.04, finger_thick=0.01,
                          bridge_thick=0.01, bridge_offset=0.04,
                          palm_thick=0.012, palm_height=0.05, palm_back_offset=0.03,
                          color='gray'):
    """
    Draw a parallel gripper with:
      - two fingers along +Z (forward)
      - a horizontal bridge connecting fingers (along X and Y thickness)
      - a vertical back link ("palm") behind the bridge along +Y (or along Z back)
    Local axes: X (left-right), Y (up-down), Z (forward)
    center: EEF origin in world frame
    quat: orientation [x,y,z,w]
    """
    left_figner_size = np.array([0.01, 0.01, 0.04])
    right_figner_size = np.array([0.01, 0.01, 0.04])
    bridge_size = np.array([0.01, 0.09, 0.01])
    wrist_size = np.array([0.01, 0.01, 0.04])
    center2left_finger = np.array([0, 0.04, -0.02])
    center2right_finger = np.array([0, -0.04, -0.02])
    center2bridge = np.array([0, 0.0, -bridge_offset])
    center2wrist = np.array([0, 0, -bridge_offset-0.02])
    uni_quat = np.array([0, 0, 0, 1.])

    robot2left_finger = tf_combine(eef_quat, eef_pos, uni_quat, center2left_finger)
    robot2right_finger = tf_combine(eef_quat, eef_pos, uni_quat, center2right_finger)
    robot2bridge = tf_combine(eef_quat, eef_pos, uni_quat, center2bridge)
    robot2wrist = tf_combine(eef_quat, eef_pos, uni_quat, center2wrist)
    draw_box(ax, robot2left_finger[1], robot2left_finger[0], left_figner_size, facecolor=color)
    draw_box(ax, robot2right_finger[1], robot2right_finger[0], right_figner_size, facecolor=color)
    draw_box(ax, robot2bridge[1], robot2bridge[0], bridge_size, facecolor=color)
    draw_box(ax, robot2wrist[1], robot2wrist[0], wrist_size, facecolor=color)

# ---------- Main visualizer ----------
def visualize_scene_3d(
    pure_obs,
    actions,
    perturb_obs,
    cubeA_size=0.05,
    cubeB_size=0.07,
    arrow_scale=0.2,
    show_frames=False,
    cmap_name="viridis",
    save_path=None,
    revert_y=False,
    fig=None,
    ax=None,
    cbar=None,
):
    """
    Draw:
      - Source cube A (5 cm)
      - Target cube B (7 cm) at cubeA + offset
      - Parallel gripper with bridge + vertical palm
      - Multiple action arrows from EEF colored by perturb_obs

    Inputs:
      pure_obs schema:
        cubeA_pos (7) + cubeA_to_B_pos (3) + eef_pose (7) + ...
        where each 7 = [qx, qy, qz, qw, px, py, pz]
      actions_xyz: (N, 3) array of action displacement vectors in world/base frame
      perturb_obs: (N,) array (can be in seconds or ratio). Mapped to colors.
      offset_in_local: if True, rotate cubeA_to_B by cubeA orientation before adding.
    """
    obs = np.asarray(pure_obs).reshape(-1)
    # Slices (adjust if your layout differs)
    cubeA_p = obs[0:3]
    cubeA_q = obs[3:7]
    cubeA_to_B = obs[7:10]
    eef_p = obs[10:13]
    eef_q = obs[13:17]
    actions_xyz = actions[:, :3]

    R_A = quat_to_rot(cubeA_q)

    cubeB_p = cubeA_p + cubeA_to_B
    cubeB_q = np.array([0, 0, 0, 1.])  # identity orientation for cubeB

    if fig is None or ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Draw cubes
    draw_box(ax, center=cubeA_p, quat=cubeA_q, size=cubeA_size, facecolor=(0.4,0.8,1.0,0.4))
    draw_box(ax, center=cubeB_p, quat=cubeB_q, size=cubeB_size, facecolor=(1.0,0.6,0.4,0.4))

    # Draw gripper with extra links
    draw_parallel_gripper(
        ax, eef_pos=eef_p, eef_quat=eef_q
    )

    if show_frames:
        draw_frame(ax, origin=eef_p, R=quat_to_rot(eef_q), length=0.05)
        draw_frame(ax, origin=cubeA_p, R=R_A, length=0.05)

    # Arrows colored by perturb_obs
    actions_xyz = np.asarray(actions_xyz).reshape(-1, 3)
    actions_xyz /= (np.linalg.norm(actions_xyz, axis=1).max() + 1e-9)
    perturb_obs = np.asarray(perturb_obs).reshape(-1)
    assert actions_xyz.shape[0] == perturb_obs.shape[0], "actions_xyz and perturb_obs must have same length"
    cmap = get_cmap(cmap_name)
    norm = Normalize(vmin=np.min(perturb_obs), vmax=np.max(perturb_obs))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Draw each arrow from EEF
    for vec, t_rem in zip(actions_xyz, perturb_obs):
        color = sm.to_rgba(t_rem)
        # Using ax.quiver: length scales the vector to given length; provide unit vector and length
        mag = np.linalg.norm(vec)
        if mag < 1e-9:
            continue
        ax.quiver(
            eef_p[0], eef_p[1], eef_p[2],
            vec[0], vec[1], vec[2],
            length=arrow_scale * mag,
            normalize=False,
            color=color,
            linewidth=2
        )

    # Colorbar for remaining time
    if cbar is not None:
        # update existing colorbar
        cbar.update_normal(sm)
    else:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=-0.02)
    cbar.set_ticks(perturb_obs)
    cbar.set_ticklabels([f"{t:.1f}" for t in perturb_obs])

    if revert_y:
        cbar.ax.invert_yaxis()

    # Aesthetics
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.view_init(elev=10, azim=10)

    # Autoscale around key objects and arrows
    end_pts = [eef_p + arrow_scale * v for v in actions_xyz]
    pts = np.vstack([cubeA_p, cubeB_p, eef_p] + end_pts)
    pad = 0.1
    min_xyz = pts.min(axis=0) - pad
    max_xyz = pts.max(axis=0) + pad
    ax.set_xlim(min_xyz[0], max_xyz[0])
    ax.set_ylim(min_xyz[1], max_xyz[1])
    ax.set_zlim(min_xyz[2], max_xyz[2])
    set_axes_equal(ax)
    misc_axes_settings(ax)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)

    ax.cla()
    return fig, ax, cbar


# ---------- Minimal test ----------
if __name__ == "__main__":
    # Dummy observation (length 43 as in your schema)
    obs = np.zeros(43)
    # cubeA pose [px, py, pz, x, y, z, w]
    obs[0:3] = [0.2, 0.0, 0.05]
    obs[3:7] = [0, 0, 0, 1]
    # cubeA_to_B position offset
    obs[7:10] = [0.12, 0.08, 0.02]
    # eef pose
    obs[10:13] = [0.05, -0.05, 0.08]
    obs[13:17] = [1, 0, 0, 0]

    # Example multiple actions and remaining times
    actions = np.array([
        [ 0.06,  0.02, -0.01],
        [ 0.04,  0.00,  0.03],
        [-0.03,  0.05,  0.01],
        [ 0.00, -0.04,  0.02],
    ])
    remaining = np.array([0.1, 0.4, 0.7, 1.0])  # could be seconds or ratio

    fig, ax = visualize_scene_3d(
        obs,
        actions=actions,
        perturb_obs=remaining,
        arrow_scale=0.1,
        show_frames=True,
    )
    plt.show()
