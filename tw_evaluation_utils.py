
import psutil
from collections import deque
import shutil
import argparse
from distutils.util import strtobool
from utils import *

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the Trained Model')
    parser.add_argument('--rendering', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument("--num_envs", type=int, default=10)
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default='train_res', required=False)
    parser.add_argument('--eval_dir', type=str, default='eval_res', required=False)
    parser.add_argument('--saving', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--index_episode', type=str, default='last')
    parser.add_argument('--eval_result', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--sim_device', type=str, default="cuda:0")
    parser.add_argument('--graphics_device_id', type=int, default=0)
    
    parser.add_argument('--random_policy', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--heuristic_policy', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--record_init_configs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_par_checkpoint', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--quiet', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--realtime', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument('--draw_time', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--draw_scevel_val', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--draw_pos', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--draw_vel', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--draw_acc', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--draw_torque', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--draw_scevel', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--scan_time', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--blender_record', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    
    # Evaluation task parameters
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--warmup_episodes', type=int, default=0)
    parser.add_argument('--target_episodes', type=int, default=20000)
    parser.add_argument('--target_success_eps', type=int, default=None)
    parser.add_argument('--target_record_eps', type=int, default=None)
    parser.add_argument('--save_threshold', type=int, default=10)
    parser.add_argument('--act_scale_eval', type=float, default=None)
    parser.add_argument('--goal_speed', type=float, default=None)
    parser.add_argument('--goal_ratio_range', type=json.loads, default=[], metavar='N')
    parser.add_argument('--goal_time', type=float, default=None)
    parser.add_argument('--episodeLength_eval', type=int, default=None)
    parser.add_argument('--budget_portion', type=json.loads, default=None, metavar='N')
    parser.add_argument('--speed_describe', type=json.loads, default=[], metavar='N')
    parser.add_argument('--scale_actions_eval', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--fixed_configs_eval', type=lambda x: bool(strtobool(x)), default=None, nargs='?', const=True)
    parser.add_argument('--global_configs_eval', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--update_configs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--par_configs_eval', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--away_dist_eval', type=float, default=None)
    parser.add_argument('--specific_idx_eval', type=int, default=None)
    parser.add_argument('--apply_noise_eval', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--init_curri_ratio', type=float, default=1.)
    parser.add_argument('--vis_configs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--keyboard_ctrl', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--simple_layout', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--strict_eval', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    
    # FrankaCubeStack specific
    parser.add_argument('--max_dist', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--apply_disturbances', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--disturbance_v', type=float, default=None)
    parser.add_argument('--disturbance_v_range', type=json.loads, default=[], metavar='N')
    parser.add_argument('--use_container', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--add_restitution', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    
    # FrankaGmPour specific
    parser.add_argument('--num_gms_eval', type=int, default=None)
    parser.add_argument('--num_gms_range', type=json.loads, default=[], metavar='N')
    
    # FrankaCabinet specific
    parser.add_argument('--friction_mul', type=float, default=1)
    parser.add_argument('--friction_mul_range', type=json.loads, default=[], metavar='N')
    parser.add_argument('--num_props_eval', type=int, default=None)
    
    # Baseline specific
    parser.add_argument('--interpolate_joints', type=int, default=1, nargs='?', const=True)
    
    # Real world specific
    parser.add_argument('--real_robot', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_sim_pure', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_fk_replay', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--debug_obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--debug_act', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--not_move', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_default_target', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_avg_t2e', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_avg_limvel', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_avg_speed', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_max_limvel', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--cam_ext_path', type=str, default='cal_results/franka2cam.txt')
    parser.add_argument('--supp_time', type=float, default=0.)
    parser.add_argument('--compensate_occlusion', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--demo_dir', type=str, default="/home/grl/Videos/Time_Aware_RL/Experiments/RealWorldDemo/Stability/DrawerOpening")
    parser.add_argument('--demo_name', type=str, default=None)
    
    args = parser.parse_args()
    _process_args(args)
    _setup_directories(args)
    
    return args


def _process_args(args):
    """Process and validate arguments."""
    if args.task_name is None:
        args.task_name = args.checkpoint.split('_')[3]
    
    # Load checkpoint configuration
    checkpoint_folder = os.path.join(args.result_dir, args.task_name, args.checkpoint)
    args.json_file_path = os.path.join(checkpoint_folder, 'config.json')
    args.checkpoint_path = os.path.join(checkpoint_folder, 'checkpoints', 'eps_' + args.index_episode)
    assert os.path.exists(args.checkpoint_path), f"Checkpoint path {args.checkpoint_path} does not exist"
    
    restored_eval_args = args.__dict__.copy()
    args_json = read_json(args.json_file_path)
    
    if args.use_par_checkpoint:
        assert args_json["checkpoint"] is not None
        par_checkpoint_folder = os.path.join(args.result_dir, args.task_name, args_json["checkpoint"])
        par_json_file_path = os.path.join(par_checkpoint_folder, 'config.json')
        restored_eval_args["checkpoint_path"] = os.path.join(par_checkpoint_folder, 'checkpoints', 'eps_' + args_json["index_episode"])
        restored_eval_args["final_name"] = args_json["final_name"]
        args_json = read_json(par_json_file_path)
    
    args.__dict__.update(args_json)
    args.__dict__.update(restored_eval_args)
    
    # Graphics device
    args.graphics_device_id = 2 if args.rendering else -1
    
    # Handle argument relations
    _validate_and_update_args(args, args_json)
    _build_evaluation_lists(args)
    _validate_constraints(args)
    _build_experiment_name(args)


def _validate_and_update_args(args, args_json):
    """Validate and update argument relationships."""
    if args.warmup_episodes is None:
        args.warmup_episodes = args.num_envs * 5
    
    if args.record_init_configs and not args.update_configs:
        if args.save_threshold is None or args.target_record_eps is None:
            raise Exception("Need to set save_threshold and target_record_eps to record initial configs")
        args.target_success_eps = args.target_record_eps * args.save_threshold
    
    if args.specific_idx_eval is not None:
        args.specific_idx = args.specific_idx_eval
        args.fixed_configs = True
    
    if args.fixed_configs_eval is not None:
        args.fixed_configs = args.fixed_configs_eval
    
    if args.global_configs_eval:
        args.global_configs = True
    
    if args.par_configs_eval or args.update_configs:
        args.par_configs = True
        if not get_args_attr(args, "global_configs", False):
            assert args.fixed_configs
            assert args_json["checkpoint"] is not None
            args.par_checkpoint = args_json["checkpoint"]
            args.par_index_episode = args_json["index_episode"]
        else:
            args.fixed_configs = True
    
    if args.episodeLength_eval is not None:
        args.episodeLength = args.episodeLength_eval
    if args.away_dist_eval is not None:
        args.away_dist = args.away_dist_eval
    if args.num_gms_eval is not None:
        args.num_gms = args.num_gms_eval
    if args.act_scale_eval is not None:
        args.act_scale = args.act_scale_eval
    if args.vis_configs:
        args.specific_idx = 0 if args.specific_idx is None else args.specific_idx
    if args.strict_eval:
        assert args.num_envs == args.target_success_eps
    
    if args.use_fk_replay:
        args.use_sim_pure = True
    if args.debug_obs:
        args.real_robot = True
    if args.real_robot:
        args.warmup_episodes = 1


def _build_evaluation_lists(args):
    """Build evaluation parameter lists."""
    # Goal speed list
    args.goal_speed_lst = [1]
    if len(args.goal_ratio_range) != 0:
        assert len(args.goal_ratio_range) == 3
        max_ratio = args.goal_ratio_range[1]
        args.goal_speed_lst = np.arange(*args.goal_ratio_range).tolist()
        args.goal_speed_lst += [max_ratio] if max_ratio not in args.goal_speed_lst else []
    args.goal_speed_lst = [args.goal_speed] if args.goal_speed is not None else args.goal_speed_lst
    
    # Disturbance velocity list
    args.disturbance_v_lst = [0]
    if args.apply_disturbances:
        assert args.disturbance_v is not None or len(args.disturbance_v_range) != 0
        if len(args.disturbance_v_range) != 0:
            assert len(args.disturbance_v_range) == 3
            max_disturbance_v = args.disturbance_v_range[1]
            args.disturbance_v_lst = np.arange(*args.disturbance_v_range).tolist()
            args.disturbance_v_lst += [max_disturbance_v] if max_disturbance_v not in args.disturbance_v_lst else []
    args.disturbance_v_lst = [args.disturbance_v] if args.disturbance_v is not None else args.disturbance_v_lst
    
    # Number of GMs list
    args.num_gms_lst = [args.num_gms] if args.num_gms_eval is not None else [args.num_gms_eval]
    if len(args.num_gms_range) != 0:
        assert len(args.num_gms_range) == 3
        max_num_gms = args.num_gms_range[1]
        args.num_gms_lst = np.arange(*args.num_gms_range).tolist()
        args.num_gms_lst += [max_num_gms] if max_num_gms not in args.num_gms_lst else []
    args.num_gms_lst = [args.num_gms] if args.num_gms is not None else args.num_gms_lst
    args.max_num_gms = max(args.num_gms_lst)
    
    # Friction multiplier list
    args.friction_mul_lst = [args.friction_mul]
    if len(args.friction_mul_range) != 0:
        assert len(args.friction_mul_range) == 3
        max_friction_mul = args.friction_mul_range[1]
        args.friction_mul_lst = np.arange(*args.friction_mul_range).tolist()
        args.friction_mul_lst += [max_friction_mul] if max_friction_mul not in args.friction_mul_lst else []
    args.friction_mul_lst = [args.friction_mul] if args.friction_mul is not None else args.friction_mul_lst


def _validate_constraints(args):
    """Validate argument constraints."""
    if args.goal_time is not None:
        assert not args.keyboard_ctrl
        assert args.goal_speed is None and args.goal_ratio_range == []
    
    if args.budget_portion is not None:
        assert (args.goal_time is not None) or (args.goal_speed is not None)
        assert np.allclose(sum(args.budget_portion), 1)
        assert len(args.speed_describe) == len(args.budget_portion)
    
    if args.scan_time:
        assert args.num_envs == 1
        args.scan_time_save_dir = os.path.join(args.eval_dir, args.task_name, "3D_Analysis")
        check_file_exist(args.scan_time_save_dir)
        os.makedirs(args.scan_time_save_dir, exist_ok=True)


def _build_experiment_name(args):
    """Build experiment name based on configuration."""
    eval_config = ''
    
    if args.random_policy:
        args.final_name = f'EVAL_RandPolicy'
    elif args.heuristic_policy:
        args.final_name = f'EVAL_HeurPolicy'
    else:
        eval_config += '_EVAL_' + args.index_episode
    
    if args.add_restitution:
        eval_config += '_Hrest'
    if args.interpolate_joints != 1:
        eval_config += f'_Intp{args.interpolate_joints}'
    if args.num_gms_eval is not None:
        eval_config += f'_Gm{args.num_gms_eval}'
    if args.num_props_eval is not None:
        eval_config += f'_Props{args.num_props_eval}'
    if args.goal_time is not None:
        eval_config += f'_RT{args.goal_time}'
    if args.specific_idx:
        eval_config += f'_Idx{args.specific_idx}'
    
    if args.apply_disturbances:
        if len(args.disturbance_v_range) > 0:
            eval_config += '_MultDisturb'
        else:
            eval_config += '_Disturb'
    
    if args.budget_portion is not None:
        eval_config += f'_Staged'
        if args.use_avg_speed:
            eval_config += f'Avg'
        if args.record_init_configs:
            eval_config += f'_Configs'
    
    temp_filename = args.final_name + eval_config
    
    maximum_name_len = 250
    if len(temp_filename) > maximum_name_len:
        shorten_name_range = len(temp_filename) - maximum_name_len
        args.final_name = args.final_name[:-shorten_name_range]
    args.final_name = args.final_name + eval_config
    
    print('Uniform name is:', args.final_name)


def _setup_directories(args):
    """Setup result directories."""
    args.save_dir = os.path.join(args.eval_dir, args.task_name)
    args.instance_dir = os.path.join(args.save_dir, args.final_name)
    args.trajectory_dir = os.path.join(args.instance_dir, 'trajectories')
    args.blender_dir = os.path.join(args.instance_dir, 'blender')
    args.csv_file_path = os.path.join(args.instance_dir, 'data.csv')
    args.json_file_path = os.path.join(args.instance_dir, 'config.json')
    
    if args.saving:
        check_file_exist(args.csv_file_path)
        check_file_exist(args.trajectory_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.instance_dir, exist_ok=True)
        os.makedirs(args.trajectory_dir, exist_ok=True)
    
    if args.saving and args.blender_record:
        check_file_exist(args.blender_dir)
        os.makedirs(args.blender_dir)


# 3D Visualization for simple scenes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from tf_utils import tf_combine

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
    # pane_color = (1, 1, 1, 0.7)  # RGBA in 0..1
    # ax.w_xaxis.set_pane_color(pane_color)
    # ax.w_yaxis.set_pane_color(pane_color)
    # ax.w_zaxis.set_pane_color(pane_color)

    # for axis in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
    #     axis.line.set_color((1,1,1,0))  # transparent
    #     axis._axinfo["grid"]["linewidth"] = 0  # grid line width to 0 (legacy)

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