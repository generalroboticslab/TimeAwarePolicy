
import argparse
import json
import os
from distutils.util import strtobool

import numpy as np

from core.common.io import check_file_exist, read_json

def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate the Trained Model')
    parser.add_argument('--rendering', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument("--num_envs", type=int, default=10)
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--train_res_dir', type=str, default='train_res', required=False)
    parser.add_argument('--eval_res_dir', type=str, default='eval_res', required=False)
    parser.add_argument('--saving', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--index_episode', type=str, default='last')
    parser.add_argument('--eval_result', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--sim_device', type=str, default="cuda:0")
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics device ID used for rendering (Vulkan ordinal)')
    
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
    parser.add_argument('--warmup_episodes', type=int, default=0, nargs='?', const=None)
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
    parser.add_argument(
        '--knn_configs_eval',
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs='?',
        const=True,
        help=(
            'Sample fresh simulation configurations and use the parent 1,000-entry '
            'reference bank only to estimate T_min and P_max with 5-nearest-neighbor averaging.'
        ),
    )
    parser.add_argument(
        '--constraint_cost_eval',
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs='?',
        const=True,
        help=(
            'Persist the success-conditioned episode sum of the manuscript '
            'instability cost max(p_t - p_max * tr, 0).'
        ),
    )
    parser.add_argument(
        '--paired_stage_eval',
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs='?',
        const=True,
        help=(
            'Record one strict-evaluation episode per environment, including '
            'the initial configuration and peak instability restricted to '
            'stages labelled stable.'
        ),
    )
    parser.add_argument(
        '--fixed_config_repeats_eval',
        type=int,
        default=None,
        help=(
            'For paired fixed-bank evaluation, enumerate the first '
            'num_envs/repeats bank entries exactly this many times instead '
            'of sampling bank entries with replacement.'
        ),
    )
    parser.add_argument(
        '--eval_tag',
        type=str,
        default=None,
        help='Optional short suffix used to separate evaluation artifacts.',
    )
    parser.add_argument(
        '--cube_gripper_clearance_m',
        type=float,
        default=0.05,
        help='Cube success clearance in meters; manuscript-aligned default: 0.05.',
    )
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
    parser.add_argument(
        '--controller_ip', type=str, default=None,
        help='Franka controller host. Required when --real_robot is enabled.',
    )
    parser.add_argument('--controller_sub_port', type=int, default=5555)
    parser.add_argument('--controller_pub_port', type=int, default=5556)
    parser.add_argument(
        '--demo_camera_index', type=int, default=0,
        help='OpenCV camera index used for optional real-robot recordings.',
    )
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
    parser.add_argument(
        '--demo_dir', type=str, default='demos',
        help='Directory for optional real-robot recordings.',
    )
    parser.add_argument('--demo_name', type=str, default=None)
    
    args = parser.parse_args(argv)
    _process_args(args)
    _setup_directories(args)
    
    return args


def _process_args(args):
    """Process and validate arguments."""
    if args.task_name is None:
        args.task_name = args.checkpoint.split('_')[3]
    
    # Load checkpoint configuration
    checkpoint_folder = os.path.join(args.train_res_dir, args.task_name, args.checkpoint)
    args.json_file_path = os.path.join(checkpoint_folder, 'config.json')
    args.checkpoint_path = os.path.join(checkpoint_folder, 'checkpoints', 'eps_' + args.index_episode)
    assert os.path.exists(args.checkpoint_path), f"Checkpoint path {args.checkpoint_path} does not exist"
    
    restored_eval_args = args.__dict__.copy()
    args_json = read_json(args.json_file_path)
    
    if args.use_par_checkpoint:
        assert args_json["checkpoint"] is not None
        par_checkpoint_folder = os.path.join(args.train_res_dir, args.task_name, args_json["checkpoint"])
        par_json_file_path = os.path.join(par_checkpoint_folder, 'config.json')
        restored_eval_args["checkpoint_path"] = os.path.join(par_checkpoint_folder, 'checkpoints', 'eps_' + args_json["index_episode"])
        restored_eval_args["final_name"] = args_json["final_name"]
        args_json = read_json(par_json_file_path)
    
    args.__dict__.update(args_json)
    args.__dict__.update(restored_eval_args)
    
    # Handle argument relations
    _validate_and_update_args(args, args_json)
    _build_evaluation_lists(args)
    _validate_constraints(args)
    _build_experiment_name(args)


def _validate_and_update_args(args, args_json):
    """Validate and update argument relationships."""
    if args.cube_gripper_clearance_m <= 0:
        raise ValueError("--cube_gripper_clearance_m must be positive")
    if args.warmup_episodes is None:
        args.warmup_episodes = args.num_envs # * 5
    
    if args.record_init_configs and not args.update_configs:
        if args.save_threshold is None or args.target_record_eps is None:
            raise Exception("Need to set save_threshold and target_record_eps to record initial configs")
        args.target_success_eps = args.target_record_eps * args.save_threshold

    if args.paired_stage_eval:
        assert args.strict_eval, "paired_stage_eval requires strict_eval"
        assert args.record_init_configs, (
            "paired_stage_eval requires record_init_configs"
        )
        assert args.save_threshold == 1, (
            "paired_stage_eval requires save_threshold=1"
        )
        assert args.target_record_eps == args.num_envs, (
            "paired_stage_eval requires target_record_eps=num_envs"
        )
        assert args.budget_portion is not None and args.speed_describe, (
            "paired_stage_eval requires stage definitions"
        )
    if args.fixed_config_repeats_eval is not None:
        assert args.paired_stage_eval, (
            "fixed_config_repeats_eval requires paired_stage_eval"
        )
        assert args.fixed_configs_eval, (
            "fixed_config_repeats_eval requires fixed_configs_eval"
        )
        assert args.fixed_config_repeats_eval > 0, (
            "fixed_config_repeats_eval must be positive"
        )
        assert args.num_envs % args.fixed_config_repeats_eval == 0, (
            "num_envs must be divisible by fixed_config_repeats_eval"
        )
    
    if args.specific_idx_eval is not None:
        args.specific_idx = args.specific_idx_eval
        args.fixed_configs = True
    
    if args.fixed_configs_eval is not None:
        args.fixed_configs = args.fixed_configs_eval
    
    if args.global_configs_eval:
        args.global_configs = True
    
    if args.knn_configs_eval:
        assert args_json["checkpoint"] is not None, (
            "5-NN evaluation requires a parent checkpoint with a recorded reference bank"
        )
        args.par_configs = True
        args.par_checkpoint = args_json["checkpoint"]
        args.par_index_episode = args_json["index_episode"]
        # The reference bank supplies only T_min and P_max. Task states are
        # freshly sampled from the full evaluation distribution.
        args.fixed_configs = False
    elif args.par_configs_eval or args.update_configs:
        args.par_configs = True
        if not getattr(args, "global_configs", False):
            assert args.fixed_configs, "Par configs evaluation requires fixed configs or global configs"
            assert args_json["checkpoint"] is not None, "Par configs evaluation requires a parent checkpoint with fixed configs"
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
        if not args.controller_ip:
            raise ValueError('--controller_ip is required when --real_robot is enabled')


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
        assert args.goal_ratio_range == []
        assert args.goal_speed is None or args.paired_stage_eval, (
            "goal_speed with parent fixed configurations is reserved for "
            "paired stage evaluation"
        )
    
    if args.budget_portion is not None:
        assert (args.goal_time is not None) or (args.goal_speed is not None)
        assert np.allclose(sum(args.budget_portion), 1)
        assert len(args.speed_describe) == len(args.budget_portion)
    
    if args.scan_time:
        assert args.num_envs == 1
        args.scan_time_save_dir = os.path.join(args.eval_res_dir, args.task_name, "3D_Analysis")
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
    if args.goal_speed is not None:
        eval_config += f'_TR{args.goal_speed}'
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
    if args.knn_configs_eval:
        eval_config += '_KNNFresh5'
    if args.constraint_cost_eval:
        eval_config += '_CostV1'
    if args.paired_stage_eval:
        eval_config += '_PairedFullStableMeanPeakV6'
        if args.fixed_configs_eval:
            eval_config += '_FixedBank'
        if args.fixed_config_repeats_eval is not None:
            eval_config += f'_Repeat{args.fixed_config_repeats_eval}'
    if args.eval_tag:
        eval_config += f'_{args.eval_tag}'
    
    temp_filename = args.final_name + eval_config
    
    maximum_name_len = 250
    if len(temp_filename) > maximum_name_len:
        shorten_name_range = len(temp_filename) - maximum_name_len
        args.final_name = args.final_name[:-shorten_name_range]
    args.final_name = args.final_name + eval_config
    
    print('Uniform name is:', args.final_name)


def _setup_directories(args):
    """Setup result directories."""
    args.save_dir = os.path.join(args.eval_res_dir, args.task_name)
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
