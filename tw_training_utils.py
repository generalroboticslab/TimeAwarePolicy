import os
import argparse
import datetime
import json
import psutil
from distutils.util import strtobool


def parse_args():
    parser = argparse.ArgumentParser(description='Time-aware Policy Learning')
    
    # Env hyper parameters
    parser.add_argument('--isaacgym', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--saving', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--task_name', type=str, required=True, help='FrankaCubeStack, FrankaGmPour, FrankaCabinet')
    parser.add_argument('--rendering', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--realtime', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--quiet', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    
    # IsaacGym specific arguments
    parser.add_argument('--use_gpu_pipeline', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--graphics_device_id', type=int, default=-1, help='Graphics Device ID')
    parser.add_argument('--buffer_multiplier', type=float, default=4.)
    parser.add_argument('--headless', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--nographics', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--subscenes', type=int, default=0)
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')
    parser.add_argument('--dt', type=float, default=1/60)
    
    # Training Env parameters
    parser.add_argument('--sequence_len', type=int, default=1)
    parser.add_argument('--ratio_range', type=json.loads, default=None)
    parser.add_argument('--torque_limits', type=float, default=None)
    parser.add_argument('--scale_actions', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--fixed_configs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--global_configs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--par_configs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--specific_idx', type=int, default=None)
    parser.add_argument('--add_cube_noise', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--episodeLength', type=int, default=900)
    parser.add_argument('--epstimeRewardScale', type=json.loads, default=[0., 0.])
    parser.add_argument('--steptimeRewardScale', type=float, default=0.)
    parser.add_argument('--scevelRewardScale', type=json.loads, default=[0., 0.])
    parser.add_argument('--sceaccRewardScale', type=float, default=0.)
    parser.add_argument('--scevelSchedule', type=float, default=1.)
    parser.add_argument('--actvelPenaltyScale', type=float, default=0.)
    parser.add_argument('--actaccPenaltyScale', type=float, default=0.)
    parser.add_argument('--exp_scheduler', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--vel_match', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--no_dense', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--accm_instability', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--time2end', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--time_ratio', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--fix_linvel', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--fix_limvel', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--fix_priv', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--meta_rl', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--reset_critic', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--act_scale', type=float, default=1.)
    parser.add_argument('--control_freq_inv', type=int, default=3)
    parser.add_argument('--gripper_freq_inv', type=int, default=10)
    parser.add_argument('--max_vel_subtract', type=float, default=0.7)
    parser.add_argument('--limit_gripper_vel', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--control_type', type=str, default="ik", nargs='?', const=True)
    parser.add_argument('--add_obs_noise', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--add_act_noise', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--init_curri_ratio', type=float, default=0.)
    parser.add_argument('--success_threshold', type=float, default=0.90)
    parser.add_argument('--curriculum_step', type=float, default=0.03)
    parser.add_argument('--curri_hold_iters', type=int, default=10)
    
    # Training Env parameters (FrankaGmPour)
    parser.add_argument('--constrain_grasp', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--num_gms', type=int, default=1)
    parser.add_argument('--successRewardScale', type=float, default=200.)
    parser.add_argument('--use_potential_r', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    
    # I/O hyper parameter
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--result_dir', type=str, default='train_res', required=False)
    parser.add_argument('--wandb', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--force_name', default=None, type=str)
    
    # Algorithm specific arguments
    parser.add_argument('--env_name', default="TimeVarRL")
    parser.add_argument('--beta', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--squashed', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument("--use_lstm", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--use_pc_extractor", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--total_timesteps", type=int, default=int(2e10))
    parser.add_argument("--num_envs", type=int, default=16384)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--pc_batchsize", type=int, default=None)
    parser.add_argument("--use_relu", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument('--scheduler', default="linear")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--minibatch-size", type=int, default=131072)
    parser.add_argument("--update_epochs", type=int, default=5)
    parser.add_argument("--norm_obs", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--norm_rew", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--norm_cost", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--use_cost", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--num_cost", type=int, default=2)
    parser.add_argument("--use_fourier", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--ent-coef", type=json.loads, default=[0.005, 0.005])
    parser.add_argument("--bounds_loss_coef", type=float, default=0.0001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=2.5)
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--gamma', type=float, default=0.995, metavar='G')
    parser.add_argument('--c_gamma', type=json.loads, default=[1, 0.99], metavar='G')
    parser.add_argument('--c_scale', type=json.loads, default=[0, 1], metavar='G')
    parser.add_argument('--tau', type=float, default=0.0005, metavar='G')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='G')
    parser.add_argument('--seed', type=int, default=123456, metavar='N')
    parser.add_argument('--lstm_hidden_size', type=json.loads, default=256, metavar='N')
    parser.add_argument('--fourier_hidden_size', type=json.loads, default=16, metavar='N')
    parser.add_argument('--hidden_size', type=json.loads, default=[256, 128, 64], metavar='N')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--index_episode', type=str, default='last')
    parser.add_argument("--freeze", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument('--random_policy', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--record_iter', type=int, default=10)
    parser.add_argument('--running_len', type=int, default=30000)
    parser.add_argument('--warmup_iters', type=int, default=0)
    parser.add_argument("--best_only", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--last_only", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--init_success", type=float, default=0.98)
    parser.add_argument('--curr_rate', type=int, default=1)
    
    # Teacher Student Training
    parser.add_argument('--stu_train', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--warmup_rand', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    
    # PyTorch specific arguments
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument('--cpus', type=int, default=[], nargs='+')
    parser.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    
    args = parser.parse_args()
    
    # Post-processing and validation
    if args.cpus:
        print('Running on specific CPUS:', args.cpus)
        process = psutil.Process()
        process.cpu_affinity(args.cpus)
    
    if args.realtime:
        args.rendering = True
    
    if args.stu_train:
        assert args.checkpoint is not None, "Student training should be fine-tuned from a teacher model"
    if args.par_configs:
        args.fixed_configs = True
    if args.ratio_range:
        assert args.fixed_configs is True, f"Speed range requires fixed_configs to be True"
        if args.task_name == "FrankaCubeStack": args.episodeLength = 1900
        if args.task_name == "FrankaGmPour": args.episodeLength = 1600
        if args.task_name == "FrankaCabinet": args.episodeLength = 3100
    if args.task_name == "FrankaCabinet":
        assert args.episodeLength >= 800, f"FrankaCabinet task requires at least 800 steps per episode"
    
    # Build naming convention
    _build_experiment_name(args)
    # Setup directories
    _setup_directories(args)
    
    return args


def _build_experiment_name(args):
    """Build experiment name based on configuration."""
    additional = f'_{args.control_type.upper()}_{args.task_name}'
    
    # Task specific
    if args.task_name == 'FrankaGmPour':
        additional += f'_{args.num_gms}Gm'
        if args.constrain_grasp:
            additional += '_ConGrasp'
    
    if args.stu_train:
        additional += '_Stu'
        if args.vf_coef == 0:
            additional += '_NoCritic'
    
    # Algorithm specific
    additional += '_MetaRL' if args.meta_rl else ''
    additional += '_Beta' if args.beta else '_Normal'
    additional += '_Squashed' if args.squashed else ''
    additional += '_LSTM' if args.use_lstm else ''
    additional += '_Fourier' if args.use_fourier else ''
    
    # Training type
    args.pre_train = True if args.checkpoint is None else False
    args.init_curri_ratio = 1. if not args.pre_train else args.init_curri_ratio
    if args.checkpoint is not None:
        assert min(args.epstimeRewardScale) >= 0
        additional += '_FT'
        additional += '_Rcritic' if args.reset_critic else ''
        args.total_timesteps = int(2e10)
        _load_checkpoint_config(args)
        
        if args.fixed_configs:
            additional += '_FConf'
            additional += 'Glob' if args.global_configs else ''
            if args.specific_idx is not None:
                additional += f'{args.specific_idx}'
            if args.add_cube_noise:
                additional += '_Cube'
    else:
        additional += '_PreT'
        args.total_timesteps = int(1e10)
    
    if not args.fix_priv:
        if args.time2end:
            additional += '_T2E'
        if args.time_ratio:
            additional += 'ratio'
    if args.use_cost:
        additional += '_P3O'
    
    # Randomization
    if args.ratio_range:
        additional += f'_SpeedVar_{args.ratio_range[0]}to{args.ratio_range[1]}'
        if args.scale_actions:
            additional += '_scaleA'
    if args.add_obs_noise:
        additional += '_obsN'
    if args.add_act_noise:
        additional += '_actN'
    
    # Weight and frequency settings
    additional += f'_gam{args.gamma}'
    if args.use_cost:
        additional += f'_cGam{args.c_gamma[0]}_{args.c_gamma[1]}'
        additional += f'_cScale{args.c_scale[0]}_{args.c_scale[1]}'
    if args.no_dense:
        additional += '_noDense'
    if args.control_freq_inv > 0:
        assert 1 / args.dt % args.control_freq_inv == 0
        control_freq = int(1 / args.dt // args.control_freq_inv)
        additional += f'_Ctrl{control_freq}Hz'
    if args.gripper_freq_inv > 0:
        assert 1 / args.dt / args.control_freq_inv % args.gripper_freq_inv == 0
        gripper_freq = int(1 / args.dt / args.control_freq_inv // args.gripper_freq_inv)
        additional += f'_Grip{gripper_freq}Hz'
    if args.max_vel_subtract > 0:
        additional += f'_maxVel{1-args.max_vel_subtract:.1f}'
    if args.limit_gripper_vel:
        additional += f'_LimGrip'
    if args.torque_limits is not None:
        additional += f'_tau{args.torque_limits}'
    if max(args.epstimeRewardScale) > 0:
        additional += f'_epsT{args.epstimeRewardScale[0]}to{args.epstimeRewardScale[1]}' \
              if args.epstimeRewardScale[0] != args.epstimeRewardScale[1] else f'_epsT{args.epstimeRewardScale[0]}'
    if args.ratio_range:
        if args.fix_linvel:
            additional += '_NoLinVel'
        if args.fix_limvel:
            additional += '_NoLimGt'
        if args.steptimeRewardScale > 0:
            additional += f'_stepT{args.steptimeRewardScale}'
        if max(args.scevelRewardScale) > 0 and not args.use_cost:
            additional += f'_sceVel{args.scevelRewardScale[0]}to{args.scevelRewardScale[1]}'
            if args.accm_instability:
                additional += 'Accm'
        if args.sceaccRewardScale > 0:
            additional += f'_sceAcc{args.sceaccRewardScale}'
        if args.scevelSchedule < 1:
            additional += f'_sceMul{args.scevelSchedule}'
        if args.exp_scheduler:
            assert args.scevelSchedule >= 1
            additional += f'_sceExp{args.scevelSchedule}'
        if args.actvelPenaltyScale > 0:
            additional += f'_actVel{args.actvelPenaltyScale}'
        if args.actaccPenaltyScale > 0:
            additional += f'_actAcc{args.actaccPenaltyScale}'
        if args.curr_rate > 1:
            additional += f'_curr{args.curr_rate}'
    
    additional += f'_step{args.num_steps}'
    additional += f'_seq{args.sequence_len}'
    additional += f'_entropy{args.ent_coef[0]}' if args.ent_coef[0] == args.ent_coef[1] else f'_entropy_{args.ent_coef[0]}to{args.ent_coef[1]}'
    additional += f'_seed{args.seed}'
    
    args.timer = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.random_policy:
        args.final_name = args.timer + additional.replace('-train', '-random_policy')
    elif args.force_name:
        args.final_name = args.force_name + args.timer
    else:
        args.final_name = args.timer + additional


def _load_checkpoint_config(args):
    """Load configuration from checkpoint."""
    ckpt_json_file_path = os.path.join(args.result_dir, args.task_name, args.checkpoint, 'config.json')
    with open(ckpt_json_file_path, 'r') as json_obj:
        ckpt_json = json.load(json_obj)
    
    args.par_checkpoint = ckpt_json["checkpoint"]
    args.par_index_episode = ckpt_json["index_episode"]
    args.act_scale = ckpt_json.get("act_scale", args.act_scale)
    args.sequence_len = ckpt_json.get("sequence_len", args.sequence_len)
    args.control_freq_inv = ckpt_json.get("control_freq_inv", args.control_freq_inv)
    args.gripper_freq_inv = ckpt_json.get("gripper_freq_inv", args.gripper_freq_inv)
    args.max_vel_subtract = ckpt_json.get("max_vel_subtract", args.max_vel_subtract)
    args.limit_gripper_vel = ckpt_json.get("limit_gripper_vel", args.limit_gripper_vel)
    args.fix_linvel = ckpt_json.get("fix_linvel", args.fix_linvel)
    args.fix_limvel = ckpt_json.get("fix_limvel", args.fix_limvel)
    assert args.control_type == ckpt_json.get("control_type", args.control_type)


def _setup_directories(args):
    """Setup result directories."""
    args.result_dir = os.path.join(args.result_dir, args.task_name)
    args.instance_dir = os.path.join(args.result_dir, args.final_name)
    args.checkpoint_dir = os.path.join(args.instance_dir, 'checkpoints')
    args.trajectory_dir = os.path.join(args.instance_dir, 'trajectories')
    args.csv_file_path = os.path.join(args.instance_dir, 'data.csv')
    args.json_file_path = os.path.join(args.instance_dir, 'config.json')
    
    if args.saving:
        os.makedirs(args.result_dir, exist_ok=True)
        os.makedirs(args.instance_dir, exist_ok=False)
        os.makedirs(args.checkpoint_dir, exist_ok=False)
        os.makedirs(args.trajectory_dir, exist_ok=False)
