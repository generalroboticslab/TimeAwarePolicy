# Time-Aware Policy Learning for Adaptive and Punctual Robot Control

### [Yinsen Jia](https://yjia.net), [Boyuan Chen](http://boyuanchen.com)
Duke University

**Paper: [https://arxiv.org/abs/2511.07654](https://arxiv.org/abs/2511.07654)** <br>
**Video: [https://youtu.be/NwvgLdydJFk](https://youtu.be/NwvgLdydJFk)** <br>
**Website: [http://generalroboticslab.com/TimeAwarePolicy](http://generalroboticslab.com/TimeAwarePolicy)**

<div align="left">
    <img src="web_assets/teaser.gif" width="800">
</div>

## Installation
```
conda create --name timeaware python=3.8
conda activate timeaware
pip install -r requirements.txt --no-cache-dir

# IsaacGym installation
cd isaacgym/python && pip install -e . && cd ../..
```

## Quick Start
Run following command to play with the time-aware policy!

🎮 Keyboard Controls
- ⬆️ Increase the time ratio by 0.1
- ⬇️ Reduce the time ratio by 0.1

<div align="left">
    <img src="web_assets/KeyboardCtrl.gif" width="800">
</div>

### Cube stacking
```
python tw_evaluation.py --rendering --num_envs 1 --par_configs --checkpoint 20250717_162724_tw_FrankaCubeStack --index_episode best_rew --keyboard_ctrl --draw_scevel --goal_speed 0.6
```

### Granular media pouring
```
python tw_evaluation.py --rendering --num_envs 1 --par_configs --checkpoint 20250715_123940_tw_FrankaGmPour --index_episode best_rew --keyboard_ctrl --draw_scevel --goal_speed 0.6
```

### Drawer opening
```
python tw_evaluation.py --rendering --num_envs 1 --par_configs --checkpoint 20250730_151924_tw_FrankaCabinet --index_episode best_rew --keyboard_ctrl --draw_scevel --goal_speed 0.6
```

## Training
Replace `TASK_NAME` to one of names from `FrankaCubeStack`, `FrankaGmPour`, or `FrankaCabinet`.
If you want to add your own custom environments, please follow steps described in [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs).

During training, all results are saved in the `\train_res` folder.
We use wandb for logging. Therefore, you might need to log to your own account.

### Time-unaware policy training
```
python tw_training.py --saving --fix_priv --task_name TASK_NAME
```

### Learning the temporal lower bound
Replace `CKPT` to the time-unaware policy ckpt name (same as its folder name and wandb name).
```
python tw_training.py --saving --fix_priv --reset_critic --warmup_iters 50 --no_dense --epstimeRewardScale "[100, 100]" --successRewardScale 1000 --index_episode init --checkpoint CKPT --task_name TASK_NAME
```

### Embed temporal observations
Replace `CKPT` to the time-optimal policy ckpt name in the previous stage.
```
python tw_training.py --saving --stu_train --lr 5e-4 --warmup_rand --time_ratio --quiet False --wandb False --index_episode best_rew --checkpoint CKPT --task_name TASK_NAME
```

### Estimate the temporal lower bound
Replace `CKPT` to the augmented time-optimal policy ckpt name in the previous stage.
```
python tw_evaluation.py --saving --num_envs 10000 --target_success_eps 10000 --target_record_eps 1000 --save_threshold 10 --record_init_configs --use_par_checkpoint --index_episode best --checkpoint CKPT
```

### Learning the time-aware policy
Replace `CKPT` to same augmented time-optimal policy ckpt name in the previous stage.
This stage will use configurations that collected in the previous stage.
```
python tw_training.py --saving --lr 2e-4 --gamma 1. --no_dense --time2end --time_ratio --ratio_range "[0.2, 1]" --use_cost --fixed_configs --epstimeRewardScale "[100, 100]" --index_episode best --checkpoint CKPT --task_name TASK_NAME
```

## Evaluation
After each evaluation, results are saved in the `\eval_res` folder.

### Experiment 1: Time awareness improves efficiency and punctuality
```
python tw_evaluation.py --saving --num_envs 2000 --target_success_eps 2000 --strict_eval 

# (For cube stacking task only) Use a container as target instead of another cube +
--use_container

# For the time-unaware policy +
--index_episode init --checkpoint CKPT

# For the time-aware policy +
--par_configs --index_episode best_rew --goal_ratio_range "[0.2, 1.0, 0.1]" --checkpoint CKPT
```

### Experiment 2: Adaptive stability and environmental robustness
```
python tw_evaluation.py --saving --num_envs 2000 --target_success_eps 2000 --strict_eval 

# Cube stacking (increase the restitution) +
--add_restitution

# Granular Media Pouring (increase the number of beans) +
--num_gms_eval 40 

# Drawer Opening
# Increase the joint friction +
--friction_mul 2

# Increase weights in the drawer +
--num_props_eval 6

# For the time-unaware policy +
--index_episode init --checkpoint CKPT

    # For the time-unaware policy & joint interpolation baseline +
    --interpolate_joints 4

# For the time-aware policy +
--par_configs --index_episode best_rew --goal_ratio_range "[0.2, 1.0, 0.1]" --checkpoint CKPT
```

### Experiment 3: Punctuallity and resiliency
This experiment uses `FrankaCubeStack` task.
```
python tw_evaluation.py --saving --num_envs 2000 --target_success_eps 2000 --strict_eval --apply_disturbances --disturbance_v 10

# For the time-unaware policy +
--index_episode init --checkpoint CKPT

# For the time-aware policy +
--par_configs --index_episode best_rew --goal_ratio_range "[0.2, 1.0, 0.1]" --checkpoint CKPT
```

### Experiment 4: Human-in-the-loop temporal control for real-time behavior adaptation
Heuristic stage-wise control: the manipulation process is divided into distinct stages. Each
stage is assigned a tailored time ratio.
```
python tw_evaluation.py --saving --num_envs 2000 --target_success_eps 2000 --strict_eval --par_configs --index_episode best_rew --goal_speed 0.5 --checkpoint CKPT

# Cube stacking +
--budget_portion "[0.15, 0.35, 0.15, 0.35]" --speed_describe "[1, 0, 1, 0]" 

# Granular media pouring +
--budget_portion "[0.5, 0.5]" --speed_describe "[1, 0]"

# Drawer opening +
--budget_portion "[0.2, 0.2, 0.3, 0.3]" --speed_describe "[1, 0, 1, 0]" 
```

Online interface control: the user provides real-time time ratio via a simple and intuitive
interface (e.g., keyboard or slider) to directly steer the behavior of the robot to align with
high-level human intents.
```
python tw_evaluation.py --rendering --draw_scevel --keyboard_ctrl --simple_layout --num_envs 1 --par_configs --index_episode best_rew --checkpoint CKPT
```


## Real Franka Robot Deployment
### Install the franka_ros_interface
We are using the joint impedance controller and the gripper control from [franka_ros_interface](https://github.com/justagist/franka_ros_interface). Please follow the documentation to setup the controller.

### Camera callibration
We using realsense camera for the experiment. To calibrate the camera external matrix, you can follow [Franka-camera calibration](https://fr3setup.readthedocs.io/en/latest/camera_calibration/moveit.html).

### Real robot evaluation
After setting up the controller and the camera, you can start to receive the command and send to the joint impedance controller and gripper controller.

```
python tw_evaluation.py --num_envs 1 --real_robot --par_configs --index_episode best_rew --checkpoint CKPT

# To specify the real world scheduled time (e.g. 10s) +
--goal_time 10s

# To specify the time ratio (e.g. 0.5) +
--goal_speed 0.5

# To use stage wise control (e.g. 10s in total, fast then slow)
--goal_time 10 --budget_portion "[0.4, 0.6]" --speed_describe "[1, 0]"
```

Real robot online interface control
```
python tw_evaluation.py --num_envs 1 --real_robot --par_configs --keyboard_ctrl --draw_scevel --goal_speed 0.2 --index_episode best_rew --checkpoint CKPT
```

## Project structure
```
.
├── envs                    # Simulaion environments
│   ├── assets
│   └── isaacgymenvs
├── model                   # Policy architecture
│   ├── agent.py
│   └── utils.py
├── train_res               # Training results and checkpoints
│   ├── FrankaCabinet
│   ├── FrankaCubeStack
│   └── FrankaGmPour
├── eval_res                # Evaluation results
│   ├── FrankaCabinet
│   ├── FrankaCubeStack
│   └── FrankaGmPour
├── real_robot              # Real robot scripts (object detector and communicator)
│   ├── DemoCamera.py
│   ├── RealSenseCamera.py
│   ├── SocketClient.py
│   └── StateEstimator.py
├── isaacgym                # Isaacgym simulator
├── plot_utils.py           # Visualization script
├── tw_training.py          # Training script
├── tw_training_utils.py    # Training helper script
├── tw_evaluation.py        # Evaluation script
├── tw_evaluation_utils.py  # Evaluation helper script
├── tf_utils.py             # Transformation helper script
├── utils.py                # General (I/O) helper script
├── requirements.txt        # Dependencies
├── README.md
```

## License
This repository is released under the CC BY-NC-ND 4.0 License. Duke University has filed patent rights for the technology associated with this article. For further license rights, including using the patent rights for commercial purposes, please contact Duke's Office for Translation and Commercialization ([otcquestions@duke.edu](mailto:otcquestions@duke.edu)) and reference OTC DU9041PROV. See [LICENSE](LICENSE-CC-BY-NC-ND-4.0.md) for additional details. 


## Acknowledgement
This work is supported by DARPA TIAMAT program under award HR00112490419, ARO under award W911NF2410405, and ARL STRONG program under awards W911NF2320182, W911NF2220113, and W911NF242021.


## BibTeX
If you find our paper or codebase helpful, please consider citing:
```
@misc{jia2025timeawarepolicylearningadaptive,
      title={Time-Aware Policy Learning for Adaptive and Punctual Robot Control}, 
      author={Yinsen Jia and Boyuan Chen},
      year={2025},
      eprint={2511.07654},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.07654}, 
}
