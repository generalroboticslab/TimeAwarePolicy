import json
from collections import OrderedDict
import os
import subprocess
import csv
import shutil
import numpy as np
import torch
import re
import h5py
import trimesh
import cv2
from PIL import Image 
import open3d as o3d
import trimesh
import numpy as np
import logging
import importlib
import xml.etree.ElementTree as ET
import traceback

# Keyboard controller
import time
try:
    from pynput import keyboard
except ImportError:
    pass


def remaining_sleep(start_time, duration, verbose=True):
    """
    Sleep for a duration, but if the time has already passed, return immediately.
    :param start_time: The time when the sleep started
    :param duration: The duration to sleep
    """
    remaining_time = duration - (time.perf_counter() - start_time)
    if remaining_time > 0:
        time.sleep(remaining_time)
    if verbose:
        print(f"Control Frequency: {1/(time.perf_counter() - start_time):.3f}Hz")


def convert_time(relative_time):
    relative_time = int(relative_time)
    hours = relative_time // 3600
    left_time = relative_time % 3600
    minutes = left_time // 60
    seconds = left_time % 60
    return f'{hours}:{minutes}:{seconds}'


def _format_time(t: float) -> str:
    """mm:ss.mmm"""
    negative = False
    if t < 0:
        negative = True
        t = abs(t)

    mm, ss = divmod(int(t), 60)
    mmm = int((t - int(t)) * 1_000)

    if negative:
        return f"-{mm:02d}:{ss:02d}.{mmm:03d}"
    else:
        return f"{mm:02d}:{ss:02d}.{mmm:03d}"


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def read_h5py_mem(h5py_path):
    with h5py.File(h5py_path, 'r') as f:
        return read_dataset_recursively(f)
    

def get_args_attr(args, attr_name, default_v=None):
    if hasattr(args, attr_name): 
        return getattr(args, attr_name)
    return default_v


def read_dataset_recursively(hdf5_group):
    """
    Recursively read datasets from an HDF5 group into a nested dictionary.

    :param hdf5_group: HDF5 group object to read the datasets from
    :return: Nested dictionary with the same structure as the HDF5 group/datasets
    """
    data_dict = {}
    for key, item in hdf5_group.items():
        if isinstance(item, h5py.Dataset):
            # Read the dataset and assign to the dictionary
            data_dict[key] = item[()]
        elif isinstance(item, h5py.Group):
            # Recursively read the group
            data_dict[key] = read_dataset_recursively(item)
    return data_dict


def create_dataset_recursively(hdf5_group, data_dict):
    """
    Recursively create datasets from a nested dictionary within an HDF5 group.

    :param hdf5_group: HDF5 group object to store the datasets
    :param data_dict: Nested dictionary containing the data to store
    """
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # Create a sub-group for nested dictionaries
            sub_group = create_or_update_group(hdf5_group, key)
            create_dataset_recursively(sub_group, value)
        else:
            # Create a dataset for non-dictionary items
            hdf5_group.create_dataset(key, data=value)


def create_or_update_group(parent, group_name):
    """Create or get a group in the HDF5 file."""
    return parent.require_group(group_name)


def save_h5py(data, h5py_path):
    with h5py.File(h5py_path, 'w') as f:
        for k, v in data.items():
            f.create_dataset(k, data=v)


def write_csv_line(result_file_path, result):
    """ write a line in a csv file; create the file and write the first line if the file does not already exist """
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(result)
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(result)


def dict2list(diction):
    """ convert a dictionary to a list of tuples """
    key_list, value_list = [], []
    for k, v in diction.items():
        if isinstance(v, dict):
            subkey_lst, subvalue_lst = dict2list(v)
            key_list.extend(subkey_lst)
            value_list.extend(subvalue_lst)
        else:
            key_list.append(k)
            value_list.append(v)
    return key_list, value_list


def sorted_dict(dictionary):
    for k, v in dictionary.items():
        if isinstance(v, dict):
            dictionary[k] = dict(sorted(v.items()))
    return dictionary


def subdir(file_path):
    return os.path.join(*file_path.split(os.path.sep)[1:])


def del_folders(folder_name, earlier_all=False, task_name="FrankaCubeStack"):
    utils_path = os.path.abspath(__file__)
    ckpt_dir = os.path.join(os.path.dirname(utils_path), "train_res", task_name)
    # find all the folders that contain the folder_name
    del_folders = []; earlier_folders = []
    for f in sorted(os.listdir(ckpt_dir)):
        if folder_name in f:
            del_folders.append(f)
            break
        if earlier_all:
            earlier_folders.append(f)
    del_folders += earlier_folders

    # Lst the folders to delete
    if len(del_folders) == 0:
        print(f"No folder found with the name {folder_name}!")
        return
    else:
        print(f"Find {len(del_folders)} folders:")
        for f in del_folders: print(f)
    
    # Remove the folders
    response = input(f"\nWhether remove or not (y/n):")
    if response.lower() == 'y':
        for f in del_folders:
            shutil.rmtree(os.path.join(ckpt_dir, f))
        print(f"{len(del_folders)} Folders removed!")
    else:
        print("Give up this evaluation because of exsiting file.")


def open_in_vscode(folder_name, task_name="FrankaCubeStack"):
    # Check if the folder exists
    utils_path = os.path.abspath(__file__)
    ckpt_dir = os.path.join(os.path.dirname(utils_path), "train_res", task_name)
    # find all the folders that contain the folder_name
    for f in sorted(os.listdir(ckpt_dir)):
        if folder_name in f:
            folder_path = os.path.join(ckpt_dir, f)
            break
    if not os.path.isdir(folder_path):
        print(f"The folder path '{folder_path}' does not exist.")
        return
    try:
        # Run the `code` command to open the folder in VS Code
        subprocess.run(["code", "--add", folder_path], check=True)
        print(f"Opened '{folder_path}' in VS Code.")
    except FileNotFoundError:
        print("The 'code' command is not found. Make sure VS Code is installed and added to PATH.")
    except Exception as e:
        print(f"An error occurred: {e}")


class KeyboardController:
    def __init__(self, envs, start_value=1.0, start_config=None):
        """
        Initializes the keyboard controller.

        :param envs: The environment object to send updates to.
        :param start_value: Initial value for the controlled parameter.
        """
        self.envs = envs
        self.current_value = start_value
        self.config_idx = start_config if start_config is not None else self.envs.cfg.get("specific_idx", None)
        self.sr_range = (0.2, 1.0)
        self.config_range = (0, 999)
        self.scevelSchedule = self.envs.cfg.get("scevelSchedule", 1.0)
        self.running = True
        self.listener = None
        self.last_key_pressed = None  # Store the last pressed key for visualization
        self.key_flash = {"UP": False, "DOWN": False}  # Track key flash states

    def start(self):
        """Starts the keyboard listener in a separate thread."""
        self.listener = keyboard.Listener(on_press=self._on_key_press)
        self.listener.start()

    def stop(self):
        """Stops the keyboard listener."""
        self.running = False
        if self.listener:
            self.listener.stop()

    def _on_key_press(self, key):
        """Callback function for keyboard key press events."""
        try:
            if key == keyboard.Key.up:  # Increase value by 0.1
                self.current_value += 0.1
                self.clip_current_value()
                self.last_key_pressed = "UP"
                self.key_flash["UP"] = True  # Trigger flash
                self.env_callback()
            elif key == keyboard.Key.down:  # Decrease value by 0.1
                self.current_value -= 0.1
                self.clip_current_value()
                self.last_key_pressed = "DOWN"
                self.key_flash["DOWN"] = True  # Trigger flash
                self.env_callback()
            elif key == keyboard.Key.left:  # Decrease value by 0.1
                if self.config_idx is not None:
                    self.config_idx -= 1
                    self.clip_current_value()
                    self.env_callback()
            elif key == keyboard.Key.right:  # Decrease value by 0.1
                if self.config_idx is not None:
                    self.config_idx += 1
                    self.clip_current_value()
                    self.env_callback()
            elif key.char == 's': 
                print(f"Current config idx: {self.config_idx}, Current speed ratio: {self.current_value}")
        except Exception as e:
            # traceback.print_exc()
            pass

    def clip_current_value(self):
        """Clips the current value to the specified range."""
        self.current_value = max(min(self.current_value, self.sr_range[1]), self.sr_range[0])
        if self.config_idx is not None:
            self.config_idx = max(min(self.config_idx, self.config_range[1]), self.config_range[0])

    def reset_key_flash(self):
        """Reset the key flash state after the flash."""
        self.key_flash["UP"] = False
        self.key_flash["DOWN"] = False

    def env_callback(self):
        """Callback to update the environment parameters."""
        self.envs.goal_speed = self.current_value
        self.envs.update_time_ratio_buf(self.current_value)
        self.envs.update_linvel_gt()
        self.envs.cfg["specific_idx"] = self.config_idx

####################################
############# 3D Utils #############
####################################
def get_on_bbox(bbox, z_half_extend:float):
    # scene_center_pos is the relative translation from the scene object's baselink to the center of the scene object's bounding box
    # All bbox given should be in the center frame (baselink is at the origin when import the urdf)
    SceneCenter_2_QRregionCenter = [bbox[0], bbox[1], bbox[9]+z_half_extend]
    orientation = bbox[3:7]
    QRregion_half_extents = bbox[7:10].copy()
    QRregion_half_extents[2] = z_half_extend
    return np.array([*SceneCenter_2_QRregionCenter, *orientation, *QRregion_half_extents])


def get_in_bbox(bbox, z_half_extend:float=None):
    if z_half_extend is None: z_half_extend = bbox[9]
    # Half extend should not be smaller than the original half extend
    z_half_extend = max(z_half_extend, bbox[9])
    # scene_center_pos is the relative translation from the scene object's baselink to the center of the scene object's bounding box
    # All bbox given should be in the center frame (baselink is at the origin when import the urdf)
    SceneCenter_2_QRregionCenter = [0, 0, z_half_extend-bbox[9]]
    orientation = [0, 0, 0, 1.]
    QRregion_half_extents = bbox[7:10].copy()
    QRregion_half_extents[2] = z_half_extend
    return np.array([*SceneCenter_2_QRregionCenter, *orientation, *QRregion_half_extents])


def pc_random_downsample(pc_array, num_points, autopad=False):
    """ Randomly downsample/shuffle a point cloud
        Args:
        pc_array: (N, 3) numpy array
        num_points: int
    """
    if num_points >= pc_array.shape[0]: 
        if autopad: # Pad the point cloud with zeros will make the next real scene points become sparse
            pc_array = np.concatenate([pc_array, np.zeros((num_points - pc_array.shape[0], 3))], axis=0) 
        return np.random.permutation(pc_array)
    else:
        return pc_array[np.random.choice(pc_array.shape[0], num_points, replace=False)]
    

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
        sampled_pc: sampled pointcloud data, [npoint, 3]
    """
    N, C = xyz.shape
    centroids = np.zeros(npoint, dtype=int)
    min_distance = np.ones(N) * 1e10 # min distance from the unsampled points to the sampled points
    farthest_idx = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest_idx
        centroid = xyz[farthest_idx, :]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < min_distance
        min_distance[mask] = dist[mask]
        farthest_idx = np.argmax(min_distance, axis=-1)
    sampled_pc = xyz[centroids]
    return sampled_pc


def inverse_sigmoid(x):
    return torch.log(x / (1 - x + 1e-10))
    

def create_mesh_grid(action_ranges=[(0, 1)]*6, num_steps=[5]*6):
    assert len(action_ranges) == len(num_steps), "action_ranges and num_steps must have the same length"
    action_steps = [torch.linspace(start, end, num_steps[j]) for j, (start, end) in enumerate(action_ranges)]
    # Use torch.meshgrid with explicit indexing argument
    meshgrid_tensors = torch.meshgrid(*action_steps, indexing='ij')
    # Stack the meshgrid tensors along a new dimension to get the final meshgrid tensor
    meshgrid_tensor = torch.stack(meshgrid_tensors, dim=-1)
    return meshgrid_tensor


def tensor_memory_in_mb(tensor):
    # Calculate the memory occupied by the tensor
    num_elements = tensor.numel()
    element_size = tensor.element_size()
    total_memory_bytes = num_elements * element_size
    total_memory_mb = total_memory_bytes / (1024 ** 2)  # Convert bytes to MB
    return total_memory_mb


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def check_file_exist(file_path):
    if os.path.exists(file_path):
        response = input(f"Find existing dir/file {file_path}! Whether remove or not (y/n):")
        if response == 'y' or response == 'Y':
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
        else: 
            raise Exception("Give up this evaluation because of exsiting file.")


### Multi-Envs Utils ###
def combine_envs_float_info2list(infos, key, env_ids=None):
    if env_ids is None: env_ids = range(len(infos))
    return [infos[id][key] for id in env_ids]


def combine_envs_dict_info2dict(infos, key, env_ids=None):
    if env_ids is None: env_ids = range(len(infos))
    merged_info = {}
    for id in env_ids:
        info_dict = infos[id][key]
        for k, v in info_dict.items():
            if k not in merged_info: 
                merged_info[k] = v
                continue
            cur_val, nums = merged_info[k]
            new_val, new_nums = v
            merged_info[k] = [(cur_val * nums + new_val * new_nums) / (nums + new_nums), nums + new_nums]
    return merged_info


# Transformation

def quaternions_to_euler_array(quaternions):
    """
    Convert an array of quaternions into Euler angles (roll, pitch, and yaw) using the ZYX convention.
    
    Parameters:
    quaternions: A numpy array of shape (N, 4) where each row contains the components of a quaternion [x, y, z, w]
    
    Returns:
    euler_angles: A numpy array of shape (N, 3) where each row contains the Euler angles [roll, pitch, yaw]
    """
    if quaternions.ndim == 1:
        quaternions = quaternions[np.newaxis, :]
        flatten_flag = True
    else:
        flatten_flag = False

    # Preallocate the output array
    euler_angles = np.zeros((quaternions.shape[0], 3))
    
    # Extract components
    w, x, y, z = quaternions[:, 3], quaternions[:, 0], quaternions[:, 1], quaternions[:, 2]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    # Combine the angles
    euler_angles[:, 0] = roll
    euler_angles[:, 1] = pitch
    euler_angles[:, 2] = yaw
    
    if flatten_flag:
        euler_angles = euler_angles.flatten()

    return euler_angles


def normalize(x, eps: float = 1e-9):
    # Normalize an array of vectors
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps, max=None)


def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([x, y, z, w], axis=-1).reshape(shape)
    return quat


def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = np.cross(xyz, b, axis=-1) * 2
    return (b + a[:, 3:] * t + np.cross(xyz, t, axis=-1)).reshape(shape)


def tf_combine(t1, q1, t2, q2):
    q1 = normalize(q1)
    q2 = normalize(q2)
    return quat_apply(q1, t2) + t1, quat_mul(q1, q2)


def se3_transform_pc(t, q, pc):
    if not isinstance(pc, np.ndarray):
        pc = np.array(pc)
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    if not isinstance(q, np.ndarray):
        q = np.array(q)
    # unsqueeze for broadcasting operation
    if len(t.shape) == 1:
        t = t[np.newaxis, :]
    if len(q.shape) == 1:
        q = q[np.newaxis, :]

    pc_shape = pc.shape
    t = t.repeat(pc_shape[0], axis=0)
    q = q.repeat(pc_shape[0], axis=0)
    return quat_apply(q, pc) + t


# Stable Placement
def generate_table(records, success_rate_name, success_rate_counts_name, table_name=None):
    success_misc = [
        [table_name]+[""]*(len(records[success_rate_name])),
        ["Num Objs in QR Scene"]+list(records[success_rate_name].keys()),
        ["Success Rate"]+list(
            map(lambda x: f"{x:.4f}" if isinstance(x, float) else x, 
                records[success_rate_name].values())
        ),
        ["Num Data Point"]+list(records[success_rate_counts_name].values()),
    ]
    return success_misc


# Video Recording
def read_video_frames(video_path):
    """
    Reads a video file and yields each frame.

    :param video_path: Path to the video file
    :return: Yields each frame of the video
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Read and yield each frame of the video
    while True:
        ret, frame = cap.read()

        # If the frame was not retrieved successfully, end of video is reached
        if not ret:
            break

        yield frame

    # When everything done, release the video capture object
    cap.release()


# Mesh Utils
def create_and_save_cuboid_mesh(file_path, extents=[0.04, 0.04, 0.04]):
    """
    Generate a cube mesh with the given edge length and save it to the specified file path.

    :param edge_length: Length of the cube's edge
    :param file_path: Path where the mesh file will be saved
    """
    # Create a cube mesh
    cube = trimesh.creation.box(extents=extents)

    # Save the mesh to the specified file path
    cube.export(file_path)

    print(f"Cube mesh saved to {file_path}")


def pc2mesh(pc_array, mesh_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_array)
    pcd.estimate_normals()

    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist   

    radii = [0.005, 0.01, 0.02, 0.04]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

    # create the triangular mesh with the vertices and faces from open3d
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), 
                               np.asarray(mesh.triangles),
                               vertex_normals=np.asarray(mesh.vertex_normals))
    print(f"Mesh is Convex: {trimesh.convex.is_convex(tri_mesh)}")
    tri_mesh.export(mesh_path)


def pc2mesh_v2(pc_array, mesh_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_array)

    alpha = 0.03
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()

    # create the triangular mesh with the vertices and faces from open3d
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), 
                               np.asarray(mesh.triangles),
                               vertex_normals=np.asarray(mesh.vertex_normals))
    print(f"Mesh is Convex: {trimesh.convex.is_convex(tri_mesh)}")
    tri_mesh.export(mesh_path)


def compute_pc_mesh_dim_ratio(pc_path, mesh_path):
    pc = np.load(pc_path, allow_pickle=True)
    mesh = trimesh.load(mesh_path)
    # Compute the pc bounding box dimensions
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(pc)
    pc_dim = o3d_pc.get_axis_aligned_bounding_box().get_half_extent() * 2
    mesh_dim = mesh.bounding_box.extents
    return pc_dim / mesh_dim


def process_mesh(org_mesh_path, save_mesh_path, scale_factor=[1e-3, 1e-3, 1e-3], rotate_angle=[0, 0, 0]):
    # scale mesh to meters, rotate mesh and transform mesh to origin
    mesh = trimesh.load(org_mesh_path)
    mesh.apply_scale(scale_factor)
    # rotate mesh using quaternion
    if rotate_angle != [0, 0, 0]:
        mesh.apply_transform(trimesh.transformations.euler_matrix(*rotate_angle))
    # transform mesh to origin
    mesh.apply_translation(-mesh.centroid)
    mesh.export(save_mesh_path)


# URDF related
# YCB dataset requires to set visual to be white. The default is black which will block the texture.
def create_urdf(robot_name, visual_mesh_filename, collision_mesh_filename, 
                mass=0.1, scale=[1, 1, 1.], origin_xyz=[0., 0., 0.], 
                inertia_ixx="0.0001", inertia_ixy="0.0", inertia_ixz="0.0", 
                inertia_iyy="0.0001", inertia_iyz="0.0", inertia_izz="0.0001", save_path=None,
                material=""):
    """
    Creates a URDF file as a string based on the given parameters.
    Same function as before. For convenience, just copy it here.

    :param robot_name: Name of the robot.
    :param visual_mesh_filename: Filename of the visual mesh.
    :param collision_mesh_filename: Filename of the collision mesh.
    :param mass: Mass of the link.
    :param origin_xyz: Origin offset, default "0.0 0.0 0.0".
    :param inertia_ixx, inertia_ixy, inertia_ixz, inertia_iyy, inertia_iyz, inertia_izz: Inertia parameters.
    :return: A string representing the URDF file.
    """
    
    urdf_template = \
f"""<?xml version='1.0' encoding='utf-8'?>
    <robot name="{robot_name}">
        <link name="link_0">
            <visual>
                <origin xyz="{' '.join(map(str, origin_xyz))}" />
                <geometry>
                    <mesh filename="{visual_mesh_filename}" scale="{' '.join(map(str, scale))}" />
                </geometry>
                {material}
            </visual>
            <collision>
                <origin xyz="{' '.join(map(str, origin_xyz))}" />
                <geometry>
                    <mesh filename="{collision_mesh_filename}" scale="{' '.join(map(str, scale))}" />
                </geometry>
            </collision>
            <inertial>
                <mass value="{mass}" />
                <inertia ixx="{inertia_ixx}" ixy="{inertia_ixy}" ixz="{inertia_ixz}" 
                            iyy="{inertia_iyy}" iyz="{inertia_iyz}" izz="{inertia_izz}" />
            </inertial>
        </link>
    </robot>"""
    if save_path:
        with open(save_path, 'w') as file:
            file.write(urdf_template)
        print(f"URDF file saved to {save_path}")

    return urdf_template


def find_urdf_mesh_files(urdf_file):
    """
    Find the visual and collision meshes specified in a URDF file.

    :param urdf_file: Path to the URDF file
    :return: Tuple containing the visual and collision mesh filenames
    """
    # Load the URDF file
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    for link in root.findall('link'):
        link_name = link.get('name')
        
        visual_mesh = None
        collision_mesh = None

        visual = link.find('visual')
        if visual is not None:
            visual_geometry = visual.find('geometry')
            if visual_geometry is not None:
                visual_mesh = visual_geometry.find('mesh')
                if visual_mesh is not None:
                    visual_mesh_filename = visual_mesh.get('filename')
                    visual_mesh_file_path = os.path.join(os.path.dirname(urdf_file), visual_mesh_filename)

        collision = link.find('collision')
        if collision is not None:
            collision_geometry = collision.find('geometry')
            if collision_geometry is not None:
                collision_mesh = collision_geometry.find('mesh')
                if collision_mesh is not None:
                    collision_mesh_filename = collision_mesh.get('filename')
                    collision_mesh_file_path = os.path.join(os.path.dirname(urdf_file), collision_mesh_filename)

    return visual_mesh_file_path, collision_mesh_file_path


def combine_images(img1, img2, alpha=1.0, beta=0.3):
    # Load another image to blend with
    width, height = img1.shape[1], img1.shape[0]
    img2 = cv2.resize(img2, (width, height))
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGBA2BGRA)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2BGRA)
    combined_img = cv2.addWeighted(img1, alpha, img2, beta, 0)
    return combined_img


# Generate and save markers
def generate_aruco_markers_on_one_image_with_ids(
          num_markers=10, 
          marker_size_cm=5, 
          dpi=300, 
          grid_rows=2, 
          grid_cols=5, 
          border_cm=1,
          save_format="pdf"):
    """
    Generates ArUco markers (ID 0-9) in a single image arranged in a grid with IDs labeled in the top-left
    corner inside the white border, and saves the result as a PDF.
    
    Args:
        marker_size_cm (int): Size of each marker in centimeters.
        dpi (int): Dots per inch for printing.
        grid_rows (int): Number of rows in the grid.
        grid_cols (int): Number of columns in the grid.
        border_cm (int): White border size around markers in centimeters.
    """
    # Calculate sizes in pixels
    cm_to_pixels = dpi / 2.54  # Conversion factor from cm to pixels
    marker_size_px = int(marker_size_cm * cm_to_pixels)
    border_px = int(border_cm * cm_to_pixels)
    
    # Create an ArUco dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    
    # Image size in pixels
    image_width = grid_cols * (marker_size_px + border_px) + border_px
    image_height = grid_rows * (marker_size_px + border_px) + border_px
    
    # Create a blank white image
    output_image = 255 * np.ones((image_height, image_width), dtype=np.uint8)
    
    marker_id = 0  # Start with ID 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if marker_id >= num_markers:  # We only need 10 markers
                break
            
            # Generate marker
            marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_px, borderBits=1)
            
            # Calculate position in the grid
            start_x = border_px + col * (marker_size_px + border_px)
            start_y = border_px + row * (marker_size_px + border_px)
            end_x = start_x + marker_size_px
            end_y = start_y + marker_size_px
            
            # Place the marker on the output image
            output_image[start_y:end_y, start_x:end_x] = marker_img
            
            # Add text for the marker ID in the top-left corner inside the white border
            text = f"ID: {marker_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7  # Enlarged text size
            font_thickness = 2  # Thicker text
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            
            # Calculate text position (inside the white border, left-top corner)
            text_x = start_x + int(border_px * 0.2)  # Slight padding from the left border
            text_y = start_y - int(border_px * 0.1)  # Slight padding from the top border
            
            # Add text to the image
            cv2.putText(output_image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
            
            marker_id += 1
    
    # Save the image as a PDF
    output_folder = "aruco_markers"
    os.makedirs(output_folder, exist_ok=True)
    if save_format == "pdf":
        output_path = os.path.join(output_folder, "aruco_markers_grid_with_ids.pdf")
        
        # Convert OpenCV image (NumPy array) to a Pillow Image
        pil_image = Image.fromarray(output_image)
        pil_image = pil_image.convert("RGB")
        pil_image.save(output_path, "PDF")
    else:
        output_path = os.path.join(output_folder, "aruco_markers_grid_with_ids.png")
        cv2.imwrite(output_path, output_image)
    
    print(f"ArUco markers with IDs saved as '{output_path}'")


# Configure logging
def set_logging_format(level=logging.INFO, simple=True):
  importlib.reload(logging)
  FORMAT = '[%(funcName)s] %(message)s' if simple else '%(asctime)s - %(levelname)s - %(message)s'
  logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    else:
        x = np.asarray(x)

    if len(x.shape) == 0:
        x = np.expand_dims(x, axis=0)
    return x
    

def replace_nan(x, value=0.0):
    """
    Replace NaN values in a numpy array with a specified value.
    
    :param x: Input numpy array
    :param value: Value to replace NaN with (default is 0.0)
    :return: Numpy array with NaN values replaced
    """
    x = np.asarray(x)  # Ensure x is a numpy array
    x[np.isnan(x)] = value
    return x


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Utils Functions')
    parser.add_argument('--task', type=str, default='d')
    parser.add_argument('--folder_name', type=str, help='folder name to delete')
    parser.add_argument('--earlier_all', action='store_true', help='delete all earlier folders')
    parser.add_argument('--task_name', type=str, default="FrankaCubeStack", help='task name')
    args = parser.parse_args()