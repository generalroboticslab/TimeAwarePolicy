"""
Numpy version of transformation functions originally implemented using PyTorch.
This script implements quaternion multiplication, rotation, Euler conversions,
scaling/denormalization and more. All functions assume that the inputs are NumPy arrays.
We really need a unit test to compare the results with torch_jit_utils.py!!
"""

import numpy as np
from scipy.spatial.transform import Rotation


def to_array(x, dtype=None):
    """Converts x to a NumPy array."""
    return np.array(x, dtype=dtype)


def normalize(x, eps=1e-9):
    """Normalizes x along its last axis using the L2 norm."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def quat_mul(a, b):
    """
    Quaternion multiplication.
    
    Both a and b are assumed to have shape (..., 4) with quaternion order: [qx, qy, qz, qw].
    """
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
    """
    Apply a rotation (represented as a quaternion) to a vector.
    
    a: quaternion(s), shape (..., 4)
    b: vector(s), shape (..., 3)
    """
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    # t = 2 * cross(xyz, b)
    t = np.cross(xyz, b, axis=-1) * 2
    result = b + a[:, 3:4] * t + np.cross(xyz, t, axis=-1)
    return result.reshape(shape)


def quat_rotate(q, v):
    """
    Rotates vectors v by quaternion q.
    
    q: shape (N, 4); v: shape (N, 3)
    Assumes quaternion order [qx, qy, qz, qw].
    """
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * (q_w ** 2) - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v, axis=-1) * (q_w[:, np.newaxis] * 2.0)
    dot = np.sum(q_vec * v, axis=-1)
    c = q_vec * (dot[:, np.newaxis] * 2.0)
    return a + b + c


def quat_rotate_inverse(q, v):
    """
    Rotates vectors v by the inverse of quaternion q.
    The only difference from quat_rotate is the sign on the term involving the cross product.
    """
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * (q_w ** 2) - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v, axis=-1) * (q_w[:, np.newaxis] * 2.0)
    dot = np.sum(q_vec * v, axis=-1)
    c = q_vec * (dot[:, np.newaxis] * 2.0)
    return a - b + c


def quat_conjugate(a):
    """
    Returns the conjugate of quaternion(s) a.
    """
    shape = a.shape
    a = a.reshape(-1, 4)
    result = np.concatenate([-a[:, :3], a[:, 3:4]], axis=-1)
    return result.reshape(shape)


def quat_unit(a):
    """Normalizes the quaternion(s) to unit length."""
    return normalize(a)


def quat_from_angle_axis(angle, axis):
    """
    Creates a quaternion from a given angle and axis.
    
    angle: in radians; axis: vector of shape (..., 3)
    """
    theta = angle / 2.0
    normalized_axis = normalize(axis)
    xyz = normalized_axis * np.sin(theta)[..., np.newaxis]
    w = np.cos(theta)
    q = np.concatenate([xyz, w[..., np.newaxis]], axis=-1)
    return quat_unit(q)


def normalize_angle(x):
    """Wraps angle to (–pi, pi] using the atan2(sin, cos) trick."""
    return np.arctan2(np.sin(x), np.cos(x))


def tf_inverse(q, t):
    """
    Returns the inverse transform:
      Inverts the rotation (quaternion conjugate) and then undoes the translation.
    """
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


def tf_apply(q, t, v):
    """
    Applies a full transform (rotation and translation) to vector v.
    """
    return quat_apply(q, v) + t


def tf_vector(q, v):
    """Applies only the rotational part of the transform."""
    return quat_apply(q, v)


def tf_combine(q1, t1, q2, t2):
    """
    Combines two transforms; where rotations are multiplied and translations are combined.
    """
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


def get_basis_vector(q, v):
    """Returns the basis vector v rotated by quaternion q."""
    return quat_rotate(q, v)


def get_axis_params(value, axis_idx, x_value=0., dtype=float, n_dims=3):
    """
    Constructs a list of parameters according to an axis index.
    
    Example: get_axis_params(value=5, axis_idx=1, x_value=0, n_dims=3)
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "The axis index is outside the vector dimensions."
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


def copysign(a, b):
    """
    Returns an array with the magnitude of a and the sign of b.
    """
    return np.abs(a) * np.sign(b)


def get_euler_xyz(q):
    """
    Converts quaternion(s) in [qx, qy, qz, qw] format into Euler angles (roll, pitch, yaw).
    
    Returns Euler angles wrapped to the range [0, 2*pi).
    """
    qx, qy, qz, qw = 0, 1, 2, 3

    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw]**2 - q[:, qx]**2 - q[:, qy]**2 + q[:, qz]**2
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = np.where(np.abs(sinp) >= 1,
                     np.copysign(np.pi / 2.0, sinp),
                     np.arcsin(sinp))

    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw]**2 + q[:, qx]**2 - q[:, qy]**2 - q[:, qz]**2
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.mod(roll, 2*np.pi), np.mod(pitch, 2*np.pi), np.mod(yaw, 2*np.pi)


def euler_from_quat(q):
    """
    Converts quaternion(s) to a single stacked Euler angle array.

    If q is a 1-D array (single quaternion), returns a 1-D Euler vector.
    Otherwise, returns an array of shape (N, 3).
    """
    original_ndim = q.ndim
    if q.ndim == 1:
        q = np.expand_dims(q, axis=0)
    roll, pitch, yaw = get_euler_xyz(q)
    euler = np.stack([roll, pitch, yaw], axis=-1)
    if original_ndim == 1:
        euler = np.squeeze(euler, axis=0)
    return euler


def transform_euler(euler):
    """Transforms Euler angles to be within the range [–pi, pi)."""
    return (euler + np.pi) % (2 * np.pi) - np.pi


def quat_from_euler_xyz(roll, pitch, yaw):
    """
    Constructs a quaternion from Euler angles (roll, pitch, yaw).
    Returns a quaternion in [qx, qy, qz, qw] form.
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.stack([qx, qy, qz, qw], axis=-1)


def np_rand_float(lower, upper, shape):
    """Returns random floats uniformly sampled from [lower, upper)."""
    return (upper - lower) * np.random.rand(*shape) + lower


def np_random_dir_2(shape):
    """
    Generates 2D random unit vectors.
    
    Expects shape to be a tuple; the last dimension is squeezed.
    """
    angle = np_rand_float(-np.pi, np.pi, shape).squeeze(-1)
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def tensor_clamp(t, min_t, max_t):
    """Clamps (limits) the values in t to be within [min_t, max_t]."""
    return np.clip(t, min_t, max_t)


def scale(x, lower, upper):
    """Scales a value from [-1, 1] to [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def unscale(x, lower, upper):
    """Inverse of scale()."""
    return (2.0 * x - upper - lower) / (upper - lower)


def unscale_np(x, lower, upper):
    """Alias for unscale; provided for compatibility."""
    return (2.0 * x - upper - lower) / (upper - lower)


def compute_heading_and_up(torso_rotation, inv_start_rot, to_target, vec0, vec1, up_idx):
    """
    Computes basis vectors for heading and "up" directions.
    
    Returns: (torso_quat, up_proj, heading_proj, up_vec, heading_vec)
    """
    num_envs = torso_rotation.shape[0]
    target_dirs = normalize(to_target)

    torso_quat = quat_mul(torso_rotation, inv_start_rot)
    up_vec = get_basis_vector(torso_quat, vec1).reshape((num_envs, 3))
    heading_vec = get_basis_vector(torso_quat, vec0).reshape((num_envs, 3))
    up_proj = up_vec[:, up_idx]
    heading_proj = np.sum(heading_vec * target_dirs, axis=1)

    return torso_quat, up_proj, heading_proj, up_vec, heading_vec


def compute_rot(torso_quat, velocity, ang_velocity, targets, torso_positions):
    """
    Computes the local velocities, angular velocities and Euler angles.
    
    Returns: (vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target)
    """
    vel_loc = quat_rotate_inverse(torso_quat, velocity)
    angvel_loc = quat_rotate_inverse(torso_quat, ang_velocity)
    roll, pitch, yaw = get_euler_xyz(torso_quat)
    walk_target_angle = np.arctan2(targets[:, 2] - torso_positions[:, 2],
                                   targets[:, 0] - torso_positions[:, 0])
    angle_to_target = walk_target_angle - yaw
    return vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target


def quat_axis(q, axis=0):
    """
    Returns the direction of the given axis after rotation by q.
    """
    basis_vec = np.zeros((q.shape[0], 3))
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def scale_transform(x, lower, upper):
    """
    Normalizes x to the range [-1, 1].
    """
    offset = (lower + upper) * 0.5
    return 2 * (x - offset) / (upper - lower)


def unscale_transform(x, lower, upper):
    """
    Denormalizes a value from [-1, 1] back to [lower, upper].
    """
    offset = (lower + upper) * 0.5
    return x * (upper - lower) * 0.5 + offset


def saturate(x, lower, upper):
    """Clamps x to be within [lower, upper]."""
    return np.clip(x, lower, upper)


def quat_diff_rad(a, b):
    """
    Computes the angular difference (in radians) between two quaternions.
    """
    b_conj = quat_conjugate(b)
    mul = quat_mul(a, b_conj)
    norm_val = np.linalg.norm(mul[:, :3], axis=-1)
    norm_val = np.clip(norm_val, 0, 1.0)
    return 2.0 * np.arcsin(norm_val)


def local_to_world_space(pos_offset_local, pose_global):
    """
    Converts a point from the local frame to the global frame.
    
    pos_offset_local: (N, 3)
    pose_global: (N, 7) with position (0:3) and quaternion (3:7)
    """
    N = pos_offset_local.shape[0]
    quat_pos_local = np.concatenate([pos_offset_local,
                                     np.zeros((N, 1), dtype=pos_offset_local.dtype)], axis=-1)
    quat_global = pose_global[:, 3:7]
    quat_global_conj = quat_conjugate(quat_global)
    pos_offset_global = quat_mul(quat_global, quat_mul(quat_pos_local, quat_global_conj))[:, :3]
    result_pos_global = pos_offset_global + pose_global[:, :3]
    return result_pos_global


def normalise_quat_in_pose(pose):
    """
    Normalises the quaternion portion of a pose.
    
    pose: (N, 7); Returns a pose with the quaternion (indices 3:7) normalized.
    """
    pos = pose[:, :3]
    quat = pose[:, 3:7]
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    return np.concatenate([pos, quat], axis=-1)


def my_quat_rotate(q, v):
    """
    Alternate quaternion rotation, equivalent to quat_rotate.
    """
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * (q_w ** 2) - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v, axis=-1) * (q_w[:, np.newaxis] * 2.0)
    dot = np.sum(q_vec * v, axis=-1)
    c = q_vec * (dot[:, np.newaxis] * 2.0)
    return a + b + c


def quat_to_angle_axis(q):
    """
    Converts a normalized quaternion to its axis-angle representation.
    
    Returns: (angle, axis)
    """
    min_theta = 1e-5
    qw = q[..., 3]
    sin_theta = np.sqrt(np.clip(1 - qw**2, 0, None))
    angle = 2 * np.arccos(qw)
    angle = normalize_angle(angle)
    sin_theta_expand = np.expand_dims(sin_theta, axis=-1)
    axis = q[..., :3] / sin_theta_expand
    mask = sin_theta > min_theta
    default_axis = np.zeros_like(axis)
    default_axis[..., -1] = 1
    angle = np.where(mask, angle, 0.0)
    axis = np.where(np.expand_dims(mask, axis=-1), axis, default_axis)
    return angle, axis


def angle_axis_to_exp_map(angle, axis):
    """Computes an exponential map from an axis-angle pair."""
    return np.expand_dims(angle, axis=-1) * axis


def quat_to_exp_map(q):
    """Computes the exponential map from a normalized quaternion."""
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map


def quaternion_to_matrix(quaternions):
    """
    Converts quaternion(s) (in [qx, qy, qz, qw] form) to rotation matrices.
    The function reorders the quaternion into [r, i, j, k] (with r = qw).
    
    Returns a rotation matrix of shape (..., 3, 3).
    """
    # Reorder from [qx, qy, qz, qw] to [r, i, j, k]
    q = quaternions
    qw = q[..., 3]
    qx = q[..., 0]
    qy = q[..., 1]
    qz = q[..., 2]
    r, i, j, k = qw, qx, qy, qz

    two_s = 2.0 / np.sum(q * q, axis=-1)
    m00 = 1 - two_s * (j*j + k*k)
    m01 = two_s * (i*j - k*r)
    m02 = two_s * (i*k + j*r)
    m10 = two_s * (i*j + k*r)
    m11 = 1 - two_s * (i*i + k*k)
    m12 = two_s * (j*k - i*r)
    m20 = two_s * (i*k - j*r)
    m21 = two_s * (j*k + i*r)
    m22 = 1 - two_s * (i*i + j*j)
    shape = quaternions.shape[:-1]
    M = np.stack([m00, m01, m02,
                  m10, m11, m12,
                  m20, m21, m22], axis=-1)
    M = M.reshape(shape + (3, 3))
    return M


def _sqrt_positive_part(x):
    """
    Returns np.sqrt(x) for positive values of x; zero otherwise.
    """
    ret = np.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = np.sqrt(x[positive_mask])
    return ret


# def matrix_to_quaternion(matrix):
#     """
#     Converts rotation matrices (shape (..., 3, 3)) to quaternions in [qx, qy, qz, qw] form.
    
#     This implementation follows the procedure of constructing candidate quaternions and
#     then selecting the one corresponding to the maximum of the computed intermediate values.
#     """
#     if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
#         raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

#     # Extract matrix elements.
#     m00 = matrix[..., 0, 0]
#     m01 = matrix[..., 0, 1]
#     m02 = matrix[..., 0, 2]
#     m10 = matrix[..., 1, 0]
#     m11 = matrix[..., 1, 1]
#     m12 = matrix[..., 1, 2]
#     m20 = matrix[..., 2, 0]
#     m21 = matrix[..., 2, 1]
#     m22 = matrix[..., 2, 2]

#     q_abs = _sqrt_positive_part(np.stack([1.0 + m00 + m11 + m22,
#                                           1.0 + m00 - m11 - m22,
#                                           1.0 - m00 + m11 - m22,
#                                           1.0 - m00 - m11 + m22], axis=-1))

#     q0 = np.stack([q_abs[..., 0]**2, m21 - m12, m02 - m20, m10 - m01], axis=-1)
#     q1 = np.stack([m21 - m12, q_abs[..., 1]**2, m10 + m01, m02 + m20], axis=-1)
#     q2 = np.stack([m02 - m20, m10 + m01, q_abs[..., 2]**2, m12 + m21], axis=-1)
#     q3 = np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3]**2], axis=-1)
#     quat_by_rijk = np.stack([q0, q1, q2, q3], axis=-2)

#     flr = 0.1
#     denom = 2.0 * np.maximum(q_abs[..., np.newaxis], flr)
#     quat_candidates = quat_by_rijk / denom

#     # Select candidate corresponding to the maximum element of q_abs.
#     idx = np.argmax(q_abs, axis=-1)
#     idx_expanded = idx[..., np.newaxis, np.newaxis]
#     quat_selected = np.take_along_axis(quat_candidates, idx_expanded, axis=-2)
#     quat_selected = np.squeeze(quat_selected, axis=-2)
#     return quat_selected


def matrix_to_quaternion(rot_matrix):
    """
    Only support single rotation matrix now.
    Convert a rotation matrix into a quaternion.
    
    Args:
    rot_matrix (np.array): A 3x3 rotation matrix.
    
    Returns:
    np.array: A numpy array with four elements [x, y, z, w], where w is the scalar 
              component, and x, y, z are the vector components of the quaternion.
    """
    m = np.array(rot_matrix, dtype=np.float64, copy=False)
    q = np.empty((4,))
    t = np.trace(m)
    if t > np.finfo(np.float64).eps:
        t = np.sqrt(t + 1.0)
        q[3] = 0.5 * t  # w component
        t = 0.5 / t
        q[0] = (m[2, 1] - m[1, 2]) * t  # x component
        q[1] = (m[0, 2] - m[2, 0]) * t  # y component
        q[2] = (m[1, 0] - m[0, 1]) * t  # z component
    else:
        i = np.argmax(np.diagonal(m))
        j = (i + 1) % 3
        k = (i + 2) % 3
        t = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0)
        q[i] = 0.5 * t  # x, y, or z component based on which diagonal is largest
        t = 0.5 / t
        q[3] = (m[k, j] - m[j, k]) * t  # w component
        q[j] = (m[j, i] + m[i, j]) * t  # Next component in order
        q[k] = (m[k, i] + m[i, k]) * t  # Last component in order
    return q


def quat_to_tan_norm(q):
    """
    Represents a rotation using the tangent and normal vectors.
    Returns the concatenation of the rotated [1, 0, 0] (tan) and [0, 0, 1] (norm) vectors.
    """
    ref_tan = np.zeros_like(q[..., :3])
    ref_tan[..., 0] = 1
    tan = my_quat_rotate(q, ref_tan)

    ref_norm = np.zeros_like(q[..., :3])
    ref_norm[..., -1] = 1
    norm = my_quat_rotate(q, ref_norm)

    norm_tan = np.concatenate([tan, norm], axis=-1)
    return norm_tan


def euler_xyz_to_exp_map(roll, pitch, yaw):
    """
    Converts Euler angles (roll, pitch, yaw) to an exponential map representation.
    """
    q = quat_from_euler_xyz(roll, pitch, yaw)
    exp_map = quat_to_exp_map(q)
    return exp_map


def exp_map_to_angle_axis(exp_map):
    """
    Converts an exponential map to an axis-angle representation.
    
    Returns: (angle, axis)
    """
    min_theta = 1e-5
    angle = np.linalg.norm(exp_map, axis=-1)
    angle = normalize_angle(angle)
    default_axis = np.zeros_like(exp_map)
    default_axis[..., -1] = 1
    angle_exp = np.expand_dims(angle, axis=-1)
    axis = np.where(angle_exp > 1e-8, exp_map / angle_exp, default_axis)
    mask = angle > min_theta
    angle = np.where(mask, angle, 0.0)
    axis = np.where(np.expand_dims(mask, axis=-1), axis, default_axis)
    return angle, axis


def exp_map_to_quat(exp_map):
    """Converts an exponential map to a quaternion."""
    angle, axis = exp_map_to_angle_axis(exp_map)
    q = quat_from_angle_axis(angle, axis)
    return q


def slerp(q0, q1, t):
    """
    Spherical linear interpolation (slerp) between two quaternions q0 and q1.
    
    t can be a scalar or an array broadcastable to the shape of the angles.
    """
    cos_half_theta = (q0[..., 3] * q1[..., 3] +
                      q0[..., 0] * q1[..., 0] +
                      q0[..., 1] * q1[..., 1] +
                      q0[..., 2] * q1[..., 2])
    neg_mask = cos_half_theta < 0
    q1 = np.where(neg_mask[..., np.newaxis], -q1, q1)
    cos_half_theta = np.abs(cos_half_theta)
    cos_half_theta_exp = cos_half_theta[..., np.newaxis]
    half_theta = np.arccos(cos_half_theta)
    sin_half_theta = np.sqrt(1.0 - cos_half_theta**2)[..., np.newaxis]

    t = np.asarray(t)
    if t.ndim == 0:
        t = t * np.ones_like(half_theta)
    ratioA = np.sin((1 - t) * half_theta)[..., np.newaxis] / sin_half_theta
    ratioB = np.sin(t * half_theta)[..., np.newaxis] / sin_half_theta
    new_q = ratioA * q0 + ratioB * q1

    mask_sin = (np.squeeze(np.abs(sin_half_theta), axis=-1) < 0.001)
    mask_cos = (np.squeeze(np.abs(cos_half_theta_exp), axis=-1) >= 1)
    new_q = np.where(mask_sin[..., np.newaxis], 0.5 * q0 + 0.5 * q1, new_q)
    new_q = np.where(mask_cos[..., np.newaxis], q0, new_q)
    return new_q


def calc_heading(q):
    """
    Calculates the heading direction (angle on the xy plane) from a quaternion.
    """
    ref_dir = np.zeros_like(q[..., :3])
    ref_dir[..., 0] = 1
    rot_dir = my_quat_rotate(q, ref_dir)
    heading = np.arctan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading


def calc_heading_quat(q):
    """
    Derives the heading rotation from quaternion q.
    """
    heading = calc_heading(q)
    axis = np.zeros_like(q[..., :3])
    axis[..., 2] = 1
    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q


def calc_heading_quat_inv(q):
    """
    Calculates the inverse heading rotation from quaternion q.
    """
    heading = calc_heading(q)
    axis = np.zeros_like(q[..., :3])
    axis[..., 2] = 1
    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q


def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.
    Args:
        quat (ndarray): (..., 4) array where the final dim is (x,y,z,w) quaternion
    Returns:
        ndarray: (..., 3) axis-angle exponential coordinates
    """
    quat = quat.astype(float) # ensure float number not int
    # reshape quaternion
    quat_shape = quat.shape[:-1]      # ignore last dim
    quat = quat.reshape(-1, 4)
    # clip quaternion
    quat[:, 3] = np.clip(quat[:, 3], -1., 1.)
    # Calculate denominator
    den = np.sqrt(1. - quat[:,3] * quat[:,3])
    
    # Create return array
    ret = np.zeros_like(quat)[:, :3]
    idx = np.nonzero(den)[0]  # NumPy nonzero returns a tuple of arrays
    
    # Calculate for non-zero denominator cases
    angle = np.arccos(quat[idx, 3])
    ret[idx, :] = (quat[idx, :3] * 2.0 * angle.reshape(-1, 1)) / den[idx].reshape(-1, 1)

    # Reshape and return output
    ret = ret.reshape(list(quat_shape) + [3])
    return ret


def axisangle2quat(vec, eps=1e-6):
    """
    Converts an axis-angle vector (exponential map) to a quaternion.
    
    vec: Array of shape (..., 3).
    """
    input_shape = vec.shape[:-1]
    vec_flat = vec.reshape(-1, 3)
    angle = np.linalg.norm(vec_flat, axis=-1, keepdims=True)
    quat = np.zeros((np.prod(input_shape, dtype=int), 4), dtype=vec.dtype)
    quat[:, 3] = 1.0
    idx = (angle.reshape(-1) > eps)
    if np.any(idx):
        sin_half = np.sin(angle[idx] / 2.0)
        quat[idx, :3] = vec_flat[idx] * (sin_half / angle[idx])
        quat[idx, 3] = np.cos(angle[idx] / 2.0)
    quat = quat.reshape(list(input_shape) + [4])
    return quat


def quaternion_slerp(q0, q1, fraction, DOT_THRESHOLD=0.9995):
    """
    Spherical linear interpolation (slerp) of two quaternions in (x, y, z, w) order.
    Assumes both q0 and q1 are normalized.
    """
    q0 = q0.copy()
    q1 = q1.copy()
    dot = np.dot(q0, q1)
    
    # Ensure the shortest path is taken.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > DOT_THRESHOLD:
        # If the quaternions are nearly identical, use linear interpolation.
        result = q0 + fraction * (q1 - q0)
        return result / np.linalg.norm(result)
    
    theta_0 = np.arccos(dot)  # Initial angle.
    theta = theta_0 * fraction
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    result = q0 * np.cos(theta) + q2 * np.sin(theta)
    return result / np.linalg.norm(result)


def quaternion_distance(q1_array: np.ndarray, q2_array: np.ndarray) -> np.ndarray:
    """
    Calculate quaternion distances for multiple quaternion pairs.
    
    Args:
        q1_array: Array of quaternions (shape: (N, 4))
        q2_array: Array of quaternions (shape: (N, 4))
        
    Returns:
        Array of quaternion distances (shape: (N,))
    """
    # Normalize quaternions (if not already normalized)
    q1_norm = q1_array / np.linalg.norm(q1_array, axis=1, keepdims=True)
    q2_norm = q2_array / np.linalg.norm(q2_array, axis=1, keepdims=True)
    
    # Compute dot product for each pair
    dot_products = np.sum(q1_norm * q2_norm, axis=1)
    
    # Clamp to [-1, 1] to avoid numerical errors with arccos
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # Calculate angular distance
    distances = 2 * np.arccos(dot_products)
    
    return distances


def adaptive_filter_pose(new_pose, prev_pose, 
                         trans_alpha=0.95, rot_alpha=0.1,
                         trans_thresh=0.05, rot_thresh=np.deg2rad(5)):
    """
    Applies adaptive low-pass filtering to a pose based on change thresholds.
    Filtering is applied independently to translation and rotation when their
    respective changes exceed the thresholds.
    
    Parameters:
    -----------
    new_pose : tuple
        New pose as (quat, trans) where quat is (x, y, z, w)
    prev_pose : tuple
        Previous pose as (quat, trans) where quat is (x, y, z, w)
    trans_alpha : float
        Translation smoothing factor (0-1). Lower = more smoothing
    rot_alpha : float
        Rotation smoothing factor (0-1). Lower = more smoothing
    trans_thresh : float
        Translation change threshold to trigger filtering
    rot_thresh : float
        Rotation change threshold (radians) to trigger filtering
    
    Returns:
    --------
    tuple
        Filtered pose as (quat, trans)
    """
    new_quat, new_trans = new_pose
    prev_quat, prev_trans = prev_pose
    
    # Check translation difference
    trans_diff = np.linalg.norm(new_trans - prev_trans)
    # Apply filtering to translation if threshold exceeded
    trans_filtered = new_trans
    if trans_diff > trans_thresh:
        trans_filtered = (1 - trans_alpha) * prev_trans + trans_alpha * new_trans
    
    # Check rotation difference
    dot = np.clip(np.abs(np.dot(new_quat, prev_quat)), 0.0, 1.0)
    rot_diff = 2 * np.arccos(dot)
    # Apply filtering to rotation if threshold exceeded
    quat_filtered = new_quat
    if rot_diff > rot_thresh:
        quat_filtered = quaternion_slerp(prev_quat, new_quat, rot_alpha)
        quat_filtered = quat_filtered / np.linalg.norm(quat_filtered)
    
    return (quat_filtered, trans_filtered)


# EOF

if __name__ == "__main__":
    # Test the tf_combine function. It sometimes not consistent with the exactly same inputs.
    # pose1 = [np.array([0.26642532, 0.03420802, 0.60059559], dtype=np.float16), np.array([0, 0, 0, 1])]
    # for i in range(100):
    #     pose_out = tf_combine(pose1[1], pose1[0], pose1[1], pose1[0])
    #     print(pose_out)

    print(axisangle2quat(np.array([0, 0, np.pi])))