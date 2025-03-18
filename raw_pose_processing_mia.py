import sys, os
import torch
import numpy as np
import time
import glob

np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.unicode = np.unicode_
np.str = np.str_

from tqdm import tqdm
from os.path import join as opj
from smplx.body_models import SMPLH

from scipy.spatial.transform import Rotation as R, Slerp


def axis_angle_to_quaternion(axis_angle):
    """Convert axis-angle to quaternion."""
    rotation = R.from_rotvec(axis_angle)
    return rotation.as_quat()


def quaternion_to_axis_angle(quaternion):
    """Convert quaternion to axis-angle."""
    rotation = R.from_quat(quaternion)
    return rotation.as_rotvec()


def slerp(t, q0, q1):
    """Spherical linear interpolation (slerp) of quaternions."""
    rotations = R.from_quat([q0, q1])
    slerp = Slerp([0, 1], rotations)
    return slerp(t).as_quat()


def interpolate_poses(pose1, pose2, n_interpolations=1):
    """Interpolate between two sets of pose parameters (axis-angles)."""
    interpolated_poses = []
    for i in range(1, n_interpolations + 1):
        t = i / (n_interpolations + 1)
        interpolated_pose = []
        for j in range(len(pose1) // 3):
            aa1 = pose1[3 * j : 3 * j + 3]
            aa2 = pose2[3 * j : 3 * j + 3]
            q1 = axis_angle_to_quaternion(aa1)
            q2 = axis_angle_to_quaternion(aa2)
            qi = slerp(t, q1, q2)
            interpolated_pose.extend(quaternion_to_axis_angle(qi))
        interpolated_poses.append(interpolated_pose)
    return interpolated_poses


def interpolate_batch(batch, interpolation_func, n_interpolations=1):
    """Interpolate each pair of consecutive samples in a batch."""
    interpolated_batch = []
    for i in range(len(batch) - 1):
        interpolated_batch.append(batch[i])
        interpolated_intermediates = interpolation_func(batch[i], batch[i + 1], n_interpolations)
        interpolated_batch.extend(interpolated_intermediates)
    interpolated_batch.append(batch[-1])
    return np.array(interpolated_batch)


def interpolate_linear(param1, param2, n_interpolations=1):
    """Linear interpolation of parameters."""
    interpolated_params = []
    for i in range(1, n_interpolations + 1):
        t = i / (n_interpolations + 1)
        interpolated_param = (1 - t) * param1 + t * param2
        interpolated_params.append(interpolated_param)
    return interpolated_params


def convert_pare_to_full_img_cam(pare_cam, bbox_width, bbox_height, bbox_center, img_w, img_h, focal_length):
    # From https://github.com/mchiquier/musclesinaction/tree/main
    # Converts weak perspective camera estimated by PARE in
    # bbox coords to perspective camera in full image coordinates
    # from https://arxiv.org/pdf/2009.06549.pdf
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    res = 224
    tz = 2 * focal_length / (res * s)
    # pdb.set_trace()
    cx = 2 * (bbox_center[:, 0] - (img_w / 2.0)) / (s * bbox_width)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.0)) / (s * bbox_height)

    cam_t = np.stack([tx + cx, ty + cy, tz], axis=-1)

    return cam_t


def get_leaf_directories(root_dir):
    leaf_directories = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not dirnames:
            leaf_directories.append(dirpath)
    return leaf_directories


def mia_to_smpl_body(pose_dir, bm):
    pose_np = np.load(opj(pose_dir, "pose.npy"))
    betas_np = np.load(opj(pose_dir, "betas.npy"))
    predcam_np = np.load(opj(pose_dir, "predcam.npy"))
    bboxes_np = np.load(opj(pose_dir, "bboxes.npy"))

    bbox_center = np.stack((bboxes_np[:, 0] + bboxes_np[:, 2] / 2, bboxes_np[:, 1] + bboxes_np[:, 3] / 2), axis=-1)
    bbox_width = bboxes_np[:, 2]
    bbox_height = bboxes_np[:, 3]

    # These parameters result from the MIA settings, extracted from https://github.com/mchiquier/musclesinaction/tree/main
    transl_np = convert_pare_to_full_img_cam(
        pare_cam=predcam_np,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        bbox_center=bbox_center,
        img_w=1920,
        img_h=1080,
        focal_length=5000,
    )

    transl_np = transl_np - transl_np[0]

    # Vibe depth estimation is not really good, but in Muscles in Action most actions (roughly) happen in a plane
    # So we can set the z translation to 0
    transl_np[:, 2] = 0

    # Interpolate in bnetween to get 59 pose samples (20 fps instead of 10 fps)
    pose_np_inter = interpolate_batch(pose_np, interpolate_poses, n_interpolations=1)
    betas_np_inter = interpolate_batch(betas_np, interpolate_linear, n_interpolations=1)
    transl_np_inter = interpolate_batch(transl_np, interpolate_linear, n_interpolations=1)

    # Ensure the shapes are correct for the SMPL model
    assert pose_np.shape[1] == 72, "Each pose should have 72 parameters (24 joints * 3 rotations)."
    assert betas_np.shape[1] == 10, "Each betas should have 10 parameters (shape coefficients)."

    pose_tensor = torch.tensor(pose_np_inter, dtype=torch.float32).cuda()
    betas_tensor = torch.tensor(betas_np_inter, dtype=torch.float32).cuda()
    transl_tensor = torch.tensor(transl_np_inter, dtype=torch.float32).cuda()

    # We rotate the model into the same orientation as the AMASS samples
    # Since HumanML3D is made for AMASS samples.

    # Assuming pose_tensor is already defined with shape (30, 3)
    orig_rotation = pose_tensor[:, :3].cpu().numpy()  # Shape (30,3), axis angle representation

    # Rotation to be applied
    rotation_matrix = np.array([[-1.0, 0.0, 0.0], [0.0, -1, 0], [0.0, 0, 1]])

    # Convert axis-angle to rotation matrices
    orig_rot_matrices = R.from_rotvec(orig_rotation).as_matrix()  # Shape (30, 3, 3)

    # Apply the new rotation matrix
    new_rot_matrices = np.einsum("ij,kjl->kil", rotation_matrix, orig_rot_matrices)  # Shape (30, 3, 3)

    # Convert back to axis-angle representation if needed
    new_orientation = R.from_matrix(new_rot_matrices).as_rotvec()  # Shape (30, 3)

    new_orientation = torch.tensor(new_orientation).float().cuda()

    # Create SMPLH body model
    # Assume zero rotation for hands
    # 15 joints per hand * 3 rotations = 45
    left_hand_pose = torch.zeros((pose_tensor.shape[0], 45)).cuda()
    right_hand_pose = torch.zeros((pose_tensor.shape[0], 45)).cuda()

    body = bm(
        betas=betas_tensor,
        body_pose=pose_tensor[:, 3:66],
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        global_orient=new_orientation,
        transl=transl_tensor,
    )

    return body


def mia_to_pose(pose_dir, bm, save_path):
    body = mia_to_smpl_body(pose_dir, bm)

    pose_seq_np = body.joints.detach().cpu().numpy()

    # Since we use the smplx package, we have a different implementation to the original HumanML3D scripts.
    # HML3D only uses the 52 kinematic tree joints from smplh, but smplx package implementation adds 21 extra joints from surface nodes.
    # such as eyes, ears, hand and feet positions on the skin.
    # We remove these extra joints to match the original implementation.
    pose_seq_np = pose_seq_np[:, :52]

    np.save(save_path, pose_seq_np)


def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


smpl_h_path = "./body_models/smpl/SMPLH_NEUTRAL_AMASS_MERGED.pkl"

# Each sample in MIA has 30 framesa at 10 fps. We convert to 20 fps by interpolating intermediate frames
# Since we only interpolate between two frames we end up with 59 frames as result.
bm = SMPLH(model_path=smpl_h_path, num_betas=10, use_pca=False, batch_size=59).cuda()

root_dir = "/cvhci/data/activity/MIADatasetOfficial"
pose_data_dir = "./MIAHML3D/pose_data"

sample_dirs = get_leaf_directories(root_dir)

print(sample_dirs[:3])


for path in tqdm(sample_dirs):
    save_path = path.replace(root_dir, pose_data_dir)
    save_path = save_path + ".npy"

    # Get the directory path
    dire = os.path.dirname(save_path)

    # Create the directories if they do not exist
    os.makedirs(dire, exist_ok=True)

    mia_to_pose(path, bm, save_path)


pose_data_dir = "MIAHML3D/pose_data"
save_dir = "MIAHML3D/joints"

# Get a list of all .npy files in the directory tree
pose_samples = glob.glob(os.path.join(pose_data_dir, "**", "*.npy"), recursive=True)


for pose_path in tqdm(pose_samples):
    dire, file = os.path.split(pose_path)

    dire = dire.replace(pose_data_dir, save_dir)

    save_filepath = opj(dire, file)
    save_filepath_m = opj(dire, "M_" + file)

    # Create the directories if they do not exist
    os.makedirs(dire, exist_ok=True)

    data = np.load(pose_path)
    data_m = swap_left_right(data)

    np.save(save_filepath, data)
    np.save(save_filepath_m, data_m)
