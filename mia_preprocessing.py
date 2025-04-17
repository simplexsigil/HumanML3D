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

DO_VISU=False

if DO_VISU:
    from visu_utils import visualize_motion_trimesh

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


def mia_to_smpl_body(pose_dir, bm, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

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

    # Convert to tensors on the specified device
    pose_tensor = torch.tensor(pose_np_inter, dtype=torch.float32).to(device)
    betas_tensor = torch.tensor(betas_np_inter, dtype=torch.float32).to(device)
    transl_tensor = torch.tensor(transl_np_inter, dtype=torch.float32).to(device)

    # We rotate the model into the same orientation as the AMASS samples
    # Since HumanML3D is made for AMASS samples.

    # Assuming pose_tensor is already defined with shape (30, 3)
    orig_rotation = pose_tensor[:, :3].cpu().numpy()  # Shape (30,3), axis angle representation

    # Rotation to be applied
    # rotation_matrix = np.array([[-1.0, 0.0, 0.0], [0.0, -1, 0], [0.0, 0, 1]])
    # rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1, 0], [0.0, 0, 1]])
    rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # Convert axis-angle to rotation matrices
    orig_rot_matrices = R.from_rotvec(orig_rotation).as_matrix()  # Shape (30, 3, 3)

    # Apply the new rotation matrix
    new_rot_matrices = np.einsum("ij,kjl->kil", rotation_matrix, orig_rot_matrices)  # Shape (30, 3, 3)

    # Convert back to axis-angle representation if needed
    new_orientation = R.from_matrix(new_rot_matrices).as_rotvec()  # Shape (30, 3)

    new_orientation = torch.tensor(new_orientation).float().to(device)

    # Create SMPLH body model
    # Assume zero rotation for hands
    # 15 joints per hand * 3 rotations = 45
    left_hand_pose = torch.zeros((pose_tensor.shape[0], 45)).to(device)
    right_hand_pose = torch.zeros((pose_tensor.shape[0], 45)).to(device)

    # Initial body for ground height computation
    tmp_body = bm(
        betas=betas_tensor,
        body_pose=pose_tensor[:, 3:66],
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        global_orient=new_orientation,
        transl=transl_tensor,
    )

    # Compute per-frame lowest vertex (Y axis is up)
    verts_np = tmp_body.vertices.detach().cpu().numpy()
    min_y = verts_np.min(axis=(1, 2))  # (N,)
    transl_tensor[:, 1] -= torch.tensor(min_y).to(transl_tensor)

    # Final body with ground-adjusted translation
    body = bm(
        betas=betas_tensor,
        body_pose=pose_tensor[:, 3:66],
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        global_orient=new_orientation,
        transl=transl_tensor,
    )

    # body = bm(
    #    betas=betas_tensor,
    #    body_pose=pose_tensor[:, 3:66],
    #    left_hand_pose=left_hand_pose,
    #    right_hand_pose=right_hand_pose,
    #    global_orient=new_orientation,
    #    transl=transl_tensor,
    # )

    if DO_VISU:
        verts = body.vertices.detach().cpu().numpy()  # shape (N_frames, N_verts, 3)

        visualize_motion_trimesh(
            verts,
            bm.faces,
            title=str(os.path.split(pose_dir)[-1]),
            fps=10,
            # rot_matrix=rotation_matrix,
        )

    return body


def mia_to_pose(pose_dir, bm, device=None):
    """Convert MIA pose to joint positions and return as numpy array"""
    body = mia_to_smpl_body(pose_dir, bm, device)
    pose_seq_np = body.joints.detach().cpu().numpy()

    # Since we use the smplx package, we have a different implementation to the original HumanML3D scripts.
    # HML3D only uses the 52 kinematic tree joints from smplh, but smplx package implementation adds 21 extra joints from surface nodes.
    # such as eyes, ears, hand and feet positions on the skin.
    # We remove these extra joints to match the original implementation.
    pose_seq_np = pose_seq_np[:, :52]

    return pose_seq_np


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


def process_mia_data(root_dir, output_dir=None, save_intermediate=False, device=None):
    """Process MIA data and return joint positions

    Args:
        root_dir: Directory containing MIA data
        output_dir: Directory to save processed data (if save_intermediate is True)
        save_intermediate: Whether to save intermediate results to disk
        device: Device to run computations on ('cuda', 'cuda:0', 'cpu', etc.)

    Returns:
        Dictionary mapping sample paths to joint position arrays
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Initialize the SMPL-H body model
    smpl_h_path = "./body_models/smpl/SMPLH_NEUTRAL_AMASS_MERGED.pkl"
    # Each sample in MIA has 30 frames at 10 fps. We convert to 20 fps by interpolating intermediate frames
    # Since we only interpolate between two frames we end up with 59 frames as result.
    bm = SMPLH(model_path=smpl_h_path, num_betas=10, use_pca=False, batch_size=59).to(device)

    sample_dirs = get_leaf_directories(root_dir)
    results = {}

    for path in tqdm(sample_dirs, desc="Processing MIA data"):
        if save_intermediate and output_dir:
            save_path = path.replace(root_dir, output_dir) + ".npy"
            # Create the directories if they do not exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Process the pose data with the specified device
        pose_seq_np = mia_to_pose(path, bm, device)

        if save_intermediate and output_dir:
            np.save(save_path, pose_seq_np)

        # Store result in memory
        results[path] = pose_seq_np

        # Also generate mirrored version
        mirrored_data = swap_left_right(pose_seq_np)
        mirrored_key = f"M_{path}"
        results[mirrored_key] = mirrored_data

        if save_intermediate and output_dir:
            mirrored_save_path = os.path.dirname(save_path) + "/M_" + os.path.basename(save_path)
            np.save(mirrored_save_path, mirrored_data)

    return results


if __name__ == "__main__":
    # This is only for standalone testing
    import argparse

    parser = argparse.ArgumentParser(description="Process MIA motion data")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing MIA data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run computations on ('cuda', 'cuda:0', 'cpu', etc.)",
    )
    args = parser.parse_args()

    process_mia_data(args.input_dir, args.output_dir, save_intermediate=True, device=args.device)
