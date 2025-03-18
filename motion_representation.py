from os.path import join as pjoin

from common.skeleton import Skeleton
import numpy as np
import os
from common.quaternion import *
from paramUtil import *

import torch
from tqdm import tqdm
import os


import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3


def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=2):
    title_sp = title.split(" ")
    if len(title_sp) > 10:
        title = "\n".join([" ".join(title_sp[:10]), " ".join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius])
        ax.set_zlim3d([-radius / 2, radius])
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = [
        "red",
        "blue",
        "black",
        "red",
        "blue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkred",
        "darkred",
        "darkred",
        "darkred",
        "darkred",
    ]
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    # data[..., 0] -= data[:, 0:1, 0]
    # data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        ax.cla()
        init()
        ax.view_init(elev=20, azim=210, roll=0, vertical_axis="y")
        ax.dist = 7.5
        plot_xzPlane(
            MINS[0],  # - trajec[index, 0],
            MAXS[0],  # - trajec[index, 0],
            0,
            MINS[2],  # - trajec[index, 1],
            MAXS[2],  # - trajec[index, 1],
        )

        if index > 1:
            ax.plot3D(
                trajec[:index, 0],  # - trajec[index, 0],
                np.zeros_like(trajec[:index, 0]),
                trajec[:index, 1],  # - trajec[index, 1],
                linewidth=1.0,
                color="blue",
            )

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(
                data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color
            )

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()


# This function scales the skeleton from source to target dimensions by comparing leg lengths.
def uniform_skeleton(positions, target_offset):
    """
    This function adjusts a source skeleton to match the dimensions of a target skeleton based on leg lengths.
    It uses the initial frame of motion data to calculate a scaling ratio between the leg lengths of the
    source and target skeletons. It then scales the root joint positions accordingly and applies inverse
    kinematics to find the appropriate joint rotations that conform to the scaled skeleton. Finally, it uses
    forward kinematics to compute the new joint positions in the target skeleton configuration.

    Parameters:
    - positions (numpy.ndarray): The joint positions of the source skeleton across frames.
    - target_offset (torch.Tensor): The joint offsets of the target skeleton.

    Returns:
    - new_joints (numpy.ndarray): The adjusted joint positions conforming to the target skeleton dimensions.
    """

    # Initialize a skeleton instance with raw joint offsets and kinematic chain for calculations
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")

    # Extract the first frame's joint offsets to use as the source skeleton reference
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    # Calculate the scaling ratio based on leg length differences between source and target
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len

    # Scale the root position of the source skeleton to match the target's size.
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    # Perform inverse kinematics to calculate joint rotations
    # Compute rotational parameters through inverse kinematics on source positions
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)

    # Set target skeleton offsets and calculate new joint positions using forward kinematics
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)

    return new_joints  # Return the new joint positions in target skeleton configuration


# Define a function to process each file, adjusting positions and orientations
def process_file(positions, feet_thre):
    """
    Processes a file of joint positions to align the motion with a target skeleton, set the animation
    to start at the origin with the character facing forward, and detect foot contacts based on
    velocity thresholds. This function first normalizes the skeleton dimensions using `uniform_skeleton`,
    aligns the root joint with the floor and the origin, rotates the entire motion to face the Z+ direction,
    and detects contacts for each foot based on predefined thresholds. It applies quaternion operations to
    adjust orientations and calculates forward kinematics to determine new joint positions.

    Parameters:
    - positions (numpy.ndarray): The joint positions across all frames.
    - feet_thre (float): Threshold below which velocity indicates a foot contact.

    Returns:
    - data (numpy.ndarray): The processed motion data including joint positions, rotations, velocities, and foot contacts.
    - global_positions (numpy.ndarray): The final positions of joints after all transformations.
    - positions (numpy.ndarray): Transformed joint positions for each frame.
    - l_velocity (numpy.ndarray): Linear velocities of the root joint.
    """

    # plot_3d_motion("./positions_non_uniform.mp4", kinematic_chain, positions, "title", fps=20)

    # Normalize the skeleton of the motion to match the target skeleton
    positions = uniform_skeleton(positions, tgt_offsets)

    # plot_3d_motion("./positions_uniform.mp4", kinematic_chain, positions, "title", fps=20)

    # Set the lowest point of the motion to the floor (height 0)
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # Ensure the root position's x and z coordinates start at the origin
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # Align character to face the Z+ direction initially using the hips and shoulders
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    # Calculate the forward direction and the initial root quaternion
    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)  # Calculate forward direction
    forward_init = forward_init / np.sqrt((forward_init**2).sum(axis=-1))[..., np.newaxis]  # Normalize

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)  # Calculate quaternion between initial and target forward
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init  # Broadcast quaternion across all positions

    # Rotate the positions to align them to the new direction
    positions_b = positions.copy()  # Backup positions before rotation
    positions = qrot_np(root_quat_init, positions)  # Apply quaternion rotation to align all frames forward

    # Save the new global positions
    global_positions = positions.copy()  # Copy of the final positions after transformations

    # Detect foot contacts based on velocity and height thresholds
    def foot_detect(positions, thres):
        """
        Detects foot contacts in a sequence of joint positions based on velocity and a predefined threshold.
        This function calculates the square of the velocity (change in position over frames squared) for each
        foot joint. It then checks if these values are below a certain velocity factor, indicating a contact
        (foot is stationary or moving very slowly). This detection helps in identifying when the feet are in
        contact with the ground during motion capture analysis.

        Parameters:
        - positions (numpy.ndarray): The joint positions for all frames, where each position includes x, y, z coordinates.
        - thres (float): A threshold value used to determine the maximum allowed velocity for a foot contact.

        Returns:
        - feet_l (numpy.ndarray): A binary array indicating foot contact for the left foot across frames.
        - feet_r (numpy.ndarray): A binary array indicating foot contact for the right foot across frames.
        """

        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)  # Compute foot contacts for the sequence
    # feet_l, feet_r = foot_detect(positions, 0.002)

    """Quaternion and Cartesian representation"""
    r_rot = None

    def get_rifke(positions):
        """
        Normalizes positions to a root-relative coordinate system and adjusts them to face a standard direction (Z+).
        This function subtracts the first frame's root joint position from all joint positions in all frames to
        normalize them around the origin. It then applies a quaternion rotation to ensure that the character or skeleton
        faces the Z+ direction across all frames, which is commonly used for consistency in animations and simulations.

        Parameters:
        - positions (numpy.ndarray): Array of joint positions that need to be adjusted.

        Returns:
        - positions (numpy.ndarray): The modified joint positions after alignment and normalization.
        """

        """Local pose"""
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]

        """All pose face Z+"""
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)

        return positions

    # Extract and manipulate pose and velocity data for further processing (e.g., for learning tasks)
    def get_quaternion(positions):
        """
        Computes quaternion rotations for joint positions using inverse kinematics, applies fixes for quaternion
        discontinuities, and calculates both linear and angular velocities of the root joint. This function aims to
        generate a compact representation of motion data that can be used for animation or further motion analysis.

        Parameters:
        - positions (numpy.ndarray): The 3D joint positions for which to compute the kinematics.

        Returns:
        - quat_params (numpy.ndarray): Quaternion parameters for all joints across all frames.
        - r_velocity (numpy.ndarray): Root angular velocity across frames.
        - velocity (numpy.ndarray): Root linear velocity across frames.
        - r_rot (numpy.ndarray): Root rotation quaternion for all frames.
        """
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")  # Initialize skeleton for kinematics

        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=False
        )  # Compute quaternions

        quat_params = qfix(quat_params)  # Fix discontinuities in quaternion data

        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()  # Root quaternion for the sequence

        """Root Linear Velocity"""
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()  # Compute linear velocity of root

        # print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)  # Rotate velocities to align with root orientation

        """Root Angular Velocity"""
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))  # Compute angular velocity for the root
        quat_params[1:, 0] = r_velocity  # Update root quaternions with angular velocities

        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot  # Return pose and velocity data

    def get_cont6d_params(positions):
        """
        Converts quaternion representations of joint rotations to a continuous 6D representation to avoid issues like
        gimbal lock and provide smoother transitions in animations. This function first computes quaternions using inverse
        kinematics with optional smoothing and then converts these quaternions to a 6D format. It also calculates both
        linear and angular velocities to provide a comprehensive motion description.

        Parameters:
        - positions (numpy.ndarray): Joint positions for which to compute the kinematics.

        Returns:
        - cont_6d_params (numpy.ndarray): Continuous 6D parameters for all joints across all frames.
        - r_velocity (numpy.ndarray): Angular velocity of the root joint across frames.
        - velocity (numpy.ndarray): Linear velocity of the root joint across frames.
        - r_rot (numpy.ndarray): Root rotation quaternion used for the velocity calculations.
        """

        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")  # Initialize skeleton for kinematics
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=True
        )  # Compute quaternions with smoothing

        """Quaternion to continuous 6D"""
        cont_6d_params = quaternion_to_cont6d_np(quat_params)  # Convert quaternions to 6D continuous representation
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()  # Root quaternion for the sequence
        #     print(r_rot[0])

        """Root Linear Velocity"""
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()  # Compute linear velocity of root
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)  # Rotate velocities to align with root orientation

        """Root Angular Velocity"""
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))  # Compute angular velocity for the root
        # (seq_len, joints_num, 4)

        return cont_6d_params, r_velocity, velocity, r_rot  # Return pose and velocity data

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    """Root height"""
    root_y = positions[:, 0, 1:2]  # Extract Y-component of the root joint across all frames

    """Root rotation and linear velocity"""
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(
        r_velocity[:, 2:3]
    )  # Compute sine inverse of angular velocity to get rotation angle around Y-axis
    l_velocity = velocity[:, [0, 2]]  # Linear velocity components in the XZ plane
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate(
        [r_velocity, l_velocity, root_y[:-1]], axis=-1
    )  # Combine rotation, linear velocity, and Y component for root data

    """Get Joint Rotation Representation"""
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(
        len(cont_6d_params), -1
    )  # Flatten rotation data for joints excluding the root

    """Get Joint Rotation Invariant Position Represention"""
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(
        len(positions), -1
    )  # Flatten local joint positions for joints excluding the root

    """Get Joint Velocity Representation"""
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1), global_positions[1:] - global_positions[:-1]
    )
    local_vel = local_vel.reshape(len(local_vel), -1)  # Flatten local velocities for all joints

    data = root_data  # Start constructing the final data array with root data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)  # Add joint positions
    data = np.concatenate([data, rot_data[:-1]], axis=-1)  # Add joint rotations
    data = np.concatenate([data, local_vel], axis=-1)  # Add joint velocities
    data = np.concatenate([data, feet_l, feet_r], axis=-1)  # Add foot contact data

    return data, global_positions, positions, l_velocity


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)


# Recover global angle and positions for rotation data
def recover_root_rot_pos(data):
    """
    Recovers the rotation angles and positions from a series of rotational velocities stored in a data structure.
    This function computes cumulative angles from angular velocities, converts them to quaternion format, and applies
    these rotations to calculate global joint positions. It uses cumulative summation and quaternion transformations
    to transition from local frame changes to a coherent global trajectory for the skeleton.

    Parameters:
    - data (tensor): The input data containing rotational velocities and other joint data.

    Returns:
    - r_rot_quat (tensor): Quaternion representation of the cumulative root rotations.
    - r_pos (tensor): Cumulative global positions of the root joint.
    """

    rot_vel = data[..., 0]  # Extract angular velocities for recovery
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)  # Prepare tensor for cumulative angle calculation
    """Get Y-axis rotation from rotation velocity"""
    r_rot_ang[..., 1:] = rot_vel[..., :-1]  # Initialize angles except for the first frame
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)  # Cumulatively sum up the angles to get rotation angles over time

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)  # Prepare tensor for quaternion representation
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)  # Compute cosine component for quaternion
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)  # Compute sine component for quaternion

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)  # Prepare tensor for root positions
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]  # Set positions for X and Z components
    """Add Y-axis rotation to root position"""
    r_pos = qrot(qinv(r_rot_quat), r_pos)  # Rotate positions based on quaternion to align

    r_pos = torch.cumsum(r_pos, dim=-2)  # Cumulatively sum up positions to get global trajectory

    r_pos[..., 1] = data[..., 3]  # Set Y component for the positions
    return r_rot_quat, r_pos  # Return root quaternions and positions


def recover_from_rot(data, joints_num, skeleton):
    """
    Converts rotation data from a quaternion format to a continuous 6D representation and computes the global
    joint positions using the skeleton's forward kinematics. This function is used to translate motion data
    from a compact format (like quaternion) to a more generalized format (6D) and apply it to get the actual
    3D coordinates of the skeleton joints.

    Parameters:
    - data (tensor): Input data including rotation and position information.
    - joints_num (int): The number of joints in the skeleton.
    - skeleton (Skeleton): The skeleton object to perform kinematic computations.

    Returns:
    - positions (tensor): The global positions of all joints computed from the 6D parameters.
    """

    r_rot_quat, r_pos = recover_root_rot_pos(data)  # Recover root quaternion and position from data

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)  # Convert root quaternion to 6D continuous representation

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]  # Extract 6D parameters for joints
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)  # Combine root and joint 6D parameters
    cont6d_params = cont6d_params.view(-1, joints_num, 6)  # Reshape for kinematics calculation

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)  # Compute positions from 6D parameters

    return positions


def recover_from_ric(data, joints_num):
    """
    Recovers joint positions from rotation-invariant coordinates by applying the reverse transformations
    of the rotations stored in quaternion format. This function translates position data stored relative
    to the root joint into absolute 3D space, using the given rotations to ensure that the local joint
    orientations are maintained relative to the global orientation of the figure.

    Parameters:
    - data (tensor): The input data containing the local joint positions and associated rotation data.
    - joints_num (int): The number of joints considered in the skeleton model.

    Returns:
    - positions (tensor): The absolute joint positions in 3D space.
    """

    r_rot_quat, r_pos = recover_root_rot_pos(data)  # Recover root quaternion and position from data
    positions = data[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))  # Reshape local positions for joints

    """Add Y-axis rotation to local joints"""
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions
    )  # Rotate positions to align with root

    """Add root XZ to joints"""
    positions[..., 0] += r_pos[..., 0:1]  # Add root X component to positions
    positions[..., 2] += r_pos[..., 2:3]  # Add root Z component to positions

    """Concate root and joints"""
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)  # Concatenate root and joint positions

    return positions  # Return positions


# The given data is used to double check if you are on the right track.
reference1 = np.load("./HumanML3D/new_joints_comp/TotalCapture/s3/walking1_poses.npy")
reference2 = np.load("./HumanML3D/new_joint_comp_vecs/TotalCapture/s3/walking1_poses.npy")


"""
Data Generation
"""
# Lower legs
l_idx1, l_idx2 = 5, 8  # Lower leg indices for scaling calculation
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]  # Indices for right and left feet
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]  # Indices for calculating facing direction
# l_hip, r_hip
r_hip, l_hip = 2, 1  # Hip indices for facing direction calculation
joints_num = 22  # Total number of joints in the skeleton
# ds_num = 8


dataset = "MIA"

if dataset == "AMASS":
    example_id = "EyesJapanDataset/frederic/walk-04-fast-frederic_poses"

    data_dir = "./joints_comp/"  # Directory containing joint data
    proc_joints_dir = "./HumanML3D/new_joints_comp/"  # Directory for saving processed data
    full_motion_dir = "./HumanML3D/new_joint_comp_vecs/"  # Directory for saving component vectors
elif dataset == "MIA":
    example_id = "train/Subject4/SlowSkater/1137"

    data_dir = "MIAHML3D/joints/"  # Directory containing joint data
    proc_joints_dir = "MIAHML3D/proc_joints/"  # Directory for saving processed data
    full_motion_dir = "MIAHML3D/motion_vects/"  # Directory for saving component vectors

os.makedirs(proc_joints_dir, exist_ok=True)
os.makedirs(full_motion_dir, exist_ok=True)

n_raw_offsets = torch.from_numpy(t2m_raw_offsets)  # Load raw offsets
kinematic_chain = t2m_kinematic_chain  # Load kinematic chain configuration

# Load example data and prepare target skeleton offsets
example_data = np.load(os.path.join(data_dir, example_id + ".npy"))
example_data = example_data.reshape(len(example_data), -1, 3)
example_data = torch.from_numpy(example_data)
tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")  # Initialize target skeleton
# (joints_num, 3)
tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])  # Get target offsets for the example
print(tgt_offsets)  # Print target offsets for debugging

example_data = np.load(os.path.join(data_dir, example_id + ".npy"))[:, :joints_num]
process_file(example_data, 0.002)


import glob

source_list = glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)  # Load all source data files
glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)
print(source_list[0])  # Print first source file for verification
print(len(source_list))  # Print number of source files for tracking


frame_num = 0  # Initialize frame count
for source_file in tqdm(source_list):  # Process each source file
    source_data = np.load(os.path.join(source_file))[:, :joints_num]
    target_file = os.path.normpath(source_file).split(os.sep)[1:]

    target_file[0] = proc_joints_dir
    dirs_1 = pjoin(*target_file[:-1])

    target_file2 = target_file[:]

    target_file2[0] = full_motion_dir
    dirs_2 = pjoin(*target_file2[:-1])

    os.makedirs(dirs_1, exist_ok=True)
    os.makedirs(dirs_2, exist_ok=True)

    try:
        data, ground_positions, positions, l_velocity = process_file(source_data, 0.002)
        rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
        np.save(pjoin(*target_file), rec_ric_data.squeeze().numpy())
        np.save(pjoin(*target_file2), data)
        frame_num += data.shape[0]
    except Exception as e:
        print(source_file)
        print(e)
#         print(source_file)
#         break

print("Total clips: %d, Frames: %d, Duration: %fm" % (len(source_list), frame_num, frame_num / 20 / 60))
