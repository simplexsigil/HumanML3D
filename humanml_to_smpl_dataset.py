import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
#import motion_representation
from humanml_to_smpl_functions import recover_full_motion_from_data, process_file
from paramUtil import t2m_raw_offsets, t2m_kinematic_chain, face_joint_indx

def swap_left_right(data):
    # Swap left/right joints without modifying x-axis (handled outside)
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1  # Negate x-axis
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain].copy()
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain].copy()
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

def preprocess_pose(pose, sample_id):
    # For non-'humanact12' samples, negate the x-axis purposefully
    if isinstance(sample_id, str) and "humanact12" not in sample_id:
        pose[..., 0] *= -1
    # Swap left/right joints
    pose = swap_left_right(pose)
    return pose

def preprocess_reverse_pose(pose, sample_id):
    # For non-'humanact12' samples, negate the x-axis purposefully
    if isinstance(sample_id, str) and "humanact12" not in sample_id:
        pose[..., 0] *= -1
    # Swap left/right joints
    pose = reverse_swap_left_right(pose)
    return pose

def reverse_swap_left_right(data):
    # Revert the left/right swap and re-negate x-axis
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    
    # Reverse x-axis flip
    data[..., 0] *= -1

    # Define joint indices again
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]

    # Reverse the joint swaps
    tmp = data[:, left_chain].copy()
    data[:, left_chain] = data[:, right_chain]
    data[:, right_chain] = tmp

    if data.shape[1] > 24:
        tmp = data[:, left_hand_chain].copy()
        data[:, left_hand_chain] = data[:, right_hand_chain]
        data[:, right_hand_chain] = tmp

    return data


class ConversionDataset(Dataset):
    def __init__(self, data_type, input_dir, dataset=None, device="cpu", body_models_dir="./body_models", tgt_skel=None, target_offset=None):
        self.data_type = data_type.lower()
        self.input_dir = input_dir
        self.device = device
        self.dataset_filter = dataset
        self.tgt_skel = tgt_skel # Skeleton object for target offsets
        self.target_offset = target_offset
        self.samples = []
        
        if self.data_type == "mia":
            from mia_preprocessing import get_leaf_directories, mia_to_pose
            self.mia_to_pose = mia_to_pose
            # List sample directories (each with required npy files)
            self.samples = get_leaf_directories(input_dir)
            # Initialize SMPLH body model
            from smplx.body_models import SMPLH
            self.smpl_h = SMPLH(
                model_path=f"{body_models_dir}/smpl/SMPLH_NEUTRAL_AMASS_MERGED.pkl", 
                num_betas=10, 
                use_pca=False, 
                batch_size=59
            ).to(self.device)
        elif self.data_type == "amass":
            from amass_preprocessing import get_amass_paths, amass_to_pose, initialize_body_models
            self.amass_to_pose = amass_to_pose
            group_paths, dataset_names = get_amass_paths(input_dir)
            # Filter samples if a specific dataset is requested
            for paths, name in zip(group_paths, dataset_names):
                if (self.dataset_filter is None) or (name == self.dataset_filter):
                    self.samples.extend(paths)
            self.male_bm, self.female_bm = initialize_body_models(body_models_dir=body_models_dir, device=self.device)
        elif self.data_type == "4dhumans":
            from humans4d_preprocessing import get_4dhumans_paths, humans4d_to_pose, initialize_body_models
            self.humans4d_to_pose = humans4d_to_pose 
            group_paths, dataset_names = get_4dhumans_paths(input_dir)
            for paths, name in zip(group_paths, dataset_names):
                if (self.dataset_filter is None) or (name == self.dataset_filter):
                    self.samples.extend(paths)
            self.male_bm, self.female_bm = initialize_body_models(body_models_dir=body_models_dir, device=self.device)          
        else:
            raise ValueError("Unsupported dataset type. Use 'mia' or 'amass'.")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        joints_num = 22  # Standard number of joints
        if self.data_type == "mia":
            # Process MIA sample using mia_to_pose and SMPLH model
            pose = self.mia_to_pose(sample_id, self.smpl_h, self.device)
        elif self.data_type == "4dhumans":
            pose, _ = self.humans4d_to_pose(sample_id, self.male_bm, self.female_bm, self.device)
            temp_pose = pose
            if pose is None:
                print(f"Pose data is None for sample {sample_id}. Skipping...")
                return sample_id, [], []
            pose = preprocess_pose(pose, sample_id)
        else:  # amass
            pose, _ = self.amass_to_pose(sample_id, self.male_bm, self.female_bm, self.device)
            
            if pose is None:
                print(f"Pose data is None for sample {sample_id}. Skipping...")
                return sample_id, [], []
                
            pose = preprocess_pose(pose, sample_id)
        
        # Preprocess pose: negate x-axis if needed and swap left/right joints

        
        # Select only the standard joints and return raw pose positions
        positions_np = pose[:, :joints_num]

        if len(positions_np) > 1:
            feature, positions, global_positions, positions, l_velocity, floor_height, root_pose_init_xz, root_quat_init  = process_file(positions_np, self.target_offset, 0.002, self.device)
        else:
            feature = []
        positions = recover_full_motion_from_data( feature, self.target_offset, t2m_raw_offsets, t2m_kinematic_chain, face_joint_indx, floor_height,root_pose_init_xz,root_quat_init)
        final_pose = preprocess_reverse_pose(positions, sample_id)
        # assert np.allclose(positions_np, final_pose, atol=1e-4)
        return sample_id, feature, pose
