import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
#import motion_representation
from HumanML3D.humanml_to_smpl_functions import recover_full_motion_from_data, process_file
from HumanML3D.paramUtil import t2m_raw_offsets, t2m_kinematic_chain, face_joint_indx
from HumanML3D.humans4d_preprocessing import joints_to_smpl_params, to_smpl_frame

TRANS_MATRIX = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0]])
INV_TRANS_MATRIX = torch.as_tensor(np.linalg.inv(TRANS_MATRIX), dtype=torch.float32)

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
            from HumanML3D.humans4d_preprocessing import get_4dhumans_paths, humans4d_to_pose, initialize_body_models, joints_to_smpl_params
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
            pose, _, bm = self.humans4d_to_pose(sample_id, self.male_bm, self.female_bm, self.device)
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
        mean = np.mean(feature, axis=0)
        std = np.std(feature, axis=0)
        print("std min/max:", np.min(std), np.max(std))
        print("Any zero?", np.any(std == 0))
        print("Any NaN?", np.any(np.isnan(std)))

        eps = 1e-6
        std_safe = np.where(std < eps, 1.0, std)
        feature_norm = (feature - mean) / std_safe
        print("Feature stats:", np.mean(feature_norm), np.std(feature_norm))
        print("Any NaN?", np.any(np.isnan(feature_norm)))
        device='cuda'
        mdm_ckpt_path = "/home/bkizilcelik/workspace/smpl-tools/mdm/save/humanml_enc_512_50steps/model000750000.pt"
        from mdm.model.mdm import MDM
        from mdm.diffusion.gaussian_diffusion import GaussianDiffusion
        from mdm.utils.model_util import load_saved_model, create_model_and_diffusion
        from mdm.data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
        import json
        from types import SimpleNamespace   
        print("Applying MDM denoising on track poses...")
        with open("/home/bkizilcelik/workspace/smpl-tools/mdm/save/humanml_enc_512_50steps/args.json", "r") as f:
            args_dict = json.load(f)
        # Step 2 (optional): Convert to Namespace if needed
        args = SimpleNamespace(**args_dict)
        class DummyData:
            pass
        dummy_data = DummyData()
        # Example: set an attribute 'dataset' with required information.
        # In your context, data.dataset must have 'num_actions' if available.
        dummy_data.dataset = type("DummyDataset", (), {"num_actions": 1})
        # Now call the function
        mdm_model, diffusion = create_model_and_diffusion(args, dummy_data)
        mdm_model = load_saved_model(model=mdm_model, model_path=mdm_ckpt_path)
        mdm_model.to(device)
        mdm_model.eval()
        poses_tensor = torch.tensor(feature_norm)
        poses_tensor = poses_tensor.transpose(0, 1).unsqueeze(0).unsqueeze(2)
        # Apply diffusion denoising.
        with torch.no_grad():
            x = poses_tensor.to(device)  # Shape: [B, ...]
            B, T = x.shape[0], x.shape[-1]  # e.g. batch size B, frames T
            model_kwargs = {
                "y": {
                    "text": [""] * B , # a list of empty strings, one per batch element
                    "mask": torch.ones((B, 1, 1, T), dtype=torch.bool, device=device),
                }
            }
            for t in reversed(range(diffusion.num_timesteps)):
                # Create a timestep tensor with shape (B,)
                t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
                out = diffusion.p_sample(mdm_model, x, t_tensor, model_kwargs=model_kwargs)
                x = out["sample"]
            denoised_poses = out["pred_xstart"].squeeze().transpose(0, 1).to("cpu")
            denoised = denoised_poses * std_safe + mean
        positions = recover_full_motion_from_data(denoised, self.target_offset, t2m_raw_offsets, t2m_kinematic_chain, face_joint_indx, floor_height,root_pose_init_xz,root_quat_init)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        def vis_joints(joints: np.ndarray, frame: int = 0):
            # joints: (T, 22, 3)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            j = joints[frame]
            ax.scatter(j[:, 0], j[:, 1], j[:, 2], s=25)
            ax.set_title(f'Frame {frame}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.savefig(f"/home/bkizilcelik/workspace/smpl-tools/frame_{frame:04d}.png")
            plt.close(fig)
        for f in range(0, positions.shape[0], 10):  # save every 10th frame
            vis_joints(positions, frame=f)

        # vis_joints(positions, frame=0)
        # original_positions = reverse_swap_left_right(positions)
        # 1) back to SMPL coordinate frame
        joints_smpl = to_smpl_frame(positions)            # torch (T,22,3)

        # 2) inverse kinematics
        glo, bpose, tra = joints_to_smpl_params(joints_smpl, bm)
        # final_pose = preprocess_reverse_pose(positions, sample_id)
        # assert np.allclose(positions_np, final_pose, atol=1e-4)
        return sample_id, feature, pose, glo, bpose, tra
