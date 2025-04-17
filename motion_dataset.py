import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import motion_representation


class MotionDataset(Dataset):
    def __init__(
        self,
        data_type,
        input_dir,
        dataset=None,
        device="cpu",
        body_models_dir="./body_models",
        tgt_skel=None,
        target_offset=None,
    ):
        self.data_type = data_type.lower()
        self.input_dir = input_dir
        self.device = device
        self.dataset_filter = dataset
        self.tgt_skel = tgt_skel  # Skeleton object for target offsets
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
                batch_size=59,
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
        else:  # amass
            pose, _ = self.amass_to_pose(sample_id, self.male_bm, self.female_bm, self.device)
            if pose is None:
                print(f"Pose data is None for sample {sample_id}. Skipping...")
                return sample_id, [], []

        # Preprocess pose: negate x-axis if needed and swap left/right joints

        # Select only the standard joints and return raw pose positions
        positions_np = pose[:, :joints_num]

        if len(positions_np) > 1:
            feature, _, _, _ = motion_representation.process_file(positions_np, self.target_offset, 0.002, self.device)
        else:
            feature = []

        return sample_id, feature, pose
