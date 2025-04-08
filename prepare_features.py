#!/usr/bin/env python3
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import glob

import mia_preprocessing
import amass_preprocessing
import motion_representation
import cal_mean_variance
from motion_dataset import MotionDataset
from paramUtil import t2m_kinematic_chain, t2m_raw_offsets, joints_num
from common.skeleton import Skeleton
from amass_preprocessing import get_amass_paths, amass_to_pose, initialize_body_models
from smplx.body_models import SMPLH


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process motion data and prepare features", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data_type", type=str, required=True, choices=["mia", "amass"], help="Type of data to process (mia or amass)"
    )

    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw motion data")

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save final processed features")

    parser.add_argument(
        "--dataset", type=str, help="For AMASS: specific dataset to process (if not specified, process all)"
    )

    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate results to disk")

    parser.add_argument(
        "--intermediate_dir", type=str, help="Directory to save intermediate results (if save_intermediate is True)"
    )

    parser.add_argument("--calc_stats", action="store_true", help="Calculate mean and standard deviation of features")

    parser.add_argument(
        "--stats_dir",
        type=str,
        help="Directory to save mean and std statistics (defaults to output_dir if not specified)",
    )

    parser.add_argument(
        "--body_models_dir",
        type=str,
        default="./body_models",
        help="Directory containing SMPL-H body models (required for AMASS processing)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run computations on ('cuda', 'cuda:0', 'cpu', etc.)",
    )

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for data loading")

    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader worker processes")

    parser.add_argument(
        "--skeleton_reference_path", type=str, default=None, help="Path to example data file for testing"
    )
    # For AMASS: "EyesJapanDataset/frederic/walk-04-fast-frederic_poses"
    # For MIA: "train/Subject4/SlowSkater/1137"

    return parser.parse_args()


def swap_left_right(data):
    # Swap left/right joints without modifying x-axis (handled outside)
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
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


def process_and_generate_features(args):
    """
    Main function to prepare motion features from input data
    """
    print(f"Processing {args.data_type} data from {args.input_dir} using {args.device}")
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)  # Load raw offsets
    kinematic_chain = t2m_kinematic_chain  # Load kinematic chain configuration
    male_bm, female_bm = initialize_body_models(body_models_dir=args.body_models_dir, device=args.device)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")

    # Here we load a sample to get a default skeleton which is used as reference
    # for the rest of the dataset. All other sample skeletons are scaled to this to normalize.
    try:
        sample_id = args.skeleton_reference_path
        amass_to_pose = amass_preprocessing.amass_to_pose
        pose, _ = amass_to_pose(sample_id, male_bm, female_bm, args.device)
        pose = preprocess_pose(pose, sample_id)

        pose = pose.reshape(len(pose), -1, 3)
        pose = torch.from_numpy(pose)
        target_offset = tgt_skel.get_offsets_joints(pose[0])
    except Exception as e:
        print(f"Error loading reference sample: {e}")
        raise

    smpl_h = SMPLH(
        model_path=f"{args.body_models_dir}/smpl/SMPLH_NEUTRAL_AMASS_MERGED.pkl",
        num_betas=10,
        use_pca=False,
        batch_size=59,
    )
    smpl_h.to(args.device)

    # Create dataset and DataLoader for parallel loading
    dataset = MotionDataset(
        data_type=args.data_type,
        input_dir=args.input_dir,
        dataset=args.dataset,
        device=args.device,
        body_models_dir=args.body_models_dir,
        tgt_skel=tgt_skel,
        target_offset=target_offset,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    features_dict = {}
    poses_dict = {}
    # Iterate over batches; each sample is a tuple: (sample_id, raw pose)
    for sample_ids, features, poses in tqdm(loader, desc="Processing motions"):
        for sample_id, feature, pose in zip(sample_ids, features, poses):
            # Process the raw pose using process_file outside the dataset.
            features_dict[sample_id] = feature
            poses_dict[sample_id] = pose
    print(f"Generated features for {len(features_dict)} samples")

    # Step 3: Calculate mean and variance if requested
    mean = None
    std = None
    if args.calc_stats:
        print("Step 3: Calculating mean and variance...")
        mean, std = cal_mean_variance.calculate_statistics(features_dict)

    return features_dict, poses_dict, mean, std


def save_results(features, mean, std, args):
    """
    Save the processed features and statistics

    Args:
        features: Dictionary of motion features
        mean: Mean of features (or None)
        std: Standard deviation of features (or None)
        args: Command line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save features
    print(f"Saving features to {args.output_dir}...")
    for key, feature in tqdm(features.items(), desc="Saving features"):
        # Create a sensible filename from the key
        if args.data_type == "amass":
            if isinstance(key, str) and os.path.isfile(key):
                # If the key is a file path, use the same directory structure
                rel_path = (
                    os.path.relpath(key, args.input_dir) if key.startswith(args.input_dir) else os.path.basename(key)
                )
                save_path = os.path.join(args.output_dir, rel_path).replace(".npz", ".npy")
            else:
                # Otherwise create a unique filename
                save_path = os.path.join(args.output_dir, f"feature_{hash(str(key))}.npy")
        else:
            # For MIA, use the sample ID directly
            rel_path = os.path.relpath(key, args.input_dir) if key.startswith(args.input_dir) else os.path.basename(key)
            save_path = os.path.join(args.output_dir, rel_path, "feature.npy")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the feature
        np.save(save_path, feature)

    # Save mean and std if calculated
    if mean is not None and std is not None:
        stats_dir = args.stats_dir if args.stats_dir else args.output_dir
        os.makedirs(stats_dir, exist_ok=True)
        np.save(os.path.join(stats_dir, "Mean.npy"), mean)
        np.save(os.path.join(stats_dir, "Std.npy"), std)
        print(f"Statistics saved to {stats_dir}")


def main():
    """Main entry point"""
    args = parse_arguments()

    if args.skeleton_reference_path is None:
        args.skeleton_reference_path = (
            "EyesJapanDataset/frederic/walk-04-fast-frederic_poses"
            if args.data_type == "amass"
            else "train/Subject4/SlowSkater/1137" if args.data_type == "mia" else None
        )

        if args.skeleton_reference_path is None:
            raise ValueError("skeleton_reference_path must be provided for testing")

    # Process data and generate features
    features, poses, mean, std = process_and_generate_features(args)

    if args.save_intermediate:
        # Save intermediate results if requested
        os.makedirs(args.intermediate_dir, exist_ok=True)
        for key, pose in tqdm(poses.items(), desc="Saving intermediate poses"):
            if args.data_type == "amass":
                if isinstance(key, str) and os.path.isfile(key):
                    # If the key is a file path, use the same directory structure
                    rel_path = (
                        os.path.relpath(key, args.input_dir)
                        if key.startswith(args.input_dir)
                        else os.path.basename(key)
                    )
                    save_path = os.path.join(args.intermediate_dir, rel_path).replace(".npz", ".npy")
                else:
                    # Otherwise create a unique filename
                    save_path = os.path.join(args.intermediate_dir, f"pose_{hash(str(key))}.npy")
            else:
                # For MIA, use the sample ID directly
                rel_path = (
                    os.path.relpath(key, args.input_dir) if key.startswith(args.input_dir) else os.path.basename(key)
                )
                save_path = os.path.join(args.intermediate_dir, rel_path, "pose.npy")

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, pose)

    print(f"Intermediate poses saved to: {args.intermediate_dir}")
    # Save results
    save_results(features, mean, std, args)

    print("Feature preparation complete!")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("fork", force=True)
    main()
