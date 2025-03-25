#!/usr/bin/env python3
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch

import mia_preprocessing
import amass_preprocessing
import motion_representation
import cal_mean_variance


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process motion data and prepare features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data_type", 
        type=str, 
        required=True, 
        choices=["mia", "amass"],
        help="Type of data to process (mia or amass)"
    )
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Directory containing raw motion data"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save final processed features"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        help="For AMASS: specific dataset to process (if not specified, process all)"
    )
    
    parser.add_argument(
        "--save_intermediate", 
        action="store_true", 
        help="Save intermediate results to disk"
    )
    
    parser.add_argument(
        "--intermediate_dir", 
        type=str, 
        help="Directory to save intermediate results (if save_intermediate is True)"
    )
    
    parser.add_argument(
        "--calc_stats", 
        action="store_true", 
        help="Calculate mean and standard deviation of features"
    )
    
    parser.add_argument(
        "--stats_dir", 
        type=str, 
        help="Directory to save mean and std statistics (defaults to output_dir if not specified)"
    )
    
    parser.add_argument(
        "--body_models_dir",
        type=str,
        default="./body_models",
        help="Directory containing SMPL-H body models (required for AMASS processing)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run computations on ('cuda', 'cuda:0', 'cpu', etc.)"
    )
    
    return parser.parse_args()


def process_and_generate_features(args):
    """
    Main function to prepare motion features from input data
    
    Args:
        args: Command line arguments
    
    Returns:
        Dictionary of generated features
        Mean and standard deviation (if calc_stats is True)
    """
    print(f"Processing {args.data_type} data from {args.input_dir} using {args.device}")
    
    # Step 1: Process raw pose data based on data type
    if args.data_type.lower() == 'mia':
        print("Step 1: Processing MIA data...")
        joint_data = mia_preprocessing.process_mia_data(
            args.input_dir, 
            args.intermediate_dir if args.save_intermediate else None,
            args.save_intermediate,
            args.device
        )
    elif args.data_type.lower() == 'amass':
        print("Step 1: Processing AMASS data...")
        joint_data = amass_preprocessing.process_amass_dataset(
            args.input_dir,
            args.dataset,
            args.intermediate_dir if args.save_intermediate else None,
            args.save_intermediate,
            args.device,
            body_models_dir=args.body_models_dir  # Pass the body_models_dir parameter
        )
    else:
        raise ValueError(f"Unsupported data type: {args.data_type}. Use 'mia' or 'amass'.")
    
    print(f"Processed {len(joint_data)} motion samples.")
    
    # Step 2: Generate motion representation features
    print("Step 2: Generating motion features...")
    features = {}
    n_raw_offsets = torch.from_numpy(motion_representation.t2m_raw_offsets)
    kinematic_chain = motion_representation.t2m_kinematic_chain
    
    # Dynamically get an example to determine the target skeleton
    example_key = next(iter(joint_data))
    example_data = joint_data[example_key]
    example_data = torch.from_numpy(example_data)
    tgt_skel = motion_representation.Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    
    # Process each motion sample
    joints_num = 22  # Standard number of joints
    for key, positions in tqdm(joint_data.items(), desc="Processing motions"):
        try:
            # Process the file using motion_representation functions
            positions_np = positions[:, :joints_num]
            data, _, _, _ = motion_representation.process_file(positions_np, 0.002, args.device)
            features[key] = data
        except Exception as e:
            print(f"Error processing {key}: {e}")
    
    print(f"Generated features for {len(features)} samples")
    
    # Step 3: Calculate mean and variance if requested
    mean = None
    std = None
    if args.calc_stats:
        print("Step 3: Calculating mean and variance...")
        mean, std = cal_mean_variance.calculate_statistics(features)
        
    return features, mean, std


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
        if isinstance(key, str) and os.path.isfile(key):
            # If the key is a file path, use the same directory structure
            rel_path = os.path.relpath(key, args.input_dir) if key.startswith(args.input_dir) else os.path.basename(key)
            save_path = os.path.join(args.output_dir, rel_path)
        else:
            # Otherwise create a unique filename
            save_path = os.path.join(
                args.output_dir,
                f"feature_{hash(str(key))}.npy"
            )
        
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
    
    # Process data and generate features
    features, mean, std = process_and_generate_features(args)
    
    # Save results
    save_results(features, mean, std, args)
    
    print("Feature preparation complete!")


if __name__ == "__main__":
    main()
