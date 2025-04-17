import os
import torch
import numpy as np
from tqdm import tqdm
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
import cProfile

# Set OpenGL platform
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Define constants
TRANS_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
TARGET_FPS = 20


def initialize_body_models(body_models_dir="./body_models", device=None):
    """Initialize SMPL-H body models for male and female"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Define paths
    male_bm_path = os.path.join(body_models_dir, "smplh/male/model.npz")
    male_dmpl_path = os.path.join(body_models_dir, "dmpls/male/model.npz")
    female_bm_path = os.path.join(body_models_dir, "smplh/female/model.npz")
    female_dmpl_path = os.path.join(body_models_dir, "dmpls/female/model.npz")
    
    # Parameters
    num_betas = 10  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters
    
    # Create body models
    male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, 
                        dmpl_fname=male_dmpl_path).to(device)
    
    female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, 
                          dmpl_fname=female_dmpl_path).to(device)
    
    return male_bm, female_bm


def amass_to_pose(src_path, male_bm, female_bm, device):
    """Convert AMASS motion data to joint positions"""
    
    bdata = np.load(src_path, allow_pickle=True, mmap_mode='r')

    fps = 0
    
    try:
        fps = bdata["mocap_framerate"]
        frame_number = bdata["trans"].shape[0]
    except:
        return None, fps
    
    fId = 0  # frame id of the mocap sequence
    pose_seq = []
    
    if bdata["gender"] == "male":
        bm = male_bm
    else:
        bm = female_bm
        
    down_sample = int(fps / TARGET_FPS)

    bdata_poses = bdata["poses"][::down_sample]
    bdata_trans = bdata["trans"][::down_sample]
    root_orients = torch.from_numpy(bdata_poses[:, :3]).float().to(device)
    pose_bodies  = torch.from_numpy(bdata_poses[:, 3:66]).float().to(device)
    pose_hands   = torch.from_numpy(bdata_poses[:, 66:]).float().to(device)
    betas = torch.from_numpy(bdata["betas"][:10][np.newaxis]).float().to(device).expand(len(bdata_poses), -1)
    trans = torch.from_numpy(bdata_trans).float().to(device)

    with torch.no_grad():
        body = bm(pose_body=pose_bodies, pose_hand=pose_hands, betas=betas, root_orient=root_orients)
        # Assuming bm returns batched joints with shape (batch, joints, 3)
        
    pose_seq_np = body.Jtr.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, TRANS_MATRIX)
    
    
    return pose_seq_np_n, fps


def get_amass_paths(dataset_dir):
    """Get all AMASS dataset paths by grouping .npz files based on the first subdirectory"""
    paths_by_dataset = {}
    
    for root, dirs, files in os.walk(dataset_dir):
        if "tars" in dirs:
            dirs.remove("tars")
        for name in files:
            if name in ["LICENSE.txt", "path_mappings.csv", "paths.txt"]:
                continue
            # Process only .npz files
            if not name.endswith(".npz"):
                continue
            full_path = os.path.join(root, name)
            # Use relative path to extract dataset name from the first folder level
            relative = os.path.relpath(full_path, dataset_dir)
            parts = relative.split(os.sep)
            dataset_name = parts[0]
            paths_by_dataset.setdefault(dataset_name, []).append(full_path)
    
    dataset_names = list(paths_by_dataset.keys())
    group_path = list(paths_by_dataset.values())
    return group_path, dataset_names


def process_amass_dataset(dataset_dir, dataset_name=None, output_dir=None, save_intermediate=False, device=None, body_models_dir="./body_models"):
    """Process AMASS dataset and return joint positions
    
    Args:
        dataset_dir: Directory containing AMASS data
        dataset_name: Name of specific dataset to process (if not specified, process all)
        output_dir: Directory to save processed data (if save_intermediate is True)
        save_intermediate: Whether to save intermediate results to disk
        device: Device to run computations on ('cuda', 'cuda:0', 'cpu', etc.)
        body_models_dir: Directory containing SMPL-H body models
    
    Returns:
        Dictionary mapping sample paths to joint position arrays
    """
    # Initialize body models
    male_bm, female_bm = initialize_body_models(body_models_dir=body_models_dir, device=device)
    
    # Get paths for datasets
    group_path, dataset_names = get_amass_paths(dataset_dir)
    print(dataset_names)
    print(group_path[0])
    
    results = {}
    all_count = sum([len(paths) for paths in group_path])
    cur_count = 0
    
    for paths,current_dataset in zip(group_path,dataset_names):
        if not paths:
            continue
        print(current_dataset)
        # Skip if not the requested dataset
        if dataset_name and current_dataset != dataset_name:
            continue
            
        pbar = tqdm(paths)
        pbar.set_description(f"Processing: {current_dataset}")
        fps = 0  # Keep track of fps for reporting

        profiler = cProfile.Profile()
        iteration_count = 0

        for path in pbar:
            # Enable profiler only for the first 10 iterations
            if iteration_count == 0:
                profiler.enable()

            if save_intermediate and output_dir:
                save_path = path.replace(dataset_dir, output_dir)
                save_path = save_path[:-3] + "npy"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            pose_data, current_fps = amass_to_pose(path, male_bm, female_bm, device)
            fps = current_fps  # Update fps

            if pose_data is not None:
                results[path] = pose_data
                if save_intermediate and output_dir:
                    np.save(save_path, pose_data)

            iteration_count += 1
            cur_count += 1

            if iteration_count == 10:
                profiler.disable()
                profiler.dump_stats("profile_output.prof")
                print("Saved profiling data for 10 iterations to profile_output.prof")
                
            # Continue processing without profiling for iterations >10

        print(f"Processed / All (fps {fps}): {cur_count}/{all_count}")
    
    return results


if __name__ == "__main__":
    # This is only for standalone testing
    import argparse
    parser = argparse.ArgumentParser(description="Process AMASS motion data")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing AMASS data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--dataset", type=str, help="Specific dataset to process")
    parser.add_argument("--body_models_dir", type=str, default="./body_models", 
                        help="Directory containing SMPL-H body models")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run computations on ('cuda', 'cuda:0', 'cpu', etc.)")
    args = parser.parse_args()
    
    process_amass_dataset(args.input_dir, args.dataset, args.output_dir, save_intermediate=True, 
                          device=args.device, body_models_dir=args.body_models_dir)
