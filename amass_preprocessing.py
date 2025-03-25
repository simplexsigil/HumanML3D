import os
import torch
import numpy as np
from tqdm import tqdm
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

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


def amass_to_pose(src_path, male_bm, female_bm):
    """Convert AMASS motion data to joint positions"""
    device = male_bm.trans.device  # Get device from body model
    
    bdata = np.load(src_path, allow_pickle=True)
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
    
    with torch.no_grad():
        for fId in range(0, frame_number, down_sample):
            root_orient = torch.Tensor(bdata["poses"][fId : fId + 1, :3]).to(device)
            pose_body = torch.Tensor(bdata["poses"][fId : fId + 1, 3:66]).to(device)
            pose_hand = torch.Tensor(bdata["poses"][fId : fId + 1, 66:]).to(device)
            betas = torch.Tensor(bdata["betas"][:10][np.newaxis]).to(device)
            trans = torch.Tensor(bdata["trans"][fId : fId + 1]).to(device)
            
            body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, root_orient=root_orient)
            joint_loc = body.Jtr[0] + trans
            pose_seq.append(joint_loc.unsqueeze(0))
            
    if not pose_seq:  # If no frames were processed
        return None, fps
        
    pose_seq = torch.cat(pose_seq, dim=0)
    pose_seq_np = pose_seq.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, TRANS_MATRIX)
    
    return pose_seq_np_n, fps


def get_amass_paths(dataset_dir):
    """Get all AMASS dataset paths"""
    paths = []
    folders = []
    dataset_names = []
    
    for root, dirs, files in os.walk(dataset_dir):
        folders.append(root)
        if "tars" in dirs:
            dirs.remove("tars")
        for name in files:
            if name in ["LICENSE.txt"]:
                continue
            # Fix: Make sure dataset_name extraction is consistent with original code
            try:
                dataset_name = root.split("/")[2]
                if dataset_name not in dataset_names:
                    dataset_names.append(dataset_name)
            except IndexError:
                # Handle case where path structure doesn't have enough components
                continue
            paths.append(os.path.join(root, name))
    
    group_path = [[path for path in paths if name in path] for name in dataset_names]
    return group_path, dataset_names


def process_amass_dataset(dataset_dir, dataset_name=None, output_dir=None, save_intermediate=False, device=None):
    """Process AMASS dataset and return joint positions
    
    Args:
        dataset_dir: Directory containing AMASS data
        dataset_name: Name of specific dataset to process (if not specified, process all)
        output_dir: Directory to save processed data (if save_intermediate is True)
        save_intermediate: Whether to save intermediate results to disk
        device: Device to run computations on ('cuda', 'cuda:0', 'cpu', etc.)
    
    Returns:
        Dictionary mapping sample paths to joint position arrays
    """
    # Initialize body models
    male_bm, female_bm = initialize_body_models(device=device)
    
    # Get paths for datasets
    group_path, dataset_names = get_amass_paths(dataset_dir)
    
    results = {}
    all_count = sum([len(paths) for paths in group_path])
    cur_count = 0
    
    for paths in group_path:
        if not paths:
            continue
            
        try:
            current_dataset = paths[0].split("/")[2]
        except IndexError:
            continue
        
        # Skip if not the requested dataset
        if dataset_name and current_dataset != dataset_name:
            continue
            
        pbar = tqdm(paths)
        pbar.set_description(f"Processing: {current_dataset}")
        fps = 0  # Keep track of fps for reporting
        
        for path in pbar:
            if save_intermediate and output_dir:
                save_path = path.replace(dataset_dir, output_dir)
                save_path = save_path[:-3] + "npy"
                # Create directories if they don't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Process the pose data
            pose_data, current_fps = amass_to_pose(path, male_bm, female_bm)
            fps = current_fps  # Update fps
            
            if pose_data is not None:
                # Store result in memory
                results[path] = pose_data
                
                if save_intermediate and output_dir:
                    np.save(save_path, pose_data)
            
            cur_count += 1
            
        print(f"Processed / All (fps {fps}): {cur_count}/{all_count}")
    
    return results


if __name__ == "__main__":
    # This is only for standalone testing
    import argparse
    parser = argparse.ArgumentParser(description="Process AMASS motion data")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing AMASS data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--dataset", type=str, help="Specific dataset to process")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run computations on ('cuda', 'cuda:0', 'cpu', etc.)")
    args = parser.parse_args()
    
    process_amass_dataset(args.input_dir, args.dataset, args.output_dir, save_intermediate=True, device=args.device)
