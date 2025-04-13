#!/usr/bin/env python
# coding: utf-8

"""
humans4d_processing.py

Example script that mirrors the structure of your AMASS processing code,
but adapted for 4DHumans .pkl data.

Usage:
  python humans4d_processing.py --input_dir ./4dhumans_dataset \
                                --output_dir ./output_4dhumans \
                                --body_models_dir ./body_models \
                                --device cuda
"""

import os
import argparse
import joblib
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

# If using human_body_prior for SMPL-H:
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

# Set OpenGL platform for offscreen
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Constants
TRANS_MATRIX = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0]])
TARGET_FPS = 20

###############################################################################
# 1) INITIALIZE BODY MODELS
###############################################################################
def initialize_body_models(body_models_dir="./body_models", device=None):
    """Initialize SMPL-H body models for male and female, same as in amass_to_pose."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Paths for SMPL-H (male/female) + DMPL if needed
    male_bm_path = os.path.join(body_models_dir, "smplh/male/model.npz")
    male_dmpl_path = os.path.join(body_models_dir, "dmpls/male/model.npz")
    female_bm_path = os.path.join(body_models_dir, "smplh/female/model.npz")
    female_dmpl_path = os.path.join(body_models_dir, "dmpls/female/model.npz")
    
    # Hyper-parameters
    num_betas = 10   # number of shape coefficients
    num_dmpls = 8    # number of DMPL coefficients

    # Build the body models
    male_bm = BodyModel(bm_fname=male_bm_path,
                        num_betas=num_betas,
                        num_dmpls=num_dmpls,
                        dmpl_fname=male_dmpl_path).to(device)

    female_bm = BodyModel(bm_fname=female_bm_path,
                          num_betas=num_betas,
                          num_dmpls=num_dmpls,
                          dmpl_fname=female_dmpl_path).to(device)

    return male_bm, female_bm

###############################################################################
# 2) CORE FUNCTION: HUMANS4D_TO_POSE
###############################################################################
def humans4d_to_pose(
    file_path,
    male_bm,                # SMPL-H model (male)
    female_bm,              # SMPL-H model (female)
    device='cuda'
):
    """
    Loads a 4DHumans .pkl, uses your existing track-building logic, then:
    - Picks the track with the most frames
    - Down-samples to TARGET_FPS
    - Runs SMPL-H forward
    - Applies TRANS_MATRIX
    - Returns (pose_seq_np_n, fps) in the same style as amass_to_pose.

    Args:
        file_path (str): path to the .pkl file
        male_bm, female_bm (BodyModel): pre-loaded SMPL-H models
        device (str): 'cuda' or 'cpu'

    Returns:
        pose_seq_np_n (np.ndarray): (num_frames_ds, num_joints, 3)
        fps (float): frames per second from the data or default
    """

    # -------------------------------------------------------------------------
    # 1) Load the .pkl data
    # -------------------------------------------------------------------------
    fdh_results = joblib.load(file_path)

    # We try to extract an fps. If missing, fallback to 30.
    fps = fdh_results.get("fps", 30)

    # -------------------------------------------------------------------------
    # 2) Use the EXACT track-building logic you provided
    # -------------------------------------------------------------------------
    track_data = defaultdict(lambda: {
        "full_body_pose": [],
        "root_translation": [],
        "root_rotation": [],
        "scale_sum": 0,
        "conf_sum": 0,
        "frame_count": 0,
        "first_frame": None,
        "last_frame": None,
        "frame_index": [],
        "3d_joints": []
    })

    # Track the first frame orientation for each track
    track_first_orientations = {}

    # Process each frame
    for frame_key, frame_data in fdh_results.items():
        # If frame_data does not have these keys (e.g., "tid", "smpl"), adjust accordingly
        if not isinstance(frame_data, dict):
            # skip non-dict entries (like "fps")
            continue
        # Zip the corresponding lists
        for idx, (tid, smpl_dict, cam_trans, scale, conf, joints_3d) in enumerate(
            zip(
                frame_data["tid"],
                frame_data["smpl"],
                frame_data["camera"],
                frame_data["scale"],
                frame_data["conf"],
                frame_data["3d_joints"]
            )
        ):
            # Get or initialize first frame orientation for this track
            if tid not in track_first_orientations:
                # e.g. shape (1,3) => smpl_dict["global_orient"][0], then .T
                track_first_orientations[tid] = torch.Tensor(smpl_dict["global_orient"][0]).T

            # Process poses
            global_orient = torch.Tensor(smpl_dict["global_orient"])   # shape e.g. (3,)
            body_pose     = torch.Tensor(smpl_dict["body_pose"])       # shape e.g. (69,) for SMPL
            full_body_pose = torch.cat((global_orient, body_pose), 0)  # shape e.g. (72,)

            translation = torch.Tensor(cam_trans)
            # root_rotation from your snippet (though you may or may not actually use it)
            root_rot = torch.tensor(track_first_orientations[tid])  # or the 'rea()' logic

            joints_3d_t = torch.Tensor(joints_3d)

            # Update track data
            track = track_data[tid]
            track["full_body_pose"].append(full_body_pose)
            track["root_translation"].append(translation)
            track["root_rotation"].append(root_rot)
            track["scale_sum"] += scale
            track["conf_sum"] += conf
            track["frame_count"] += 1

            # Update frame range
            current_time = frame_data["time"]
            if track["first_frame"] is None:
                track["first_frame"] = current_time
            track["first_frame"] = min(track["first_frame"], current_time)
            track["last_frame"] = max(track["last_frame"] or current_time, current_time)

            track["frame_index"].append(frame_data["time"])
            track["3d_joints"].append(joints_3d_t)

    # -------------------------------------------------------------------------
    # 3) Pick one track to emulate amass_to_pose's single sequence output
    # -------------------------------------------------------------------------
    # Let's pick the track with the MOST frames (highest "frame_count")
    if len(track_data) == 0:
        # Means no valid tracks found
        return None, 0

    best_tid = None
    best_count = 0
    for tid, data_dict in track_data.items():
        if data_dict["frame_count"] > best_count:
            best_tid = tid
            best_count = data_dict["frame_count"]

    # If still None, no track had frames
    if best_tid is None:
        return None, 0

    chosen_track = track_data[best_tid]

    # -------------------------------------------------------------------------
    # 4) Convert the chosen track's data into batched tensors
    # -------------------------------------------------------------------------
    full_body_poses = chosen_track["full_body_pose"]       # list of torch.Size([72]) or similar
    root_translations = chosen_track["root_translation"]    # list of torch.Size([3])
    # root_rotations = chosen_track["root_rotation"]        # you have them, not necessarily used

    # Build a single (num_frames, 72) etc.
    full_body_poses_t = torch.stack(full_body_poses, dim=0).to(device)
    root_translations_t = torch.stack(root_translations, dim=0).to(device)
    num_frames = full_body_poses_t.shape[0]

    # -------------------------------------------------------------------------
    # 5) Down-sample frames to TARGET_FPS
    # -------------------------------------------------------------------------
    down_sample = int(fps / TARGET_FPS) if fps >= TARGET_FPS else 1
    down_sample = max(down_sample, 1)

    full_body_poses_t   = full_body_poses_t[::down_sample]
    root_translations_t = root_translations_t[::down_sample]

    # Update final frame count
    final_count = full_body_poses_t.shape[0]

    # -------------------------------------------------------------------------
    # 6) Separate out global_orient vs body_pose if your SMPL model expects them
    #    For standard SMPL: the first 3 are global orientation, the next 69 are body
    # -------------------------------------------------------------------------
    root_orients_t = full_body_poses_t[:, :1]
    body_poses_t   = full_body_poses_t[:, 1:22]
    # If you have SMPL-H hands in "body_pose", slice them out accordingly

    # -------------------------------------------------------------------------
    # 7) We do not have betas in your snippet, so let's just use zero betas
    # -------------------------------------------------------------------------
    betas_t = torch.zeros((final_count, 10), dtype=torch.float32, device=device)

    # -------------------------------------------------------------------------
    # 8) Pick a model (male or female). If you have a track-level gender, adapt here
    #    For demonstration, let's just pick male:
    # -------------------------------------------------------------------------
    bm = male_bm

    # -------------------------------------------------------------------------
    # 9) Forward pass through SMPL-H
    # -------------------------------------------------------------------------
    with torch.no_grad():
        body_out = bm(
            pose_body=body_poses_t,
            root_orient=root_orients_t,
            trans=root_translations_t,
            betas=betas_t
        )

    # body_out.Jtr => (final_count, num_joints, 3)
    pose_seq_np = body_out.Jtr.cpu().numpy()

    # -------------------------------------------------------------------------
    # 10) Apply TRANS_MATRIX to reorder axes
    # -------------------------------------------------------------------------
    pose_seq_np_n = np.dot(pose_seq_np, TRANS_MATRIX)

    # Return the same style as amass_to_pose
    return pose_seq_np_n, fps

###############################################################################
# 3) GET 4DHUMANS PATHS
###############################################################################
def get_4dhumans_paths(dataset_dir):
    """
    Example function that scans `dataset_dir` for .pkl files, grouping them by 
    the first subdirectory (similar to how get_amass_paths works).

    Returns: (group_path, dataset_names)
        group_path: list of lists (each list is the .pkl paths for one dataset_name)
        dataset_names: parallel list of dataset names
    """
    paths_by_dataset = {}
    
    for root, dirs, files in os.walk(dataset_dir):
        for name in files:
            if not name.endswith(".pkl"):
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

###############################################################################
# 4) PROCESS 4DHUMANS DATASET
###############################################################################
def process_4dhumans_dataset(dataset_dir,
                             dataset_name=None,
                             output_dir=None,
                             save_intermediate=False,
                             device=None,
                             body_models_dir="./body_models"):
    """
    Process 4DHumans dataset and return joint positions in the same style
    as process_amass_dataset. We:
      - Gather .pkl paths
      - For each path, call `humans4d_to_pose(...)`
      - Optionally save .npy results
      - Return a dict of {file_path: pose_array}

    Args:
        dataset_dir (str): root directory containing .pkl data
        dataset_name (str): optional, name of subfolder to process
        output_dir (str): if saving to disk, location to store .npy
        save_intermediate (bool): if True, save .npy to disk
        device (str or torch.device): e.g. "cuda" or "cpu"
        body_models_dir (str): path to the SMPL-H body models

    Returns:
        results (dict): { file_path: (num_frames_ds, num_joints, 3) array }
    """
    # 1) Initialize SMPL-H models
    male_bm, female_bm = initialize_body_models(body_models_dir=body_models_dir, device=device)
    
    # 2) Gather .pkl paths
    group_path, dataset_names = get_4dhumans_paths(dataset_dir)
    print("Datasets found:", dataset_names)
    if len(group_path) > 0 and len(group_path[0]) > 0:
        print("Example pkl path from first dataset:", group_path[0][0])

    # Stats
    all_count = sum([len(paths) for paths in group_path])
    cur_count = 0

    results = {}

    for paths, current_dataset in zip(group_path, dataset_names):
        if not paths:
            continue
        print(f"\n--- Processing dataset: {current_dataset} ---")

        # Skip if a specific dataset is requested and this isn't it
        if dataset_name and current_dataset != dataset_name:
            continue

        pbar = tqdm(paths)
        pbar.set_description(f"Processing: {current_dataset}")

        iteration_count = 0
        fps_report = 0

        for pkl_path in pbar:
            # Call humans4d_to_pose
            pose_data, current_fps = humans4d_to_pose(pkl_path, male_bm, female_bm, device)
            fps_report = current_fps

            if pose_data is not None:
                results[pkl_path] = pose_data
                if save_intermediate and output_dir:
                    # Mirror directory structure
                    save_path = pkl_path.replace(dataset_dir, output_dir)
                    # Replace .pkl with .npy
                    if save_path.endswith(".pkl"):
                        save_path = save_path[:-3] + "npy"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, pose_data)

            iteration_count += 1
            cur_count += 1

        print(f"Processed / All (fps {fps_report}): {cur_count}/{all_count}")

    return results

###############################################################################
# 5) MAIN
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process 4DHumans .pkl data in a style similar to AMASS.")
    parser.add_argument("--input_dir",  type=str, required=True, help="Directory containing 4DHumans .pkl data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed .npy data")
    parser.add_argument("--dataset",    type=str, help="Specific dataset subfolder to process (optional)")
    parser.add_argument("--body_models_dir", type=str, default="./body_models",
                        help="Directory containing SMPL-H body models (male/female)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run computations on ('cuda', 'cuda:0', 'cpu', etc.)")

    args = parser.parse_args()

    # Call the main processing function
    process_4dhumans_dataset(
        dataset_dir=args.input_dir,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        save_intermediate=True,
        device=args.device,
        body_models_dir=args.body_models_dir
    )
