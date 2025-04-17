import os
import argparse
import joblib
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from HumanML3D.human_body_prior.tools.omni_tools import copy2cpu as c2c
from HumanML3D.human_body_prior.body_model.body_model import BodyModel
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Constants
TRANS_MATRIX = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0]])
TARGET_FPS = 20

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

def get_4dhumans_paths(dataset_dir):
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

from pytorch3d.transforms import matrix_to_axis_angle
def convert_body_pose(rot_matrices):
    B, N, _, _ = rot_matrices.shape
    rot_matrices_flat = rot_matrices.view(-1, 3, 3)
    axis_angles_flat = matrix_to_axis_angle(rot_matrices_flat)
    return axis_angles_flat.view(B, N * 3)

def humans4d_to_pose(
    file_path,
    male_bm,                
    female_bm,              
    device='cuda'
):
    fdh_results = joblib.load(file_path)
    fps = fdh_results.get("fps", 30)
    track_data = defaultdict(lambda: {
        "global_orient": [],
        "body_pose": [],
        "translation": []
    })

    for frame_key, frame_data in fdh_results.items():
        if not isinstance(frame_data, dict):
            continue
        for idx, (tid, smpl_dict, cam_trans) in enumerate(
            zip(
                frame_data["tid"],
                frame_data["smpl"],
                frame_data["camera"],
            )
        ):
            # Update track data
            track = track_data[tid]
            track["global_orient"].append(torch.tensor(smpl_dict["global_orient"]))
            track["body_pose"].append(torch.tensor(smpl_dict["body_pose"]))
            track["translation"].append(torch.tensor(cam_trans))

    if len(track_data) == 0:
        return None, 0
    
    best_tid = max(track_data.items(), key=lambda x: len(x[1]["body_pose"]))[0]
    chosen = track_data[best_tid]
    global_orient = torch.stack(chosen["global_orient"])[::max(fps // TARGET_FPS, 1)].to(device)
    body_pose = torch.stack(chosen["body_pose"])[::max(fps // TARGET_FPS, 1), :21].to(device)
    body_pose = convert_body_pose(body_pose)
    translation = torch.stack(chosen["translation"])[::max(fps // TARGET_FPS, 1)].to(device)
    # betas = torch.zeros((global_orient.shape[0], 10), device=device)
    bm = male_bm
    with torch.no_grad():
            out = bm(global_orient=global_orient,
                    pose_body=body_pose,
                    # betas=betas,
                    transl=translation)
    joints = out.Jtr.cpu().numpy()
    joints = np.dot(joints, TRANS_MATRIX)
    return joints, TARGET_FPS, bm

def process_4dhumans_dataset(input_dir, output_dir, body_models_dir, device='cuda'):
    male_bm, female_bm = initialize_body_models(body_models_dir, device)
    all_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pkl"):
                all_paths.append(os.path.join(root, file))

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for path in tqdm(all_paths, desc="Processing 4DHumans"):
        joints, fps = humans4d_to_pose(path, male_bm, female_bm, device)
        if joints is None:
            continue

        # Store in memory as well
        results[path] = joints

    return results


# ----------------------------------------------------------------------
#  1.  bring joints back to the native SMPL frame
# ----------------------------------------------------------------------
def to_smpl_frame(joints_np: np.ndarray) -> torch.Tensor:
    """
    joints_np : (T, 22, 3) after recover_full_motion_from_data,
                i.e. still rotated by TRANS_MATRIX
    returns    : (T, 22, 3) torch tensor in the SMPL coordinate frame
    """
    invT = torch.as_tensor(np.linalg.inv(TRANS_MATRIX), dtype=torch.float32)
    j_torch = torch.as_tensor(joints_np, dtype=torch.float32)          # (T,22,3)
    return torch.einsum('ij,tbj->tbi', invT, j_torch)                  # undo rotation


# ----------------------------------------------------------------------
#  2.  inverse‑kinematics solver (per frame, differentiable)
# ----------------------------------------------------------------------
def joints_to_smpl_params(joints_smpl: torch.Tensor,
                          body_model: BodyModel,
                          n_iters: int = 80,
                          lr: float = 5e-3,
                          device: str = 'cuda'):
    """
    joints_smpl : (T, 22, 3)  in the SMPL coordinate frame
    body_model  : SAME BodyModel instance that was used in humans4d_to_pose
                  (same weights, betas, etc.)
    returns
        global_orient (T,3), body_pose (T,63), translation (T,3)
    """
    joints_smpl = joints_smpl.to(device)
    body_model  = body_model.to(device)

    T = joints_smpl.shape[0]

    g_list, p_list, t_list = [], [], []
    init_orient = torch.zeros(1, 3, device=device)   # start pose
    init_pose   = torch.zeros(1, 63, device=device)
    init_trans  = torch.zeros(1, 3, device=device)

    for t in tqdm(range(T), desc='IK solve'):
        target = joints_smpl[t, :22].detach()          # (22,3)

        # ---- variables to optimise ------------------------------------
        g = init_orient.clone().requires_grad_(True)
        p = init_pose  .clone().requires_grad_(True)
        x = init_trans .clone().requires_grad_(True)

        opt = torch.optim.Adam([g, p, x], lr=lr)

        for _ in range(n_iters):
            opt.zero_grad()
            out = body_model(global_orient=g, pose_body=p, transl=x)
            loss = ((out.Jtr[0, :22] - target) ** 2).mean()
            loss.backward()
            opt.step()

        # keep solution & use it as warm start for next frame
        init_orient, init_pose, init_trans = g.detach(), p.detach(), x.detach()
        g_list.append(g.detach().cpu())
        p_list.append(p.detach().cpu())
        t_list.append(x.detach().cpu())

    return (torch.cat(g_list),    # (T,3)
            torch.cat(p_list),    # (T,63)
            torch.cat(t_list))    # (T,3)

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
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
        body_models_dir=args.body_models_dir
    )
