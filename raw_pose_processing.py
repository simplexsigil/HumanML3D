import sys, os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm


from human_body_prior.tools.omni_tools import copy2cpu as c2c

os.environ["PYOPENGL_PLATFORM"] = "egl"


if len(sys.argv) > 2:
    dataset_top = sys.argv[1]
    cuda_dev = sys.argv[2]
    print(dataset_top)
else:
    dataset_top = "ACCAD"
    print("Not enough argument provided.")

# Choose the device to run the body model on.
comp_device = torch.device(f"cuda:0")


from human_body_prior.body_model.body_model import BodyModel

male_bm_path = "./body_models/smplh/male/model.npz"
male_dmpl_path = "./body_models/dmpls/male/model.npz"

female_bm_path = "./body_models/smplh/female/model.npz"
female_dmpl_path = "./body_models/dmpls/female/model.npz"

num_betas = 10  # number of body parameters
num_dmpls = 8  # number of DMPL parameters

male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to(
    comp_device
)
faces = c2c(male_bm.f)

female_bm = BodyModel(
    bm_fname=female_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=female_dmpl_path
).to(comp_device)

paths = []
folders = []
dataset_names = []
for root, dirs, files in os.walk("./amass_data"):
    #     print(root, dirs, files)
    #     for folder in dirs:
    #         folders.append(os.path.join(root, folder))
    folders.append(root)
    if "tars" in dirs:
        dirs.remove("tars")
    for name in files:
        if name in ["LICENSE.txt"]:
            continue
        dataset_name = root.split("/")[2]
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)
        paths.append(os.path.join(root, name))

save_root = "./pose_data"
save_folders = [folder.replace("./amass_data", "./pose_data") for folder in folders]
for folder in save_folders:
    os.makedirs(folder, exist_ok=True)
group_path = [[path for path in paths if name in path] for name in dataset_names]
print(len(group_path))

trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
ex_fps = 20


def amass_to_pose(src_path, save_path):
    bdata = np.load(src_path, allow_pickle=True)
    fps = 0
    try:
        fps = bdata["mocap_framerate"]
        frame_number = bdata["trans"].shape[0]
    except:
        #         print(list(bdata.keys()))
        return fps

    fId = 0  # frame id of the mocap sequence
    pose_seq = []
    if bdata["gender"] == "male":
        bm = male_bm
    else:
        bm = female_bm
    down_sample = int(fps / ex_fps)
    #     print(frame_number)
    #     print(fps)

    with torch.no_grad():
        for fId in range(0, frame_number, down_sample):
            root_orient = torch.Tensor(bdata["poses"][fId : fId + 1, :3]).to(
                comp_device
            )  # controls the global root orientation
            pose_body = torch.Tensor(bdata["poses"][fId : fId + 1, 3:66]).to(comp_device)  # controls the body
            pose_hand = torch.Tensor(bdata["poses"][fId : fId + 1, 66:]).to(
                comp_device
            )  # controls the finger articulation
            betas = torch.Tensor(bdata["betas"][:10][np.newaxis]).to(comp_device)  # controls the body shape
            trans = torch.Tensor(bdata["trans"][fId : fId + 1]).to(comp_device)
            body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, root_orient=root_orient)
            joint_loc = body.Jtr[0] + trans
            pose_seq.append(joint_loc.unsqueeze(0))
    pose_seq = torch.cat(pose_seq, dim=0)

    pose_seq_np = pose_seq.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)

    np.save(save_path, pose_seq_np_n)
    return fps


group_path = group_path
all_count = sum([len(paths) for paths in group_path])
cur_count = 0


import time

for paths in group_path:
    dataset_name = paths[0].split("/")[2]
    print(dataset_name)
    print(dataset_top)
    if dataset_name != dataset_top:
        continue
    pbar = tqdm(paths)
    pbar.set_description("Processing: %s" % dataset_name)
    fps = 0
    for path in pbar:
        save_path = path.replace("./amass_data", "./pose_data")
        save_path = save_path[:-3] + "npy"
        fps = amass_to_pose(path, save_path)

    cur_count += len(paths)
    print("Processed / All (fps %d): %d/%d" % (fps, cur_count, all_count))
    time.sleep(0.005)
