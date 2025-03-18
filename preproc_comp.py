import codecs as cs
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from os.path import join as pjoin
import os.path
import glob


def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


# index_path = './index.csv'
save_dir = "./joints_comp"
# index_file = pd.read_csv(index_path)
all_files = glob.glob("./pose_data/**/*.npy", recursive=True)
all_out_files = [pjoin(save_dir, *f.split("/")[2:]) for f in all_files]
all_fnames = [f.split("/")[-1] for f in all_files]
total_amount = len(all_files)
fps = 20


for i in tqdm(range(total_amount)):
    source_path = all_files[i]
    save_path = all_out_files[i]
    msave_path = pjoin(*save_path.split("/")[:-1], "M_" + all_fnames[i])

    if not os.path.exists(source_path):
        print(source_path)
        continue

    data = np.load(source_path)
    if "humanact12" not in source_path:
        data[..., 0] *= -1

    data_m = swap_left_right(data)

    os.makedirs(os.path.split(save_path)[0], exist_ok=True)

    np.save(save_path, data)
    np.save(msave_path, data_m)
