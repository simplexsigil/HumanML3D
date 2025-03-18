import numpy as np
import sys
import os
from os.path import join as pjoin
from tqdm import tqdm
import glob


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
# Define a function to calculate the mean and standard deviation of motion capture data files.
def mean_variance(data_dir, save_dir, joints_num):
    # Search and list all .npy files in the specified directory and its subdirectories.
    file_list = glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)
    data_list = []  # Create an empty list to hold data arrays from each file.

    # Loop through each file path in the file list.
    for file in tqdm(file_list):
        data = np.load(file)  # Load the data from the file.

        # Check if the loaded data contains any NaN values and skip if so.
        if np.isnan(data).any():
            print(file)  # Print the file name with NaN values.
            continue

        # Append the valid data array to the list.
        data_list.append(data)

    # Concatenate all the arrays from the data list into a single array for processing.
    data = np.concatenate(data_list, axis=0)
    print(data.shape)  # Print the shape of the concatenated data array.

    Mean = data.mean(
        axis=0
    )  # Compute the mean of the concatenated data along the first dimension (across all samples).
    Std = data.std(axis=0)  # Compute the standard deviation of the concatenated data along the first dimension.

    # The following blocks normalize standard deviations by averaging specific sections of the Std array.
    # These sections correspond to different types of data in the motion capture data vectors:
    # root rotation velocities, root linear velocities, root heights, relative invariant coordinates,
    # rotation data, local velocities, and foot contacts. The indices used reflect this structure.
    Std[0:1] = Std[0:1].mean() / 1.0
    # Normalize the second segment.
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4 : 4 + (joints_num - 1) * 3] = Std[4 : 4 + (joints_num - 1) * 3].mean() / 1.0
    Std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9] = (
        Std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9].mean() / 1.0
    )
    Std[4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3] = (
        Std[4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3].mean() / 1.0
    )
    Std[4 + (joints_num - 1) * 9 + joints_num * 3 :] = Std[4 + (joints_num - 1) * 9 + joints_num * 3 :].mean() / 1.0

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

    np.save(pjoin(save_dir, "Mean.npy"), Mean)
    np.save(pjoin(save_dir, "Std.npy"), Std)

    return Mean, Std


if __name__ == "__main__":
    dataset = "MIA"

    if dataset == "AMASS":
        data_dir = "./HumanML3D/new_joint_comp_vecs/"
        save_dir = "./HumanML3D/"
    elif dataset == "MIA":
        data_dir = "MIAHML3D/motion_vects/"
        save_dir = "MIAHML3D/"

    mean, std = mean_variance(data_dir, save_dir, 22)
    print(mean)
    print(std)
