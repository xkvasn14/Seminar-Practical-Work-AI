import csv
from os import makedirs
from os.path import isdir
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

"""
VERY IMPORTANT BEFORE USE
HELP -> EDIT CUSTOM PROPERTIES -> add this:
idea.max.content.load.filesize=25000
idea.max.intellisense.filesize=25000
"""

"""Some global variables"""


# vivid_colors = [
#         "#e6194b",  # Red
#         "#7097BB",  # Midnight Blue - step 2
#         "#5885AF",  # Blue Gray - step 3
#         "#F1580E",  # Grapefruit
#         "#911eb4",  # Purple
#         "#274472",  # Dark Blue - step 1
#         "#BE42B1",  # Dark Pink - state
#         "#bcf60c",  # Lime
#         "#F5A01F",  # Gold
#         "#008080",  # Teal
#         "#e6beff"  # Lavender
#     ]

# vivid_colors = [
#         "#9e0142",  # Red
#         "#d53e4f",  # Midnight Blue - step 2
#         "#f46d43",  # Blue Gray - step 3
#         "#fdae61",  # Grapefruit
#         "#fafa5f",  # Purple
#         "#e6f598",  # Dark Blue - step 1
#         "#abdda4",  # Dark Pink - state
#         "#66c2a5",  # Lime
#         "#3288bd",  # Gold
#         "#3259bd",  # Teal
#         "#5e4fa2"  # Lavender
#     ]

def get_vivid_colors():
    dataset_path = "data_in_use/tmp_file.csv"
    dataset_df = pd.read_csv(dataset_path)
    dataset_labels = dataset_df["Cluster"].values.astype(int)
    unique_labels = pd.unique(dataset_labels)
    num_classes = len(unique_labels)
    vivid_colors = sns.mpl_palette("viridis", num_classes)
    return vivid_colors


# vivid_colors = sns.mpl_palette("Paired", 11)


def remove_unwanted_points(arr, skeleton_baselines):
    """
    Removes unwanted points from the array based on skeleton_baselines.

    Parameters:
        arr (np.ndarray): The data array with columns corresponding to points.
        skeleton_baselines (list): A list of point labels for each column in `arr`.
        points_to_remove (list): A list of point labels to remove.

    Returns:
        np.ndarray: The array with unwanted points removed.
    """
    # points to remove arms and hands
    points_to_remove = [
        'Skeleton Baseline_41:LShoulder',
        'Skeleton Baseline_41:LUArm',
        'Skeleton Baseline_41:LFArm',
        'Skeleton Baseline_41:LHand',
        'Skeleton Baseline_41:RShoulder',
        'Skeleton Baseline_41:RUArm',
        'Skeleton Baseline_41:RFArm',
        'Skeleton Baseline_41:RHand'
    ]

    # Find indices of columns to remove
    # remove arms and hands since we are doing just squats we will need legs and back
    list_to_remove = [
        i for i, label in enumerate(skeleton_baselines)
        if label in points_to_remove
    ]

    # Remove the columns
    arr_filtered = np.delete(arr, list_to_remove, axis=1)
    skeleton_baselines_filtered = [label for i, label in enumerate(skeleton_baselines) if i not in list_to_remove]

    return arr_filtered, skeleton_baselines_filtered


def data_parser(csv_file):
    """
    Reads, parses and saves data into new csv file
    :param csv_file: path to csv_file with 3D data from OptiTrack
    :returns: data_entry in format: (file path, torso position)
    """
    # Remove first two rows to create (x,y) array with labels and points
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        next(reader)

        os.makedirs("data_in_use", exist_ok=True)
        with open(f"data_in_use/data_{os.path.basename(csv_file).split('.')[0]}.csv", mode='w', encoding='utf-8',
                  newline='') as file:
            writer = csv.writer(file)
            for row in reader:
                writer.writerow(row)

    # Read data now as array and remove all unnecessary data
    arr = np.loadtxt(f"data_in_use/data_{os.path.basename(csv_file).split('.')[0]}.csv", delimiter=',', dtype=str)

    # cut all: non Bone, non Position, plc gimme torso position (2)
    index = 0
    while (True):
        if arr.shape[1] == index:
            break
        if arr[0][index] != "Bone":
            arr = np.delete(arr, index, axis=1)
            continue
        if arr[3][index] != "Position":
            arr = np.delete(arr, index, axis=1)
            continue
        index += 1

    skeleton_baselines = arr[1].copy()
    # delete 0-Bone, 1-SkelyBaseline, 2-ID, 3-Position, 4-XYZ label
    arr = np.delete(arr, [0, 1, 2, 3], axis=0)

    # REMOVE UNWANTED POINTS
    arr, skeleton_baselines = remove_unwanted_points(arr, skeleton_baselines=skeleton_baselines)

    # save the data into csv
    with open(f"data_in_use/data_{os.path.basename(csv_file).split('.')[0]}.csv", mode='w', encoding="utf-8",
              newline='') as file:
        writer = csv.writer(file)
        writer.writerows(arr)
    print(f"Parsed data saved to data_in_use/data_{os.path.basename(csv_file).split('.')[0]}.csv")

    # return skeleton torso number
    skeleton_baselines = skeleton_baselines[::3]
    for i, sk in enumerate(skeleton_baselines):
        if skeleton_baselines[i] == "Skeleton Baseline_41:Chest":
            return (f"data_in_use/data_{os.path.basename(csv_file).split('.')[0]}.csv", i)


def data_normalization(data_entry):
    """
    Normalizes data in the csv file (x,y,z,x,...,z)
    :param csv_file: path to parsed csv_file
    """

    csv_file, torso_position = data_entry
    # torso position needs to be multiplied by 3, because of the (x,y,z) format to get the torso position index
    # skip first row for XYZ labels

    data = np.loadtxt(csv_file, delimiter=',', dtype=float, skiprows=1)
    arr = []
    for row in data:
        counter = 0
        arr_row = []
        arr_xyz = []
        for r in row:
            if counter == 3:
                counter = 0
                arr_row.append(arr_xyz)
                arr_xyz = []
            else:
                arr_xyz.append(r)
                counter += 1
        arr.append(arr_row)

    arr = np.array(arr)
    # used to create testing data from original data
    # arr = arr + 1.56987
    # arr = arr * 1.0008
    chest_adjusted_data = arr - arr[:, torso_position:torso_position + 1, :]
    flattened_data = chest_adjusted_data.reshape(-1, 3)

    mins = flattened_data.min(axis=0)
    maxs = flattened_data.max(axis=0)

    normalized_data = (flattened_data - mins) / (maxs - mins)
    normalized_data = normalized_data.reshape(chest_adjusted_data.shape)

    # data_animation(arr)

    reshaped_data = normalized_data.reshape(normalized_data.shape[0], -1)

    if reshaped_data[0][0] != "X":
        xyz_labels = []
        for i in range(reshaped_data.shape[1]):
            if (i % 3) == 0:
                xyz_labels.append("X")
            elif (i % 3) == 1:
                xyz_labels.append("Y")
            elif (i % 3) == 2:
                xyz_labels.append("Z")
        xyz_labels = np.array(xyz_labels)
        reshaped_data = np.vstack((xyz_labels, reshaped_data))

    np.savetxt(csv_file, reshaped_data, delimiter=",", encoding="utf-8", fmt="%s")
    print(f"Normalization data saved to {csv_file}")


def data_annotation(norm_csv_path="data_in_use", create_dataset=True):
    """
    uses k-means to separate different postures and annotates the saved data, opt: displays them
    :param norm_csv_path: path to folder/file with normalized data
    :param visualize: should kmean algorithm do visualization
    :param save_fig: should kmean algo save the figure or display it
    :param create_dataset: should it create one superfile for kmean or perform kmean independently on each file if folder given
    """
    # like the stuff couldn't be done simply...
    loop = 0
    norm_csv_paths = []
    tmp_file = "data_in_use/tmp_file.csv"
    if isdir(norm_csv_path):
        for (root, dirs, files) in os.walk(norm_csv_path):
            if len(files) == 1:
                loop = 1
                norm_csv_paths.append(root + "/" + files[0])
            else:
                if create_dataset:
                    loop = 1
                    norm_csv_paths.append(tmp_file)
                    concatenate_files_with_condition([root + "/" + file for file in files], tmp_file)
                else:
                    loop = len(files)
                    for file in files:
                        with open(root + "/" + file, 'r') as f:
                            f_content = f.read()
                            norm_csv_paths.append(root + "/" + file)
    else:
        loop = 1
        norm_csv_paths.append(norm_csv_path)

    # perform k-means
    for i in range(loop):
        kmeans(norm_csv_paths[i])

    if create_dataset:
        separate_tmp_file_into_files()


def kmeans(norm_csv_path, n_clusters=5, visualize=True, save_fig=True):
    # Load the normalized data
    data = pd.read_csv(norm_csv_path)
    # data = pd.read_csv("data_in_use/data_12343658_1_023.csv")

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)

    # Annotate the data with cluster labels
    data['Cluster'] = cluster_labels

    # Save the annotated data
    data.to_csv(norm_csv_path, index=False)
    print(f"Annotated data saved to {norm_csv_path}")

    cmap = ListedColormap(get_vivid_colors())

    # Optional: Visualize the clusters in 3D
    if visualize:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            data.iloc[:, 0],  # X-axis (normalized x)
            data.iloc[:, 1],  # Y-axis (normalized y)
            data.iloc[:, 2],  # Z-axis (normalized z)
            c=data['Cluster'], cmap=cmap, s=50
        )

        ax.set_title("3D Cluster Visualization")
        ax.set_xlabel("X (Normalized)")
        ax.set_ylabel("Y (Normalized)")
        ax.set_zlabel("Z (Normalized)")
        cbar = plt.colorbar(scatter, label='Cluster Label', pad=0.15)
        cbar.ax.set_yticks([])

        if save_fig:
            name_file = norm_csv_path.split('.')[0].split('/')[1]
            makedirs("figures", exist_ok=True)
            plt.savefig(f"figures/fig_{name_file}")
            plt.close()
        else:
            plt.show()


def concatenate_files_with_condition(file_paths, output_path):
    how_many_cols = 15
    with open(file_paths[0], 'r') as readfile:
        first_line = readfile.readline().strip()
        how_many_cols = first_line.count("X")

    outputfile_string = ""
    for i in range(how_many_cols):
        outputfile_string += "X,Y,Z,"
    outputfile_string = outputfile_string.rstrip(",") + "\n"

    with open(output_path, 'w') as outfile:
        outfile.write(outputfile_string)
        for idx, file_path in enumerate(file_paths):
            with open(file_path, 'r') as f:
                if any("Cluster" in line for line in f):
                    continue

            with open(file_path, 'r') as infile:
                lines = infile.readlines()
                lines = lines[1:]
                outfile.writelines(lines)
    print("Created one super file")


def separate_tmp_file_into_files(visualize: bool = True, save_fig: bool = True):
    tmp_file = "data_in_use/tmp_file.csv"
    data_in_use = "data_in_use"
    sizes_and_files = []

    for (root, dirs, files) in os.walk(data_in_use):
        for file in files:
            if file.__contains__("tmp_file"):
                continue
            data = pd.read_csv(root + "/" + file)
            sizes_and_files.append((data.shape[0], root + "/" + file))

    large_df = pd.read_csv(tmp_file)
    start_idx = 0
    for length, filename in sizes_and_files:
        subset_df = large_df.iloc[start_idx:start_idx + length]
        subset_df.to_csv(filename, index=False, header=True)
        start_idx += length

    print("Data successfully split into original files with headers included.")

    cmap = ListedColormap(get_vivid_colors())
    norm = plt.Normalize(vmin=0, vmax=10)

    # Optional: Visualize the clusters in 3D
    if visualize:
        for length, filename in sizes_and_files:
            data = pd.read_csv(filename)
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(
                data.iloc[:, 0],  # X-axis (normalized x)
                data.iloc[:, 1],  # Y-axis (normalized y)
                data.iloc[:, 2],  # Z-axis (normalized z)
                c=data['Cluster'], cmap=cmap, s=50, norm=norm
            )

            ax.set_title("3D Cluster Visualization")
            ax.set_xlabel("X (Normalized)")
            ax.set_ylabel("Y (Normalized)")
            ax.set_zlabel("Z (Normalized)")
            cbar = plt.colorbar(scatter, label='Cluster Label', pad=0.15)
            cbar.ax.set_yticks([])

            if save_fig:
                name_file = filename.split('.')[0].split('/')[1]
                makedirs("figures", exist_ok=True)
                plt.savefig(f"figures/fig_{name_file}")
                print(f"Figure {name_file} saved into figures")
                plt.close()
            else:
                plt.show()


def data_animation(arr):
    # Time range for the sequence
    start_time, end_time = 0, arr.size

    # Create the figure and 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Initial scatter plot (empty)
    scatter = ax.scatter([], [], [], c='blue', s=50)

    # Setting static axis limits for clarity
    ax.set_xlim(0, 1000)  # Normalized range for X
    ax.set_ylim(0, 1000)  # Normalized range for Y
    ax.set_zlim(0, 1000)  # Normalized range for Z
    ax.set_title("Normalized 3D Data Sequence")
    ax.set_xlabel("X (Normalized)")
    ax.set_ylabel("Y (Normalized)")
    ax.set_zlabel("Z (Normalized)")

    # Update function for animation
    def update(frame):
        points = arr[frame]
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        ax.set_title(f"Time Step: {frame}")

    ani = FuncAnimation(fig, update, frames=range(start_time, end_time + 1), interval=100)

    # Save as GIF or display
    # ani.save("normalized_data_sequence.gif", writer=PillowWriter(fps=10))
    plt.show()


def posture_analysis_pipeline(data_folder="../data", do_annotation=True, do_normalization=True):
    """
    Pipeline of the posture analysis, preparing data, normalization, k-means annotation
    :param data_folder: path to the data folder with the csv files
    """
    if not isdir(data_folder):
        print("path must be directory!")
        return
    if do_normalization:
        for (root, dirs, file) in os.walk(data_folder):
            for f in file:
                if ".csv" in f:
                    try:
                        data_entry = data_parser(root + "/" + f)
                        data_normalization(data_entry)
                    except:
                        print("Something went wrong :)")
    if do_annotation:
        data_annotation()


if __name__ == "__main__":
    posture_analysis_pipeline("../data", do_normalization=True, do_annotation=True)
