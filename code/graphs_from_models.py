import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, Normalize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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


def make_graph_from_model(data_model_path: str = "data_in_use/", save_fig: bool = True):
    data = pd.read_csv(data_model_path)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    cmap = ListedColormap(get_vivid_colors())
    norm = plt.Normalize(vmin=0, vmax=10)

    mistakes = data["Cluster"] != data["Prediction"]

    data_correct = data[~mistakes]
    data_mistakes = data[mistakes]

    scatter_correct = ax.scatter(
        data_correct.iloc[:, 0],  # X-axis (normalized x)
        data_correct.iloc[:, 1],  # Y-axis (normalized y)
        data_correct.iloc[:, 2],  # Z-axis (normalized z)
        c=data_correct['Cluster'], cmap=cmap, norm=norm,
        s=50, label="Correct Prediction"
    )

    scatter_mistake = ax.scatter(
        data_mistakes.iloc[:, 0],  # X-axis (normalized x)
        data_mistakes.iloc[:, 1],  # Y-axis (normalized y)
        data_mistakes.iloc[:, 2],  # Z-axis (normalized z)
        color='red', s=50, label="Mis-prediction", alpha=0.8
    )

    ax.set_title("3D Cluster Visualization")
    ax.set_xlabel("X (Normalized)")
    ax.set_ylabel("Y (Normalized)")
    ax.set_zlabel("Z (Normalized)")
    cbar = plt.colorbar(scatter_correct, ax=ax, label='Cluster Label', pad=0.15)
    cbar.ax.set_yticks([])
    ax.legend(loc="upper left")

    if save_fig:
        name_file = data_model_path.split('.')[0].split('/')[1]
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/fig_{name_file}")
        print(f"Figure {name_file} saved into figures")
        plt.close()
    else:
        plt.show()


def make_graphs(dataset_path: str = "data_in_use"):
    paths = os.listdir(dataset_path)

    for path in paths:
        if path.endswith(".csv"):
            if "_lstm" in path or "_dsvm" in path or "_stgcn" in path:
                make_graph_from_model(dataset_path + "/" + path)
        else:
            continue


def concatenate_files(dataset_path):
    # Initialize lists to hold data for each type
    file_types = {
        "_lstm": [],
        "_dsvm": [],
        "_stgcn": []
    }

    # List all files in the dataset path
    paths = os.listdir(dataset_path)

    # Identify and categorize files
    for path in paths:
        full_path = os.path.join(dataset_path, path)
        if os.path.isfile(full_path) and path.endswith(".csv"):
            for key in file_types:
                if key in path:
                    file_types[key].append(full_path)

    # Process each file type
    for key, file_list in file_types.items():
        if file_list:
            combined_df = pd.concat([pd.read_csv(file) for file in file_list], ignore_index=True)
            output_file = os.path.join(dataset_path, f"tmp_file{key}.csv")
            combined_df.to_csv(output_file, index=False)
            print(f"Saved {output_file} with {len(file_list)} files.")
        else:
            print(f"No files found for {key}.")

    return file_types


def create_confusion_matrix(file_path, output_path="confusion_matrix.png", save_fig: bool = True):
    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Check if the required columns exist
    if 'Cluster' not in df.columns or 'Prediction' not in df.columns:
        print("The file must contain 'Cluster' and 'Prediction' columns.")
        return

    # Extract the 'Cluster' and 'Prediction' columns
    clusters = df['Cluster']
    predictions = df['Prediction']

    all_classes = sorted(set(clusters.unique()).union(predictions.unique()))

    # Generate confusion matrix
    cm = confusion_matrix(clusters, predictions, labels=all_classes)
    cm_df = pd.DataFrame(cm, index=all_classes, columns=all_classes)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Classes")
    plt.ylabel("True Classes")
    plt.tight_layout()

    if save_fig:
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/fig_{output_path}")
        print(f"Figure {output_path} saved into figures")
        plt.close()
    else:
        plt.show()


def make_confusion_matrices(dataset_path: str = "data_in_use"):
    file_lists = concatenate_files(dataset_path)
    create_confusion_matrix("data_in_use/tmp_file_lstm.csv", "confusion_matrix_lstm.png")
    create_confusion_matrix("data_in_use/tmp_file_dsvm.csv", "confusion_matrix_dsvm.png")
    create_confusion_matrix("data_in_use/tmp_file_stgcn.csv", "confusion_matrix_stgcn.png")


if __name__ == "__main__":
    make_graphs("data_in_use")
    make_confusion_matrices("data_in_use")
