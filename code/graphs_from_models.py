import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, Normalize

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

vivid_colors = sns.mpl_palette("viridis", 11)


def make_graph_from_model(data_model_path: str = "data_in_use/", save_fig: bool = True):
    data = pd.read_csv(data_model_path)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    cmap = ListedColormap(vivid_colors)
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
    plt.colorbar(scatter_correct, ax=ax, label='Cluster Label', pad=0.15)
    ax.legend(loc="upper left")

    if save_fig:
        name_file = data_model_path.split('.')[0].split('/')[1]
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/fig_{name_file}")
        print(f"Figure {name_file} saved into figures")
    else:
        plt.show()


def make_graphs(dataset_path:str="data_in_use"):
    paths = os.listdir(dataset_path)

    for path in paths:
        if path.endswith(".csv"):
            if "_lstm" in path or "_dsvm" in path or "_stgcn" in path or "_cnn" in path:
                make_graph_from_model(dataset_path + "/" + path)
        else:
            continue


if __name__ == "__main__":
    make_graphs("data_in_use")
