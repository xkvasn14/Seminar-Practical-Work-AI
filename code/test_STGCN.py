import multiprocessing
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split  # if needed
# Import your STGCN implementation; it must define a class 'STGCN'
from STGCN import STGCN


# Define your dataset class
class SkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data  # Shape: (N, C, T, V)
        self.labels = labels  # Shape: (N,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_stgcn_model(path: str, num_class: int, A: torch.Tensor, in_channels: int = 3,
                     dropout: float = 0.3, kernel_size: int = 9, edge_importance_weighting: bool = False) -> nn.Module:
    """
    Instantiate an STGCN model, load weights from a .pth checkpoint, and return it.
    """
    model = STGCN(in_channels=in_channels, num_class=num_class, A=A,
                  edge_importance_weighting=edge_importance_weighting, dropout=dropout, kernel_size=kernel_size)
    # model = STGCN(num_class=num_class, in_channels=in_channels, num_nodes=15)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    print(f"Loaded model from {path}")
    return model


def test_stgcn(normalized_data_path: str = "data_in_use/data_12343658_1_022.csv",
               model_stgcn_path: str = "best_stgcn.pth", save_csv: bool = True):
    multiprocessing.freeze_support()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the CSV and parse features and label
    df = pd.read_csv(normalized_data_path)
    labels = df['Cluster'].values.astype(int)
    features = df.drop(columns=['Cluster']).values.astype(float)  # shape: (6690, 45)

    # Reshape the features:
    # Each row is one time step with 45 features (3 channels × 15 vertices).
    # First, reshape to (6690, 3, 15)
    data = df.drop(columns=['Cluster'])  # Shape (6690, 45)

    x_columns = [col for col in data.columns if col.startswith("X")]
    y_columns = [col for col in data.columns if col.startswith("Y")]
    z_columns = [col for col in data.columns if col.startswith("Z")]

    data_x = data[x_columns].values  # Shape (6690, 15)
    data_y = data[y_columns].values  # Shape (6690, 15)
    data_z = data[z_columns].values  # Shape (6690, 15)

    # Reorder columns
    data = data[x_columns + y_columns + z_columns]
    reshaped_data = np.stack([data_x, data_y, data_z], axis=0)
    # We want the input to STGCN to be (batch, channels, time, vertices) i.e. (1, 3, 6690, 15).
    # Since our current tensor is (6690, 3, 15) with time first, we transpose it to get (3, 6690, 15)
    features = reshaped_data.transpose(1, 0, 2)  # shape: (3, 6690, 15)
    # Then add the batch dimension:
    # features = features[None, ...]  # shape: (1, 3, 6690, 15)
    # print("Reshaped features shape:", features.shape)
    # print(features.shape)
    # Convert to torch tensors
    torch_sequence = torch.from_numpy(features).float()
    torch_label = torch.from_numpy(labels).long()
    # torch_label = torch.tensor([label], dtype=torch.long)  # label shape: (1,)

    # Create dataset and loader
    dataset = SkeletonDataset(torch_sequence, torch_label)
    # print(torch_sequence.shape)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # Create an adjacency matrix A (for example, an identity matrix; adjust as needed)
    A = torch.eye(15, dtype=torch.float32)

    # AGAIN the number of classes must equal number of k-mean classes
    model = load_stgcn_model(model_stgcn_path, num_class=11, A=A, in_channels=3, dropout=0.3, kernel_size=9)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss, total_correct = 0, 0
    predictions = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            # print(batch_x.shape)
            # print(batch_y.shape)
            # print(outputs.shape)
            # exit()
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            predictions.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / len(loader.dataset)
    print(f"Test Loss: {avg_loss:.4f}  Acc: {accuracy:.4f}")

    if save_csv:
        # Because the entire sequence received one prediction, save that prediction
        df["Prediction"] = [predictions[0]] * len(df)
        path_components = normalized_data_path.split("/")
        file = path_components[-1]
        file_name, ext = file.split(".")
        file_name += "_stgcn"
        file = file_name + "." + ext
        path_components[-1] = file
        out_path = "/".join(path_components)
        print(f"Saving into {out_path}")
        df.to_csv(out_path, index=False)


if __name__ == '__main__':
    test_stgcn(normalized_data_path=f"data_in_use/data_12343658_1_012.csv", model_stgcn_path="best_stgcn.pth",
               save_csv=True)
    test_stgcn(normalized_data_path=f"data_in_use/data_12343658_1_022.csv", model_stgcn_path="best_stgcn.pth",
               save_csv=True)
