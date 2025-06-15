import multiprocessing
import timeit

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from STGCN import STGCN
from sklearn.model_selection import train_test_split


class SkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data  # Shape: (N, C, T, V)
        self.labels = labels  # Shape: (N,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += data.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += data.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def test(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += data.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    print(f"Test Val Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def train_stgcn(dataset_path: str = "data_in_use/tmp_file.csv", model_path: str = "best_stgcn.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = pd.read_csv(dataset_path)
    features = df.drop(columns=['Cluster']).values.astype(float)

    x_columns = [col for col in df.columns if col.startswith("X")]
    y_columns = [col for col in df.columns if col.startswith("Y")]
    z_columns = [col for col in df.columns if col.startswith("Z")]

    data_x = df[x_columns].values  # shape: (6690, 15)
    data_y = df[y_columns].values  # shape: (6690, 15)
    data_z = df[z_columns].values  # shape: (6690, 15)

    reshaped_data = np.stack([data_x, data_y, data_z], axis=0)
    skeleton_data = reshaped_data
    skeleton_labels = df['Cluster'].values.astype(int)
    skeleton_data = np.transpose(skeleton_data, (1, 0, 2))

    train_seqs, temp_seqs, train_labels, temp_labels = train_test_split(
        skeleton_data, skeleton_labels, test_size=0.3, random_state=42, stratify=skeleton_labels
    )
    val_seqs, test_seqs, val_labels, test_labels = train_test_split(
        temp_seqs, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    torch_train_seqs = torch.from_numpy(train_seqs).float()
    torch_train_labels = torch.from_numpy(train_labels).long()
    torch_val_seqs = torch.from_numpy(val_seqs).float()
    torch_val_labels = torch.from_numpy(val_labels).long()
    torch_test_seqs = torch.from_numpy(test_seqs).float()
    torch_test_labels = torch.from_numpy(test_labels).long()

    # Instantiate Dataset and DataLoader
    train_dataset = SkeletonDataset(torch_train_seqs, torch_train_labels)
    val_dataset = SkeletonDataset(torch_val_seqs, torch_val_labels)
    test_dataset = SkeletonDataset(torch_test_seqs, torch_test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    input_channels = skeleton_data.shape[1]
    num_classes = len(set(skeleton_labels))
    A = torch.ones((15, 15), dtype=torch.float32)
    model = STGCN(in_channels=input_channels, num_class=num_classes, A=A).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    epochs = 20
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)
        print(f"Saved best model at epoch {epoch}")

    test(model, test_loader, criterion, device)


if __name__ == "__main__":
    s = timeit.default_timer()
    train_stgcn(dataset_path="data_in_use/tmp_file.csv", model_path="best_stgcn.pth")
    e = timeit.default_timer()
    print("ELAPSED:", e - s)
