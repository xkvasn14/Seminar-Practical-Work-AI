"""
This file encapsulates all functionality.
- Has network architecture
- has training and evaluation
- has testing function with file saving functionality
"""

import os
import timeit
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SkeletonDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, device=torch.device("cpu")):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_accuracy = 0

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        accuracy = correct / total * 100
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_cnn.pth")
            print(f"Best model saved with accuracy: {best_accuracy:.2f}%")

    return model


def extract_features(data_frames):
    skeleton_data_list = []
    skeleton_labels_list = []

    for df in data_frames:
        x_columns = [col for col in df.columns if col.startswith("X")]
        y_columns = [col for col in df.columns if col.startswith("Y")]
        z_columns = [col for col in df.columns if col.startswith("Z")]
        data_x = df[x_columns].values
        data_y = df[y_columns].values
        data_z = df[z_columns].values
        skeleton_data = np.stack([data_x, data_y, data_z], axis=0)
        skeleton_labels = df['Cluster'].values.astype(int)
        skeleton_data = np.transpose(skeleton_data, (1, 0, 2))
        skeleton_data_list.append(skeleton_data)
        skeleton_labels_list.append(skeleton_labels)

    skeleton_data = np.concatenate(skeleton_data_list, axis=0)
    skeleton_labels = np.concatenate(skeleton_labels_list, axis=0)
    return skeleton_data, skeleton_labels


def test_model(model, file_path, save_path="data_in_use/data_12343658_1_022_single_label.csv",
               device=torch.device("cpu")):
    df = pd.read_csv(file_path)

    x_columns = [col for col in df.columns if col.startswith("X")]
    y_columns = [col for col in df.columns if col.startswith("Y")]
    z_columns = [col for col in df.columns if col.startswith("Z")]

    data_x = df[x_columns].values
    data_y = df[y_columns].values
    data_z = df[z_columns].values

    skeleton_data = np.stack([data_x, data_y, data_z], axis=0)
    skeleton_labels = df["Cluster"].values.astype(int)

    input_tensor = torch.from_numpy(skeleton_data).float().unsqueeze(0)
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    df["Prediction"] = predicted_class
    label = np.unique(skeleton_labels)
    print("Prediction", predicted_class)
    print("Label", label)
    print()

    path_components = file_path.split("/")
    file = path_components[-1]
    file_name, ext = file.split(".")
    file_name += "_cnn"
    file = file_name + "." + ext
    path_components[-1] = file
    out_path = "/".join(path_components)
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

    return df


def load_model(model_class, checkpoint_path, device=torch.device("cpu")):
    model = model_class().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def train_cnn(dataset_path: str = "data_in_use", model_path: str = "best_cnn.pth", test: bool = True,
              train: bool = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_frames = []
    for file in os.listdir(dataset_path):
        if file.endswith("_single_label.csv"):
            file_path = dataset_path + "/" + file
            df = pd.read_csv(file_path)
            data_frames.append(df)

    skeleton_data, skeleton_labels = extract_features(data_frames)

    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        skeleton_data, skeleton_labels, test_size=0.3, random_state=42, stratify=skeleton_labels)

    torch_train_seqs = torch.from_numpy(train_seqs).float()
    torch_train_labels = torch.from_numpy(train_labels).long()
    torch_val_seqs = torch.from_numpy(val_seqs).float()
    torch_val_labels = torch.from_numpy(val_labels).long()

    torch_train_seqs = torch_train_seqs.unsqueeze(2)
    torch_val_seqs = torch_val_seqs.unsqueeze(2)

    train_dataset = SkeletonDataset(torch_train_seqs, torch_train_labels)
    val_dataset = SkeletonDataset(torch_val_seqs, torch_val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = None
    if train:
        model = CNN()
        model = train_model(model=model, train_loader=train_loader, num_epochs=30, val_loader=val_loader, device=device)
    if test:
        if model == None:
            model = load_model(CNN, model_path, device=device)
        for file in os.listdir(dataset_path):
            if file.endswith("_single_label.csv"):
                file_path = dataset_path + "/" + file
                test_model(model=model, file_path=file_path, device=device)


if __name__ == "__main__":
    test = True
    train = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s = timeit.default_timer()

    train_cnn(dataset_path="data_in_use", model_path="best_cnn.pth", test=True, train=True)

    e = timeit.default_timer()
    print("ELAPSED", e - s)
