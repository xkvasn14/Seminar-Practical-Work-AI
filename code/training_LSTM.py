import multiprocessing
import timeit

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from LSTM import LSTM


# Define your custom Dataset
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, transform=None):
        """
        sequences: Tensor of shape (N, seq_len, input_dim)
        labels: Tensor of shape (N,)
        """
        self.sequences = sequences
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = self.sequences[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


# Training loop
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == batch_y).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = total_correct / len(train_loader.dataset)
    print(f"Train Loss: {avg_loss:.4f}  Acc: {accuracy:.4f}")
    return avg_loss, accuracy


# Validation loop
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()

    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = total_correct / len(val_loader.dataset)
    print(f"Val   Loss: {avg_loss:.4f}  Acc: {accuracy:.4f}")
    return avg_loss, accuracy


# Test loop
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = total_correct / len(test_loader.dataset)
    print(f"Test Loss: {avg_loss:.4f}  Acc: {accuracy:.4f}")
    return avg_loss, accuracy


def train_lstm(dataset_path: str = 'data_in_use/tmp_file.csv', model_path: str = 'best_lstm.pth'):
    # got an error on my pc because of not having this
    multiprocessing.freeze_support()

    # Load and parse CSV to create datasets
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    df = pd.read_csv(dataset_path)
    labels = df['Cluster'].values.astype(int)
    unique_labels = pd.unique(labels)
    num_classes = len(unique_labels)
    features = df.drop(columns=['Cluster']).values.astype(float)
    input_dim = features.shape[1]  # shape (6690, 45)
    sequences = features.reshape(-1, 1, input_dim)  # shape (6690, 1, 45)

    train_seqs, temp_seqs, train_labels, temp_labels = train_test_split(
        sequences, labels, test_size=0.3, random_state=42, stratify=labels
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
    train_dataset = SequenceDataset(torch_train_seqs, torch_train_labels)
    val_dataset = SequenceDataset(torch_val_seqs, torch_val_labels)
    test_dataset = SequenceDataset(torch_test_seqs, torch_test_labels)

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

    # Model, criterion, optimizer, scheduler
    # Assume `LSTM` class is already imported
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")



    model = LSTM(input_dim=input_dim, hidden_dim=128, fc_dim=32, output_dim=num_classes, dropout=0.3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Main training loop
    num_epochs = 30
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model at epoch {epoch}")

    # Run test after training
    test(model, test_loader, criterion, device)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    s = timeit.default_timer()
    train_lstm(dataset_path='data_in_use/tmp_file.csv', model_path='best_lstm.pth')
    e = timeit.default_timer()
    print("ELAPSED:", e - s)
