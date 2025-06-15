import multiprocessing

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from LSTM import LSTM


def load_model(path: str, input_dim: int, hidden_dim: int = 128, fc_dim: int = 32, output_dim: int = 11,
               dropout: float = 0.3) -> nn.Module:
    """ Instantiate an LSTM model, load weights from a .pth checkpoint, and return it. """
    model = LSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        fc_dim=fc_dim,
        output_dim=output_dim,
        dropout=dropout
    )
    # Load state dict
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    print(f"Loaded model from {path}")
    return model


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


def test_lstm(normalized_data_path: str = "data_in_use/data_12343658_1_022.csv", model_lstm_path: str = "best_lstm.pth",
              save_csv: bool = True):
    multiprocessing.freeze_support()

    # Load and parse CSV to create datasets
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = pd.read_csv(normalized_data_path)
    labels = df['Cluster'].values.astype(int)
    features = df.drop(columns=['Cluster']).values.astype(float)
    input_dim = features.shape[1]  # shape (6690, 45)
    sequences = features.reshape(-1, 1, input_dim)  # shape (6690, 1, 45)

    torch_seqs = torch.from_numpy(sequences).float()
    torch_labels = torch.from_numpy(labels).long()
    dataset = SequenceDataset(torch_seqs, torch_labels)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model = load_model(model_lstm_path, input_dim=input_dim, hidden_dim=128, fc_dim=32, output_dim=11, dropout=0.3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss, total_correct = 0, 0
    predictions = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()

            predictions.append(preds[0].item())  # shape: [1]

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / len(loader.dataset)
    print(f"Test   Loss: {avg_loss:.4f}  Acc: {accuracy:.4f}")

    if save_csv:
        df["Prediction"] = predictions
        some_path = normalized_data_path.split("/")
        file = some_path[-1]
        file_name, ext = file.split(".")
        file_name += "_lstm"
        file = file_name + "." + ext
        some_path[-1] = file
        out_path = "/".join(some_path)
        print(f"Saving into {out_path}")
        df.to_csv(out_path, index=False)


if __name__ == '__main__':
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_012.csv", model_lstm_path="best_lstm.pth", save_csv=True)
    test_lstm(normalized_data_path="data_in_use/data_12343658_1_022.csv", model_lstm_path="best_lstm.pth",save_csv=True)
