import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time

# --- Configuration & Constants ---
# Ensure the output directory exists
os.makedirs('output/tuning_results', exist_ok=True)

# --- File Paths (Copied from app.py) ---
TUNING_RESULTS_PATH = 'output/tuning_results'
SCALER_PATH = 'output/scaler.pkl'
H5_FILE_PATH = 'Dataset_DP_ESE/metr-la.h5'
ADJ_FILE_PATH = 'Dataset_DP_ESE/adj_mx.pkl'

# --- Training Hyperparameters ---
LEARNING_RATE = 0.001
EPOCHS = 50 # You can adjust this value
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Model & Data Constants (Copied from app.py) ---
HISTORY_LEN = 12
HORIZON = 12
MODEL_PARAMS = {
    "DSTAGNN_run_1": {'spatial_hidden': 32, 'temporal_hidden': 64, 'dropout': 0.3},
    "STGCN_run_1": {'out_channels': 32, 'spatial_channels': 8},
}

# --- Model Class Definitions (Copied from app.py) ---
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return torch.relu(self.dropout(output))

class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvLayer, self).__init__()
        self.conv_a = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=((kernel_size - 1) // 2, 0))
        self.conv_b = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=((kernel_size - 1) // 2, 0))
    def forward(self, x):
        return self.conv_a(x) * torch.sigmoid(self.conv_b(x))

class STConvBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(STConvBlock, self).__init__()
        self.tcn1 = TemporalConvLayer(in_channels, out_channels)
        self.gcn = GraphConvolution(out_channels, spatial_channels)
        self.tcn2 = TemporalConvLayer(spatial_channels, out_channels)
        self.layer_norm = nn.LayerNorm([num_nodes, out_channels])
    def forward(self, x, adj):
        residual = x
        x = self.tcn1(x)
        x = self.gcn(x.permute(0, 2, 3, 1), adj).permute(0, 3, 1, 2)
        x = self.tcn2(x)
        return self.layer_norm((x + residual).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class DSTAGNNEnhanced(nn.Module):
    def __init__(self, adj, num_nodes, forecast_horizon, spatial_hidden, temporal_hidden, dropout, **kwargs):
        super(DSTAGNNEnhanced, self).__init__()
        self.register_buffer('adj', adj)
        self.gcn = GraphConvolution(1, spatial_hidden, dropout=dropout)
        self.lstm = nn.LSTM(spatial_hidden, temporal_hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(temporal_hidden, forecast_horizon)
        self.num_nodes, self.forecast_horizon = num_nodes, forecast_horizon
    def forward(self, x, adj_matrix=None):
        adj = adj_matrix if adj_matrix is not None else self.adj
        batch_size, lookback, _, _ = x.size()
        spatial_outputs = [self.gcn(x[:, t, :, :], adj) for t in range(lookback)]
        spatial_features = torch.stack(spatial_outputs, dim=1)
        lstm_input = spatial_features.permute(0, 2, 1, 3).reshape(batch_size * self.num_nodes, lookback, -1)
        lstm_out, _ = self.lstm(lstm_input)
        out = self.fc(lstm_out[:, -1, :])
        return out.reshape(batch_size, self.num_nodes, self.forecast_horizon)

class STGCN(nn.Module):
    def __init__(self, adj, num_nodes, forecast_horizon, out_channels, spatial_channels, **kwargs):
        super(STGCN, self).__init__()
        self.register_buffer('adj', adj)
        self.block1 = STConvBlock(1, spatial_channels, out_channels, num_nodes)
        self.block2 = STConvBlock(out_channels, spatial_channels, out_channels, num_nodes)
        self.final_conv = nn.Conv2d(out_channels, forecast_horizon, (HISTORY_LEN, 1))
    def forward(self, x, adj_matrix=None):
        adj = adj_matrix if adj_matrix is not None else self.adj
        x = self.block1(x.permute(0, 3, 1, 2), adj)
        x = self.block2(x, adj)
        x = self.final_conv(x).squeeze(2)
        return x.permute(0, 2, 1)

# --- Data Loading and Helper Functions (Copied and adapted from app.py) ---
def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return (d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt).astype(np.float32)

def create_dataset(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon])
    return np.array(X)[..., np.newaxis], np.transpose(np.array(y), (0, 2, 1))

def load_and_prepare_data():
    print("Loading and preparing data for training...")
    # Load raw data
    with open(ADJ_FILE_PATH, 'rb') as f: _, _, adj_matrix = pickle.load(f, encoding='latin1')
    df_traffic = pd.read_hdf(H5_FILE_PATH)
    traffic_values = df_traffic.values

    # Normalize adjacency matrix
    adj_norm = torch.tensor(normalize_adj(adj_matrix), device=DEVICE)

    # Split data
    train_size = int(len(traffic_values) * 0.7)
    val_size = int(len(traffic_values) * 0.1)
    
    train_data = traffic_values[:train_size]
    val_data = traffic_values[train_size:train_size + val_size]
    
    # Fit scaler ONLY on training data and transform all sets
    scaler = pd.read_pickle(SCALER_PATH)
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    
    # Create time-series datasets
    X_train, y_train = create_dataset(train_scaled, HISTORY_LEN, HORIZON)
    X_val, y_val = create_dataset(val_scaled, HISTORY_LEN, HORIZON)

    print(f"Dataset shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val: {y_val.shape}")

    # Create PyTorch DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, adj_norm, adj_matrix.shape[0]

# --- Plotting and Table Functions ---
def plot_and_save_history(history, model_name):
    """Plots the training and validation loss and saves it to a file."""
    plt.figure(figsize=(12, 7))
    plt.plot(history['train_loss'], label='Training Loss', color='blue', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', color='red', marker='s')
    plt.title(f'Training & Validation Loss for {model_name}', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(TUNING_RESULTS_PATH, f"{model_name}_loss_plot.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    plt.close()

def print_history_table(history, model_name):
    """Prints a formatted table of the training history."""
    print("\n" + "="*50)
    print(f"Training History Summary for {model_name}")
    print("="*50)
    
    # Create a DataFrame for nice formatting
    history_df = pd.DataFrame({
        'Epoch': range(1, len(history['train_loss']) + 1),
        'Train Loss (MAE)': history['train_loss'],
        'Validation Loss (MAE)': history['val_loss']
    })
    print(history_df.to_string(index=False))
    print("="*50)

# --- Core Training Function ---
def run_training_pipeline(model, model_name, train_loader, val_loader):
    """
    Manages the full training, validation, and saving process for a model.
    """
    print(f"\n--- Starting Training for {model_name} ---")
    
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.L1Loss() # MAE Loss

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                output = model(X_batch)
                loss = loss_fn(output, y_batch)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Duration: {epoch_duration:.2f}s")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(TUNING_RESULTS_PATH, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  -> Validation loss improved. Model saved to {model_path}")

    print(f"--- Finished Training for {model_name} ---")
    return history

# --- Main Execution Block ---
if __name__ == '__main__':
    # 1. Load and prepare data
    train_loader, val_loader, adj_norm, num_nodes = load_and_prepare_data()

    # 2. Define models
    models_to_train = {
        "DSTAGNN_run_1": DSTAGNNEnhanced(
            adj=adj_norm,
            num_nodes=num_nodes,
            forecast_horizon=HORIZON,
            **MODEL_PARAMS["DSTAGNN_run_1"]
        ),
        "STGCN_run_1": STGCN(
            adj=adj_norm,
            num_nodes=num_nodes,
            forecast_horizon=HORIZON,
            **MODEL_PARAMS["STGCN_run_1"]
        )
    }

    # 3. Loop through and train each model
    for name, model_instance in models_to_train.items():
        training_history = run_training_pipeline(model_instance, name, train_loader, val_loader)
        
        # 4. Plot history and print table for the completed model
        plot_and_save_history(training_history, name)
        print_history_table(training_history, name)
        
    print("\nAll models have been trained successfully.")