import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- Configuration & File Paths (from your app.py) ---
SCALER_PATH = 'output/scaler.pkl'
H5_FILE_PATH = 'Dataset_DP_ESE/metr-la.h5'
HISTORY_LEN = 12
HORIZON = 12

# --- Re-create the necessary functions ---
def create_dataset(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon])
    return np.array(X)[..., np.newaxis], np.transpose(np.array(y), (0, 2, 1))

print("Starting pre-computation...")

# Load the scaler and traffic data
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

df_traffic = pd.read_hdf(H5_FILE_PATH)
traffic_values = df_traffic.values

# Split data exactly as in app.py
train_size = int(len(traffic_values) * 0.7)
val_size = int(len(traffic_values) * 0.1)
test_data = traffic_values[train_size + val_size:]

# Scale the test data
test_data_scaled = scaler.transform(test_data)

# The heavy computation step
print("Creating test dataset arrays...")
X_test_np, Y_test_np = create_dataset(test_data_scaled, HISTORY_LEN, HORIZON)

# Save the results to .npy files
print("Saving arrays to .npy files...")
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.save(os.path.join(output_dir, 'X_test.npy'), X_test_np)
np.save(os.path.join(output_dir, 'Y_test.npy'), Y_test_np)

print("\nPre-computation complete!")
print(f"Created: {os.path.join(output_dir, 'X_test.npy')} (Shape: {X_test_np.shape})")
print(f"Created: {os.path.join(output_dir, 'Y_test.npy')} (Shape: {Y_test_np.shape})")
print("\nNext step: Upload these two .npy files to your 'output/' folder in the S3 bucket.")