import torch
from torchsummary import summary
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Import necessary components from your Flask app
# This assumes 'print_models.py' is in the same directory as 'app.py'
from app import (
    DSTAGNNEnhanced,
    STGCN,
    MODEL_PARAMS,
    ADJ_NORM,
    NUM_NODES,
    HORIZON,
    HISTORY_LEN,
    X_TEST_NP,
    Y_TEST_NP,
    SCALER,
    TUNING_RESULTS_PATH,
    BATCH_SIZE
)

# Use CPU for model inspection and evaluation
DEVICE = torch.device("cpu")

def evaluate_model(model, model_name_full, X_test, y_test, scaler):
    """
    Calculates and prints evaluation metrics for a given model on the test set.

    Args:
        model (torch.nn.Module): The model instance to evaluate.
        model_name_full (str): The full name of the model (e.g., 'DSTAGNN_run_1').
        X_test (np.ndarray): The test input data.
        y_test (np.ndarray): The ground truth test data.
        scaler: The scaler object used for inverse transforming data.
    """
    print("\n" + "-"*20 + f" Evaluating {model_name_full} " + "-"*20)

    # Load the trained model state from the .pth file
    model_path = os.path.join(TUNING_RESULTS_PATH, f"{model_name_full}.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: Model weights not found at '{model_path}'. Skipping evaluation.")
        print("="*60)
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_predictions_scaled = []
    
    # Process the test set in batches to prevent memory errors
    num_samples = X_test.shape[0]
    for i in range(0, num_samples, BATCH_SIZE):
        X_batch = X_test[i:i + BATCH_SIZE]
        X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            preds_batch_scaled = model(X_batch_tensor).cpu().numpy()
            all_predictions_scaled.append(preds_batch_scaled)

    # Concatenate predictions from all batches into a single array
    predictions_scaled = np.concatenate(all_predictions_scaled, axis=0)

    # Inverse transform predictions and actuals to their original scale (e.g., mph)
    # Reshaping is crucial to match the scaler's expected input format (n_samples, n_features)
    # where n_features is NUM_NODES.
    predictions_for_scaler = predictions_scaled.transpose(0, 2, 1).reshape(-1, NUM_NODES)
    y_test_for_scaler = y_test.transpose(0, 2, 1).reshape(-1, NUM_NODES)

    predictions_unscaled_flat = scaler.inverse_transform(predictions_for_scaler)
    y_test_unscaled_flat = scaler.inverse_transform(y_test_for_scaler)
    
    # Reshape back to (num_samples, num_nodes, horizon) for metric calculation
    predictions_unscaled = predictions_unscaled_flat.reshape(y_test.shape[0], HORIZON, NUM_NODES).transpose(0, 2, 1)
    y_test_unscaled = y_test_unscaled_flat.reshape(y_test.shape[0], HORIZON, NUM_NODES).transpose(0, 2, 1)

    # --- Calculate and Print Metrics ---
    print("\nMetrics on Test Set (unscaled):")

    # Calculate metrics for each individual forecast horizon (e.g., 5, 10, 15 min ahead)
    for h in range(HORIZON):
        pred_h = predictions_unscaled[:, :, h]
        true_h = y_test_unscaled[:, :, h]
        
        # Filter out zero values in ground truth to avoid errors in MAPE calculation
        non_zero_mask = true_h > 1e-6
        
        mae = mean_absolute_error(true_h[non_zero_mask], pred_h[non_zero_mask])
        rmse = np.sqrt(mean_squared_error(true_h[non_zero_mask], pred_h[non_zero_mask]))
        mape = np.mean(np.abs((true_h[non_zero_mask] - pred_h[non_zero_mask]) / true_h[non_zero_mask])) * 100
        
        # Assuming each horizon step is 5 minutes as is standard for METR-LA
        print(f"\nHorizon: {h+1} ({(h+1)*5} min ahead)")
        print(f"  MAE:  {mae:.4f} mph")
        print(f"  RMSE: {rmse:.4f} mph")
        print(f"  MAPE: {mape:.2f} %")

    # Calculate overall metrics across all horizons
    non_zero_mask_overall = y_test_unscaled > 1e-6
    overall_mae = mean_absolute_error(y_test_unscaled[non_zero_mask_overall], predictions_unscaled[non_zero_mask_overall])
    overall_rmse = np.sqrt(mean_squared_error(y_test_unscaled[non_zero_mask_overall], predictions_unscaled[non_zero_mask_overall]))
    overall_mape = np.mean(np.abs((y_test_unscaled[non_zero_mask_overall] - predictions_unscaled[non_zero_mask_overall]) / y_test_unscaled[non_zero_mask_overall])) * 100
    
    print("\n" + "-"*15 + " Overall Metrics (All Horizons) " + "-"*15)
    print(f"  Overall MAE:  {overall_mae:.4f} mph")
    print(f"  Overall RMSE: {overall_rmse:.4f} mph")
    print(f"  Overall MAPE: {overall_mape:.2f} %")
    print("="*60)


# --- 1. Inspect and Evaluate DSTAGNN (DSTAGNNEnhanced) Model ---

print("="*60)
print("          DSTAGNN (DSTAGNNEnhanced) Architecture")
print("="*60)

dstagnn_model_name = 'DSTAGNN_run_1'
dstagnn_params = MODEL_PARAMS[dstagnn_model_name]
dstagnn_model = DSTAGNNEnhanced(
    adj=ADJ_NORM,
    num_nodes=NUM_NODES,
    forecast_horizon=HORIZON,
    **dstagnn_params
).to(DEVICE)

# A simple print() is more illustrative for this model due to its internal loop
print(dstagnn_model)
print("\nNote: The DSTAGNN model uses a Python loop in its forward pass,")
print(f"making a linear summary difficult. It applies the GCN layer {HISTORY_LEN} times,")
print("stacks the outputs, and then feeds the sequence to the LSTM.")

# Evaluate the model on the test set
evaluate_model(dstagnn_model, dstagnn_model_name, X_TEST_NP, Y_TEST_NP, SCALER)


# --- 2. Inspect and Evaluate STGCN Model ---

print("\n" + "="*60)
print("                  STGCN Architecture")
print("="*60)

stgcn_model_name = 'STGCN_run_1'
stgcn_params = MODEL_PARAMS[stgcn_model_name]
stgcn_model = STGCN(
    adj=ADJ_NORM,
    num_nodes=NUM_NODES,
    forecast_horizon=HORIZON,
    **stgcn_params
).to(DEVICE)

# Define the input shape: (lookback_history, num_sensors, num_features)
stgcn_input_shape = (HISTORY_LEN, NUM_NODES, 1)

# Generate and print the summary using torchsummary
summary(stgcn_model, input_size=stgcn_input_shape, device=DEVICE.type)

# Evaluate the model on the test set
evaluate_model(stgcn_model, stgcn_model_name, X_TEST_NP, Y_TEST_NP, SCALER)