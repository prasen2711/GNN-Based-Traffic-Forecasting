# <<< START: FINAL SIMPLIFIED app.py FOR AWS >>>
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydeck as pdk
import networkx as nx
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
import traceback

# --- Configuration & File Paths (Simplified for Local Structure on Server) ---
app = Flask(__name__)
# Paths are relative to the app.py file
TUNING_RESULTS_PATH = 'output/tuning_results'
SCALER_PATH = 'output/scaler.pkl'
X_TEST_PATH = 'output/X_test.npy'
Y_TEST_PATH = 'output/Y_test.npy'
H5_FILE_PATH = 'Dataset_DP_ESE/metr-la.h5'
ADJ_FILE_PATH = 'Dataset_DP_ESE/adj_mx.pkl'
LOCATIONS_FILE_PATH = 'Dataset_DP_ESE/graph_sensor_locations.csv'

# --- Data Loading and Helper Functions (OPTIMIZED) ---
def load_persistent_data():
    print("Loading all data for the web app from local project directory...")
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    with open(ADJ_FILE_PATH, 'rb') as f: _, _, adj_matrix = pickle.load(f, encoding='latin1')
    adj_norm = torch.tensor(normalize_adj(adj_matrix), device=DEVICE)
    df_traffic = pd.read_hdf(H5_FILE_PATH)
    df_locations = pd.read_csv(LOCATIONS_FILE_PATH)
    df_locations['sensor_ids'] = df_locations['sensor_id'].astype(str)
    
    print("Loading pre-computed test arrays (X_test.npy, Y_test.npy)...")
    X_test_np = np.load(X_TEST_PATH)
    y_test_np = np.load(Y_TEST_PATH)
    
    print("Data loaded and processed successfully.")
    return scaler, adj_norm, adj_matrix, df_traffic, df_locations, X_test_np, y_test_np

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return (d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt).astype(np.float32)

# --- Global Variables & Constants ---
HISTORY_LEN = 12
HORIZON = 12
DEVICE = torch.device("cpu")
MAPBOX_API_KEY = "pk.eyJ1IjoicGFydGh1YmhlIiwiYSI6ImNtZmp6bGlsNjEwMXMyanNhdjU0YmZtamsifQ.vD1YKgPOzpX_DMaqIbbpLQ"
VISUALIZATION_CACHE = {}

# --- Model Hyperparameters ---
MODEL_PARAMS = {
    "DSTAGNN_run_1": {'spatial_hidden': 32, 'temporal_hidden': 64, 'dropout': 0.3},
    "STGCN_run_1": {'out_channels': 32, 'spatial_channels': 8},
}

# --- Model Class Definitions ---
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return F.relu(self.dropout(output))

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

MODEL_CLASSES = {"DSTAGNN": DSTAGNNEnhanced, "STGCN": STGCN}

# --- Load all data into memory at startup ---
SCALER, ADJ_NORM, ADJ_MATRIX, DF_TRAFFIC, DF_LOCATIONS, X_TEST_NP, Y_TEST_NP = load_persistent_data()

NUM_NODES = ADJ_MATRIX.shape[0]
TRAIN_SIZE = int(len(DF_TRAFFIC) * 0.7)
VAL_SIZE = int(len(DF_TRAFFIC) * 0.1)

# --- Pydeck Visualization Helpers and Flask Routes ---
def get_viz_styling(df):
    def speed_to_color(speed):
        if speed > 55: return [161, 217, 155, 255]
        if speed > 35: return [254, 217, 118, 255]
        return [255, 0, 0, 255]
    df['color'] = df['predicted_speed'].apply(speed_to_color)
    df['elevation'] = 18 + (65 - df['predicted_speed']).clip(lower=0) * 1.8
    df['glow_inner'] = df['color'].apply(lambda c: [c[0], c[1], c[2], 240])
    df['glow_mid'] = df['color'].apply(lambda c: [c[0], c[1], c[2], 140])
    df['glow_outer'] = df['color'].apply(lambda c: [c[0], c[1], c[2], 50])
    df['glow_peak'] = df['color'].apply(lambda c: [min(255, int(c[0]*1.25+40)), min(255, int(c[1]*1.25+40)), min(255, int(c[2]*1.25+40)), 255])
    return df

def get_pydeck_layers(df):
    material = {"ambient": 0.55, "diffuse": 0.7, "shininess": 120, "specularColor": [255, 255, 255]}
    return [
        pdk.Layer("ScatterplotLayer", data=df, get_position='[longitude, latitude]', get_fill_color='glow_outer', get_radius=160, radius_scale=2.5, billboard=False, pickable=False),
        pdk.Layer("ScatterplotLayer", data=df, get_position='[longitude, latitude]', get_fill_color='glow_mid', get_radius=90, radius_scale=2.2, billboard=False, pickable=False),
        pdk.Layer("ColumnLayer", data=df, get_position='[longitude, latitude]', get_elevation='elevation', elevation_scale=5, radius=50, diskResolution=4, get_fill_color='color', material=material, pickable=True, auto_highlight=True),
        pdk.Layer("ScatterplotLayer", data=df, get_position='[longitude, latitude]', get_fill_color='glow_inner', get_radius=40, radius_scale=1.8, billboard=False, pickable=False),
        pdk.Layer("ScatterplotLayer", data=df, get_position='[longitude, latitude]', get_fill_color='glow_peak', get_radius=18, radius_scale=2.5, billboard=True, pickable=False),
    ]

def get_pydeck_config():
    lighting_effect = {"@@type": "LightingEffect", "shadowColor": [0,0,0,0.45], "ambientLight": {"@@type":"AmbientLight", "color":[255,255,255], "intensity":1.0}, "pointLight1": {"@@type":"PointLight", "color":[82,194,230], "intensity":0.85, "position":[-0.14,49.7,80000]},"pointLight2": {"@@type":"PointLight", "color":[255,255,255], "intensity":0.7, "position":[-3.8,54.1,8000]}}
    view_state = pdk.ViewState(longitude=-118.3, latitude=34.0, zoom=9.5, pitch=55, bearing=15)
    return view_state, lighting_effect

def run_prediction(model_name_full, input_data_np):
    model_type = model_name_full.split('_')[0]
    params = MODEL_PARAMS[model_name_full]
    ModelClass = MODEL_CLASSES[model_type]
    model = ModelClass(adj=ADJ_NORM, num_nodes=NUM_NODES, forecast_horizon=HORIZON, **params).to(DEVICE)
    model_path = os.path.join(TUNING_RESULTS_PATH, f"{model_name_full}.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    input_tensor = torch.tensor(input_data_np, dtype=torch.float32)
    with torch.no_grad():
        predictions_scaled = model(input_tensor.to(DEVICE)).cpu().numpy()
    num_samples, num_sensors, _ = predictions_scaled.shape
    predictions_unscaled = SCALER.inverse_transform(predictions_scaled.reshape(-1, num_sensors)).reshape(num_samples, num_sensors, -1)
    return predictions_unscaled

def generate_test_set_map(model_name):
    sample_to_visualize = X_TEST_NP[150:151]
    actuals_sample = Y_TEST_NP[150:151]
    predictions = run_prediction(model_name, sample_to_visualize)
    actuals = SCALER.inverse_transform(actuals_sample.reshape(-1, NUM_NODES)).reshape(actuals_sample.shape)
    viz_timestamp_idx, viz_horizon_idx = 150, 2
    viz_df = DF_LOCATIONS.copy()
    viz_df['predicted_speed'] = predictions[0, :, viz_horizon_idx]
    viz_df['actual_speed'] = actuals[0, :, viz_horizon_idx]
    viz_df['tooltip_text'] = viz_df.apply(lambda row: f"<b>Sensor ID:</b> {row['sensor_ids']}<br/><b>Predicted:</b> {row['predicted_speed']:.2f} mph<br/><b>Actual:</b> {row['actual_speed']:.2f} mph", axis=1)
    viz_df = get_viz_styling(viz_df)
    timestamp = DF_TRAFFIC.index[TRAIN_SIZE + VAL_SIZE + viz_timestamp_idx + HISTORY_LEN + viz_horizon_idx]
    text_layer = pdk.Layer("TextLayer", [{"position": [-118.476, 34.09], "text": f"{model_name} - {timestamp.date()}"}], get_position='position', get_text='text', get_size=20, get_color=[161, 217, 155, 255])
    view_state, lighting_effect = get_pydeck_config()
    layers = get_pydeck_layers(viz_df) + [text_layer]
    tooltip = {"html": "{tooltip_text}", "style": {"backgroundColor": "rgba(0,0,0,0.75)", "color": "white"}}
    r = pdk.Deck(layers=layers, initial_view_state=view_state, api_keys={'mapbox': MAPBOX_API_KEY}, map_provider="mapbox", map_style="mapbox://styles/mapbox/dark-v10", effects=[lighting_effect], tooltip=tooltip)
    return r.to_html(as_string=True)

def generate_gridlock_map(model_name):
    event_indices = DF_LOCATIONS.sort_values(by='latitude').index[50:65]
    fab_input = np.full((1, HISTORY_LEN, NUM_NODES), 65.0, dtype=np.float32)
    fab_input[0, :, event_indices] = 5.0
    fab_input_scaled = SCALER.transform(fab_input.reshape(-1, NUM_NODES)).reshape(1, HISTORY_LEN, NUM_NODES)
    prediction = run_prediction(model_name, fab_input_scaled[..., np.newaxis])
    viz_df = DF_LOCATIONS.copy()
    viz_df['predicted_speed'] = prediction.reshape(NUM_NODES, HORIZON)[:, 2]
    viz_df.loc[event_indices, 'predicted_speed'] = 5.0
    viz_df['tooltip_text'] = viz_df.apply(lambda row: f"<b>Sensor ID:</b> {row['sensor_ids']}<br/><b>Predicted Speed:</b> {row['predicted_speed']:.2f} mph", axis=1)
    viz_df = get_viz_styling(viz_df)
    view_state, lighting_effect = get_pydeck_config()
    text_layer = pdk.Layer("TextLayer", [{"position": [-118.75, 34.25], "text": f"{model_name}: Gridlock Scenario"}], get_position='position', get_text='text', get_size=20, get_color=[240, 240, 240, 255])
    r = pdk.Deck(layers=get_pydeck_layers(viz_df) + [text_layer], initial_view_state=view_state, api_keys={'mapbox': MAPBOX_API_KEY}, map_provider="mapbox", map_style="mapbox://styles/mapbox/dark-v10", effects=[lighting_effect], tooltip={"html": "{tooltip_text}"})
    return r.to_html(as_string=True)

def generate_rerouting_map(model_name):
    hollywood_bounds = {"lat_min": 34.085, "lat_max": 34.11, "lon_min": -118.35, "lon_max": -118.28}
    hollywood_df = DF_LOCATIONS[DF_LOCATIONS['latitude'].between(hollywood_bounds['lat_min'], hollywood_bounds['lat_max']) & DF_LOCATIONS['longitude'].between(hollywood_bounds['lon_min'], hollywood_bounds['lon_max'])]
    if hollywood_df.empty: return "<p style='color:red'>ERROR: No sensors found.</p>"
    hollywood_sensor_indices = hollywood_df.index.tolist()
    start_node, end_node = 80, 25
    fab_input = np.full((1, HISTORY_LEN, NUM_NODES), 65.0, dtype=np.float32)
    fab_input[0, :, hollywood_sensor_indices] = 5.0
    fab_input_scaled = SCALER.transform(fab_input.reshape(-1, NUM_NODES)).reshape(1, HISTORY_LEN, NUM_NODES)
    prediction_jam = run_prediction(model_name, fab_input_scaled[..., np.newaxis])
    predicted_speeds = prediction_jam.reshape(NUM_NODES, HORIZON)[:, 2]
    G = nx.from_numpy_array(ADJ_MATRIX)
    try:
        original_path = nx.shortest_path(G, source=start_node, target=end_node)
        G_weighted = G.copy()
        for u, v in G_weighted.edges():
            G_weighted.edges[u, v]['weight'] = 1.0 / (((predicted_speeds[u] + predicted_speeds[v]) / 2) + 1e-6)
        alternate_path = nx.shortest_path(G_weighted, source=start_node, target=end_node, weight='weight')
    except nx.NetworkXNoPath: return f"<p style='color:red'>ERROR: No path found.</p>"
    path_data = [{"name": "Original (Congested) Route", "path": DF_LOCATIONS.loc[original_path, ['longitude', 'latitude']].values.tolist(), "color": [255, 0, 0, 255]},
                 {"name": "Suggested Alternate Route", "path": DF_LOCATIONS.loc[alternate_path, ['longitude', 'latitude']].values.tolist(), "color": [0, 255, 255, 255]}]
    viz_df_jam = DF_LOCATIONS.copy()
    viz_df_jam['predicted_speed'] = predicted_speeds
    viz_df_jam.loc[hollywood_sensor_indices, 'predicted_speed'] = 5.0
    viz_df_jam = get_viz_styling(viz_df_jam)
    path_layer = pdk.Layer('PathLayer', data=pd.DataFrame(path_data), get_path='path', get_color='color', get_width=20, width_scale=5, rounded=True, pickable=True)
    view_state = pdk.ViewState(longitude=hollywood_df['longitude'].mean(), latitude=hollywood_df['latitude'].mean(), zoom=12, pitch=55, bearing=15)
    _, lighting_effect = get_pydeck_config()
    layers = get_pydeck_layers(viz_df_jam)
    text_layer = pdk.Layer("TextLayer", [{"position": [-118.35, 34.12], "text": f"{model_name}: Hollywood Rerouting"}], get_position='position', get_text='text', get_size=20, get_color=[240, 240, 240, 255])
    r = pdk.Deck(layers=layers + [path_layer, text_layer], initial_view_state=view_state, api_keys={'mapbox': MAPBOX_API_KEY}, map_provider="mapbox", map_style="mapbox://styles/mapbox/dark-v10", effects=[lighting_effect], tooltip={"html": "<b>{name}</b>"})
    return r.to_html(as_string=True)

def get_trained_models():
    if not os.path.exists(TUNING_RESULTS_PATH): return []
    return [f.replace('.pth', '') for f in os.listdir(TUNING_RESULTS_PATH) if f.endswith('.pth')]

@app.route('/')
def index():
    model_list = get_trained_models()
    return render_template('index.html', models=model_list)

@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.get_json()
    model_name = data.get('model_name')
    viz_type = data.get('viz_type')
    if not model_name or not viz_type: return jsonify({'error': 'Missing model_name or viz_type.'}), 400
    cache_key = f"{model_name}_{viz_type}"
    if cache_key in VISUALIZATION_CACHE: return jsonify({'map_html': VISUALIZATION_CACHE[cache_key]})
    try:
        if viz_type == 'test_set': map_html = generate_test_set_map(model_name)
        elif viz_type == 'gridlock': map_html = generate_gridlock_map(model_name)
        elif viz_type == 'rerouting': map_html = generate_rerouting_map(model_name)
        else: return jsonify({'error': 'Invalid visualization type.'}), 400
        VISUALIZATION_CACHE[cache_key] = map_html
        return jsonify({'map_html': map_html})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f"An internal error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Gunicorn is used for production
    pass
    
# <<< END: FINAL SIMPLIFIED app.py FOR AWS >>>