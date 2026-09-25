GNN-Based Traffic Forecasting

A deep learning project for traffic forecasting using Graph Neural Networks (GNNs) to learn spatial and temporal traffic patterns.

✨ Features

Traffic forecasting using Graph Neural Networks

STGCN and DSTAGNN models

METR-LA traffic dataset

Spatial and temporal dependency modeling

Pre-trained model checkpoints

Flask-based web application

Traffic sensor network visualization

Data preprocessing and model training scripts

🏗️ Architecture

Technology Stack

Python

PyTorch

Flask

NumPy

Pandas

Scikit-learn

NetworkX

PyDeck

Jupyter Notebook

Workflow

METR-LA Dataset
      ↓
Data Preprocessing
      ↓
Traffic Sensor Graph
      ↓
Spatio-Temporal GNN
      ↓
 ┌───────────┐
 │           │
STGCN     DSTAGNN
 │           │
 └─────┬─────┘
       ↓
Traffic Forecast
       ↓
Flask Web Application

🤖 Models
STGCN

Spatio-Temporal Graph Convolutional Network

Uses graph convolution and temporal convolution to learn spatial and temporal traffic dependencies.

DSTAGNN

Dynamic Spatial-Temporal Attention Graph Neural Network

Uses attention mechanisms to model dynamic spatial and temporal relationships between traffic sensors.

📊 Dataset

This project uses the METR-LA traffic dataset, containing traffic measurements from sensors across the Los Angeles highway network.

Dataset files:

Dataset_DP_ESE/
├── metr-la.h5
├── metr_ids.txt
├── adj_mx.pkl
├── W_metrla.csv
├── SE_metrla.txt
├── distances_la_2012.csv
└── graph_sensor_locations.csv

📂 Project Structure
GNN-Based-Traffic-Forecasting/
│
├── Dataset_DP_ESE/
├── output/
│   ├── scaler.pkl
│   └── tuning_results/
│
├── static/
├── templates/
│
├── app.py
├── app_cc.py
├── precompute.py
├── train_models.py
├── print_models.py
├── GeminiV3.ipynb
├── dl_report.docx
├── requirements.txt
└── README.md

🚀 Installation
Clone the repository
git clone https://github.com/prasen2711/GNN-Based-Traffic-Forecasting.git
cd GNN-Based-Traffic-Forecasting

Create a virtual environment

Windows

python -m venv venv
venv\Scripts\activate


Linux / macOS

python3 -m venv venv
source venv/bin/activate

Install dependencies
pip install -r requirements.txt

▶️ Run the Application
python app.py


Open the application at:

http://127.0.0.1:5000/

🧪 Training

Preprocess the dataset:

python precompute.py


Train the models:

python train_models.py


Inspect the models:

python print_models.py

📦 Pre-trained Models

The repository contains trained model files in:

output/tuning_results/


Including:

STGCN_run_1.pth

DSTAGNN_run_1.pth

Training loss plots

📓 Notebook

The project includes the Jupyter Notebook:

GeminiV3.ipynb


Run it with:

jupyter notebook GeminiV3.ipynb

🔮 Future Improvements

Real-time traffic data

Additional GNN models

Multiple forecasting horizons

Improved visualization

REST API

Docker deployment

Cloud deployment
