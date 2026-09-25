GNN-Based Traffic Forecasting

A deep learning-based traffic forecasting system using Graph Neural Networks (GNNs) to model spatial and temporal relationships in traffic sensor data.

<p align="center"> <img src="https://github.com/user-attachments/assets/069ae074-70bd-4f3c-8f67-8c74dcf985a3" alt="GNN-Based Traffic Forecasting" width="900"> </p>
✨ Features

Traffic Forecasting: Predict traffic conditions using spatio-temporal Graph Neural Networks.

STGCN: Spatio-Temporal Graph Convolutional Network implementation.

DSTAGNN: Dynamic Spatial-Temporal Attention Graph Neural Network implementation.

METR-LA Dataset: Traffic sensor data from the Los Angeles highway network.

Pre-trained Models: Includes trained STGCN and DSTAGNN checkpoints.

Graph Modeling: Uses traffic sensor locations, adjacency matrices, and graph relationships.

Web Interface: Flask-based interface for traffic forecasting and visualization.

Training Pipeline: Includes preprocessing, training, and model inspection scripts.

🏗 Architecture

Models: STGCN, DSTAGNN

Deep Learning: PyTorch

Backend: Flask

Data Processing: NumPy, Pandas, Scikit-learn

Graph Processing: NetworkX, PyDeck

Dataset: METR-LA

Experimentation: Jupyter Notebook

Pipeline
METR-LA Dataset
       │
       ▼
Data Preprocessing
       │
       ▼
Traffic Sensor Graph
       │
       ▼
Spatial-Temporal Modeling
       │
   ┌───┴────┐
   ▼        ▼
 STGCN   DSTAGNN
   │        │
   └───┬────┘
       ▼
Traffic Forecast
       │
       ▼
Flask Web Application

📂 Project Structure
GNN-Based-Traffic-Forecasting/
│
├── Dataset_DP_ESE/          # METR-LA dataset and graph files
├── output/                  # Preprocessing and trained model outputs
│   ├── scaler.pkl
│   └── tuning_results/
│
├── static/                  # CSS files
├── templates/               # Flask HTML templates
│
├── app.py                   # Flask application
├── app_cc.py                # Alternative Flask application
├── precompute.py            # Data preprocessing
├── train_models.py          # Model training
├── print_models.py          # Model inspection/evaluation
├── GeminiV3.ipynb           # Jupyter Notebook
├── dl_report.docx           # Project report
├── requirements.txt         # Python dependencies
└── README.md

📊 Dataset

This project uses the METR-LA traffic dataset, containing traffic measurements collected from sensors across the Los Angeles highway network.

Dataset and graph files are located in:

Dataset_DP_ESE/


Important files include:

metr-la.h5
metr_ids.txt
adj_mx.pkl
W_metrla.csv
SE_metrla.txt
distances_la_2012.csv
graph_sensor_locations.csv

🤖 Models
STGCN

Spatio-Temporal Graph Convolutional Network

Combines graph convolution and temporal convolution to learn spatial and temporal traffic dependencies.

DSTAGNN

Dynamic Spatial-Temporal Attention Graph Neural Network

Uses attention mechanisms to model dynamic spatial and temporal relationships between traffic sensors.

📦 Pre-trained Models

Trained model checkpoints are included under:

output/tuning_results/

DSTAGNN_run_1.pth
STGCN_run_1.pth


Training loss plots are also included:

DSTAGNN_run_1_loss_plot.png
STGCN_run_1_loss_plot.png

🛠 Prerequisites

Python 3

Git

pip

PyTorch

Python packages listed in requirements.txt

🚀 Installation
1. Clone the repository
git clone https://github.com/prasen2711/GNN-Based-Traffic-Forecasting.git
cd GNN-Based-Traffic-Forecasting

2. Create a virtual environment

Windows:

python -m venv venv
venv\Scripts\activate


Linux/macOS:

python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

▶️ Run the Application

Start the Flask application:

python app.py


Open the application in your browser:

http://127.0.0.1:5000/

🧪 Training
Preprocess the dataset
python precompute.py

Train the models
python train_models.py

Inspect the models
python print_models.py

📓 Notebook

The project includes:

GeminiV3.ipynb


Open it using Jupyter Notebook or JupyterLab:

jupyter notebook GeminiV3.ipynb

🔮 Future Improvements

Real-time traffic data integration

Additional GNN architectures

Multiple forecasting horizons

Improved prediction visualization

More extensive model evaluation

REST API for inference

Docker deployment

Cloud deployment

Interactive traffic-network visualization

