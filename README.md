GNN-Based Traffic Forecasting

A deep learning-based traffic forecasting system using Graph Neural Networks (GNNs) to model spatial and temporal relationships in traffic sensor data.

The project uses the METR-LA traffic dataset and implements two spatio-temporal deep learning architectures:

STGCN — Spatio-Temporal Graph Convolutional Network

DSTAGNN — Dynamic Spatial-Temporal Attention Graph Neural Network

The project also includes a Flask-based web application for interacting with the traffic forecasting system.

Overview

Traffic forecasting is an important component of intelligent transportation systems. Traffic sensors are distributed across a road network, and traffic conditions at one location can be influenced by conditions at nearby locations.

Traditional time-series forecasting methods primarily focus on temporal patterns. Graph Neural Networks can additionally model the spatial relationships between traffic sensors, making them suitable for traffic forecasting over road networks.

This project combines:

Graph-based spatial modeling

Temporal traffic patterns

Deep learning

Traffic sensor network information

Pre-trained forecasting models

STGCN and DSTAGNN architectures

A Flask-based web interface

Traffic sensor location information

Model preprocessing and training pipelines

Key Features

Traffic forecasting using Graph Neural Networks

Spatial dependency modeling between traffic sensors

Temporal dependency modeling

STGCN implementation

DSTAGNN implementation

METR-LA traffic dataset

Pre-trained model checkpoints

Traffic sensor graph data

Flask-based web application

Interactive traffic forecasting interface

Training and preprocessing scripts

Jupyter Notebook for experimentation

Model training loss visualization

System Architecture

The overall traffic forecasting pipeline follows the workflow below:

                 METR-LA Traffic Dataset
                           │
                           ▼
                  Data Preprocessing
                           │
                           ▼
                   Graph Construction
                           │
                           ▼
              Spatial-Temporal Features
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
                STGCN            DSTAGNN
                  │                 │
                  └────────┬────────┘
                           │
                           ▼
                  Traffic Forecast
                           │
                           ▼
                  Flask Web Application
                           │
                           ▼
                Forecast Visualization

Models
STGCN

STGCN (Spatio-Temporal Graph Convolutional Network) combines graph convolution and temporal convolution to learn spatial and temporal dependencies in traffic sensor data.

The graph component captures relationships between traffic sensors, while temporal convolution captures changes in traffic conditions over time.

DSTAGNN

DSTAGNN (Dynamic Spatial-Temporal Attention Graph Neural Network) uses attention mechanisms to model dynamic spatial and temporal relationships within the traffic sensor network.

The model is designed to capture changing dependencies between different traffic sensors over time.

Model Comparison
Model	Full Name	Main Approach
STGCN	Spatio-Temporal Graph Convolutional Network	Graph convolution + temporal convolution
DSTAGNN	Dynamic Spatial-Temporal Attention Graph Neural Network	Dynamic spatial-temporal attention
Dataset

This project uses the METR-LA traffic dataset, which contains traffic measurements collected from sensors in the Los Angeles highway network.

The dataset and supporting graph files are located in:

Dataset_DP_ESE/

Dataset Files
File	Description
metr-la.h5	Traffic measurements
metr_ids.txt	Traffic sensor identifiers
adj_mx.pkl	Sensor adjacency matrix
W_metrla.csv	Graph/weight information
SE_metrla.txt	Sensor-related graph data
distances_la_2012.csv	Distance information between sensors
graph_sensor_locations.csv	Geographic locations of traffic sensors

The metr-la.h5 file contains the traffic measurements used by the forecasting pipeline.

Project Structure
GNN-Based-Traffic-Forecasting/
│
├── Dataset_DP_ESE/
│   ├── SE_metrla.txt
│   ├── W_metrla.csv
│   ├── adj_mx.pkl
│   ├── distances_la_2012.csv
│   ├── graph_sensor_locations.csv
│   ├── metr-la.h5
│   └── metr_ids.txt
│
├── output/
│   ├── scaler.pkl
│   └── tuning_results/
│       ├── DSTAGNN_run_1.pth
│       ├── DSTAGNN_run_1_loss_plot.png
│       ├── STGCN_run_1.pth
│       └── STGCN_run_1_loss_plot.png
│
├── static/
│   └── css/
│       ├── home_style.css
│       └── style.css
│
├── templates/
│   ├── home.html
│   ├── index.html
│   └── system_arch.html
│
├── GeminiV3.ipynb
├── app.py
├── app_cc.py
├── precompute.py
├── print_models.py
├── train_models.py
├── dl_report.docx
├── requirements.txt
├── .gitignore
└── README.md

Technologies Used

Python

PyTorch

Graph Neural Networks

STGCN

DSTAGNN

Flask

NumPy

Pandas

Scikit-learn

Matplotlib

NetworkX

PyDeck

Jupyter Notebook

HTML

CSS

Requirements

The project requires Python and the Python packages listed in requirements.txt.

Main dependencies include:

PyTorch

NumPy

Pandas

Scikit-learn

Flask

Matplotlib

NetworkX

PyDeck

PyTables

The exact dependencies used by the project are specified in:

requirements.txt

Installation
1. Clone the repository
git clone https://github.com/prasen2711/GNN-Based-Traffic-Forecasting.git
cd GNN-Based-Traffic-Forecasting

2. Create a virtual environment
Windows
python -m venv venv
venv\Scripts\activate

Linux/macOS
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt


If PyTorch needs to be installed separately for your hardware or CUDA configuration, install the appropriate PyTorch build before running the project.

Running the Application

The project includes a Flask-based web application.

Run:

python app.py


After the application starts, Flask will display the local address in the terminal.

Typically, the application can be accessed at:

http://127.0.0.1:5000/


Open the displayed address in a web browser.

Web Application

The Flask application provides a web-based interface for interacting with the traffic forecasting system.

The main HTML templates are located in:

templates/
├── home.html
├── index.html
└── system_arch.html


The corresponding CSS files are located in:

static/css/
├── home_style.css
└── style.css


The web application is designed to provide access to the traffic forecasting functionality and system information.

Application Components

The application includes functionality related to:

Traffic forecasting

Traffic sensor information

Graph-based traffic network visualization

Model inference

Forecast results

System architecture information

Screenshots

Screenshots of the web application can be added here.

Create a folder:

screenshots/


and place application screenshots inside it.

For example:

screenshots/
├── home.png
├── forecasting.png
└── architecture.png


Then add them to this section:

Home Page
<p align="center">
  <img src="screenshots/home.png" alt="Traffic Forecasting Home Page" width="900">
</p>

Forecasting Interface
<p align="center">
  <img src="screenshots/forecasting.png" alt="Traffic Forecasting Interface" width="900">
</p>

System Architecture
<p align="center">
  <img src="screenshots/architecture.png" alt="System Architecture" width="900">
</p>

Pre-trained Models

The repository contains trained model checkpoints under:

output/tuning_results/


Current model files include:

DSTAGNN_run_1.pth
STGCN_run_1.pth


Training loss plots are also included:

DSTAGNN_run_1_loss_plot.png
STGCN_run_1_loss_plot.png


These checkpoints can be used by the application for inference according to the configuration implemented in the project.

Results

The project includes training loss plots for both implemented models.

STGCN Training Loss

The STGCN training loss plot is available at:

output/tuning_results/STGCN_run_1_loss_plot.png

DSTAGNN Training Loss

The DSTAGNN training loss plot is available at:

output/tuning_results/DSTAGNN_run_1_loss_plot.png

Evaluation Metrics

Evaluation metrics can be added below after running the model evaluation:

Model	MAE	RMSE	MAPE
STGCN	To be added	To be added	To be added
DSTAGNN	To be added	To be added	To be added

The evaluation values depend on the dataset split, preprocessing configuration, forecasting horizon, and trained model parameters.

Preprocessing

The project includes a preprocessing script:

precompute.py


The preprocessing pipeline prepares the traffic data and associated graph information for model training and inference.

Run:

python precompute.py


The preprocessing process may generate or update files under:

output/

Model Training

The project includes a training script:

train_models.py


To start model training:

python train_models.py


The training process uses the traffic dataset and graph information to train the forecasting models.

Depending on the configuration, trained model checkpoints and training plots are stored under:

output/tuning_results/

Model Inspection

The project also includes:

print_models.py


This script can be used to inspect the implemented models and calculate or display evaluation-related information according to the code configuration.

Run:

python print_models.py

Notebook

The repository contains the Jupyter Notebook:

GeminiV3.ipynb


The notebook can be opened using Jupyter Notebook or JupyterLab.

Jupyter Notebook
jupyter notebook GeminiV3.ipynb

JupyterLab
jupyter lab GeminiV3.ipynb


The notebook can be used for experimentation, analysis, visualization, and model development.

Forecasting Workflow

The complete workflow can be summarized as:

Traffic Sensor Data
        │
        ▼
Data Loading
        │
        ▼
Data Preprocessing
        │
        ▼
Normalization / Scaling
        │
        ▼
Traffic Sensor Graph
        │
        ▼
Spatial-Temporal Modeling
        │
        ├───────────────┐
        ▼               ▼
      STGCN          DSTAGNN
        │               │
        └───────┬───────┘
                ▼
        Traffic Prediction
                │
                ▼
       Forecast Evaluation
                │
                ▼
        Flask Web Interface

Research Objectives

The main objective of this project is to explore how graph-based deep learning can be applied to traffic forecasting by considering both spatial and temporal dependencies.

Temporal Dependencies

Traffic conditions change over time. Historical traffic measurements can therefore provide useful information for predicting future traffic conditions.

Spatial Dependencies

Traffic sensors are connected through the road network. Traffic conditions at one sensor can be related to conditions at neighboring or connected sensors.

Combined Modeling

Graph Neural Networks provide a way to model these spatial relationships while deep learning architectures can capture temporal patterns.

This project explores the combination of these two types of dependencies through STGCN and DSTAGNN architectures.

Future Improvements

Possible future improvements include:

Adding multiple forecasting horizons

Hyperparameter optimization

Additional GNN architectures

Improved prediction visualization

Real-time traffic data integration

More extensive model evaluation

MAE, RMSE, and MAPE comparison

Interactive traffic-network visualization

REST API for model inference

Docker deployment

Cloud deployment

Model monitoring

Real-time dashboard integration

Disclaimer

This repository is intended for academic, research, and educational purposes.

The forecasting results depend on the dataset, preprocessing pipeline, model architecture, model configuration, and trained model parameters.

The included trained models represent the state of the project at the time they were generated.

Author

Prasen

GitHub:

https://github.com/prasen2711

License

No open-source license has currently been specified for this repository.

If the project is intended for public reuse or distribution, an appropriate open-source license can be added by creating a LICENSE file in the repository.

Acknowledgements

This project uses the METR-LA traffic dataset for traffic forecasting research and experimentation.

The project builds on the broader research area of spatio-temporal Graph Neural Networks for intelligent transportation systems.

Repository

GitHub Repository:

https://github.com/prasen2711/GNN-Based-Traffic-Forecasting
