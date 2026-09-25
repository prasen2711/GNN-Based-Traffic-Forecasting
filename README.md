GNN-Based Traffic Forecasting

A deep learning-based traffic forecasting project using Graph Neural Networks (GNNs) to model spatial and temporal traffic patterns.

<p align="center"> <img src="https://github.com/user-attachments/assets/069ae074-70bd-4f3c-8f67-8c74dcf985a3" alt="GNN-Based Traffic Forecasting" width="900"> </p>
Overview

This project uses the METR-LA traffic dataset to predict traffic conditions using two spatio-temporal GNN models:

STGCN — Spatio-Temporal Graph Convolutional Network

DSTAGNN — Dynamic Spatial-Temporal Attention Graph Neural Network

A Flask web application is included to provide an interface for traffic forecasting and visualization.

Features

Traffic forecasting using GNNs

Spatial and temporal dependency modeling

STGCN and DSTAGNN models

METR-LA traffic dataset

Pre-trained model checkpoints

Flask web interface

Traffic network visualization

Training and preprocessing scripts

Project Structure
GNN-Based-Traffic-Forecasting/
│
├── Dataset_DP_ESE/       # METR-LA dataset and graph files
├── output/               # Scaler and trained models
├── static/               # CSS files
├── templates/            # Flask HTML templates
│
├── app.py                # Flask application
├── app_cc.py             # Alternative Flask application
├── precompute.py         # Data preprocessing
├── train_models.py       # Model training
├── print_models.py       # Model inspection/evaluation
├── GeminiV3.ipynb        # Jupyter Notebook
├── requirements.txt      # Python dependencies
└── README.md

Models
Model	Description
STGCN	Learns spatial and temporal traffic patterns using graph and temporal convolutions
DSTAGNN	Uses spatial-temporal attention to model dynamic traffic relationships
Dataset

The project uses the METR-LA dataset, containing traffic measurements from sensors on the Los Angeles highway network.

Dataset files are stored in:

Dataset_DP_ESE/

Installation
Clone the repository
git clone https://github.com/prasen2711/GNN-Based-Traffic-Forecasting.git
cd GNN-Based-Traffic-Forecasting

Create virtual environment

Windows:

python -m venv venv
venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Run the Application

Start the Flask application:

python app.py


Then open:

http://127.0.0.1:5000/

Training

Preprocess the data:

python precompute.py


Train the models:

python train_models.py


Model files are stored in:

output/tuning_results/

Technologies

Python

PyTorch

Graph Neural Networks

STGCN

DSTAGNN

Flask

Pandas

NumPy

Scikit-learn

NetworkX

PyDeck

Jupyter Notebook



License

No open-source license has currently been specified.
