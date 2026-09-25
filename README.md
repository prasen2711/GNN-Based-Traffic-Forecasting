GNN-Based Traffic Forecasting



A deep learning-based traffic forecasting project using Graph Neural Network (GNN) approaches for modeling spatial and temporal relationships in traffic sensor data.



The project works with the METR-LA traffic dataset and includes implementations for STGCN (Spatio-Temporal Graph Convolutional Network) and DSTAGNN (Dynamic Spatial-Temporal Attention Graph Neural Network) models. It also provides a web-based interface for interacting with the forecasting system.



Project Overview



Traffic forecasting is an important component of intelligent transportation systems. Traffic sensors are distributed across a road network, where measurements at one location can be influenced by traffic conditions at nearby locations.



Traditional time-series models primarily focus on temporal patterns. Graph Neural Networks can additionally model the spatial relationships between traffic sensors, making them suitable for traffic forecasting problems.



This project combines:



Graph-based spatial modeling



Temporal traffic patterns



Deep learning



Traffic sensor network information



Pre-trained forecasting models



A Flask-based web interface



Models

STGCN



STGCN (Spatio-Temporal Graph Convolutional Network) combines graph convolution with temporal convolution to model spatial and temporal dependencies in traffic data.



DSTAGNN



DSTAGNN (Dynamic Spatial-Temporal Attention Graph Neural Network) uses attention mechanisms to model dynamic spatial and temporal relationships in the traffic network.



The trained model files are stored in:



output/tuning\_results/

├── DSTAGNN\_run\_1.pth

├── DSTAGNN\_run\_1\_loss\_plot.png

├── STGCN\_run\_1.pth

└── STGCN\_run\_1\_loss\_plot.png



Dataset



This project uses the METR-LA traffic dataset, which contains traffic measurements collected from sensors in the Los Angeles highway network.



The dataset and supporting graph files are located in:



Dataset\_DP\_ESE/

├── SE\_metrla.txt

├── W\_metrla.csv

├── adj\_mx.pkl

├── distances\_la\_2012.csv

├── graph\_sensor\_locations.csv

├── metr-la.h5

└── metr\_ids.txt





The metr-la.h5 file contains the traffic measurements used by the forecasting pipeline.



Project Structure

GNN-Based-Traffic-Forecasting/

│

├── Dataset\_DP\_ESE/

│   ├── SE\_metrla.txt

│   ├── W\_metrla.csv

│   ├── adj\_mx.pkl

│   ├── distances\_la\_2012.csv

│   ├── graph\_sensor\_locations.csv

│   ├── metr-la.h5

│   └── metr\_ids.txt

│

├── output/

│   ├── scaler.pkl

│   └── tuning\_results/

│       ├── DSTAGNN\_run\_1.pth

│       ├── DSTAGNN\_run\_1\_loss\_plot.png

│       ├── STGCN\_run\_1.pth

│       └── STGCN\_run\_1\_loss\_plot.png

│

├── static/

│   └── css/

│       ├── home\_style.css

│       └── style.css

│

├── templates/

│   ├── home.html

│   ├── index.html

│   └── system\_arch.html

│

├── GeminiV3.ipynb

├── app.py

├── app\_cc.py

├── precompute.py

├── print\_models.py

├── train\_models.py

├── dl\_report.docx

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



Jupyter Notebook



HTML



CSS



Installation



Clone the repository:



git clone https://github.com/prasen2711/GNN-Based-Traffic-Forecasting.git

cd GNN-Based-Traffic-Forecasting





Create a virtual environment:



Windows

python -m venv venv

venv\\Scripts\\activate



Linux/macOS

python3 -m venv venv

source venv/bin/activate





Install the required Python packages.



If the project contains a requirements.txt file:



pip install -r requirements.txt





Otherwise, install the dependencies required by the Python source files and your PyTorch environment.



Running the Application



The project includes a Flask application.



Run:



python app.py





After starting the application, open the local address displayed by Flask in your browser.



For example:



http://127.0.0.1:5000/



Model Training



The project includes scripts related to preprocessing and model training:



precompute.py

train\_models.py

print\_models.py





The general workflow is:



Traffic Dataset

&#x20;     │

&#x20;     ▼

Data Preprocessing

&#x20;     │

&#x20;     ▼

Graph Construction

&#x20;     │

&#x20;     ▼

Spatio-Temporal Modeling

&#x20;     │

&#x20;     ├───────────────┐

&#x20;     ▼               ▼

&#x20;   STGCN          DSTAGNN

&#x20;     │               │

&#x20;     └───────┬───────┘

&#x20;             ▼

&#x20;      Traffic Forecast

&#x20;             │

&#x20;             ▼

&#x20;       Web Application



Web Interface



The project contains a Flask-based web interface with HTML templates located in:



templates/

├── home.html

├── index.html

└── system\_arch.html





The corresponding CSS files are located in:



static/css/





The interface is intended to provide access to the traffic forecasting functionality and system information.



Output Files



Model and preprocessing outputs are stored under:



output/





Currently included outputs include:



scaler.pkl



STGCN trained model



DSTAGNN trained model



Model loss plots



These files can be used by the application when performing inference, depending on the configuration implemented in the Python scripts.



Notebook



The repository also contains:



GeminiV3.ipynb





The notebook can be opened using Jupyter Notebook or JupyterLab:



jupyter notebook GeminiV3.ipynb





or:



jupyter lab GeminiV3.ipynb



Research / Project Objectives



The project focuses on exploring how graph-based deep learning can be applied to traffic forecasting by considering both:



Temporal dependencies — how traffic conditions change over time.



Spatial dependencies — how traffic conditions at connected sensors influence each other.



The objective is to build a forecasting pipeline that can learn these relationships from traffic sensor data.



Future Improvements



Possible future improvements include:



Adding more forecasting horizons



Hyperparameter optimization



Additional GNN architectures



Improved visualization of predictions



Real-time traffic data integration



Model performance comparison using MAE, RMSE and MAPE



Docker deployment



Cloud deployment



Interactive traffic-network visualization



REST API for model inference



Disclaimer



This repository is intended for academic, research, and educational purposes.



The forecasting results depend on the dataset, preprocessing pipeline, model configuration, and trained model parameters.



Author



Prasen



GitHub:



https://github.com/prasen2711



License



A license has not yet been specified for this repository.



If this project is intended for public reuse, consider adding an appropriate open-source license such as MIT before distributing the code.

