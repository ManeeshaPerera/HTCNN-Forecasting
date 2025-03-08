## Regional solar power forecasting with hierarchical temporal convolutional neural networks
This repository includes the code for the paper titled "Day-ahead regional solar power forecasting with hierarchical temporal convolutional neural networks using historical power generation and weather data".

```
@article{perera2024day,
  title={Day-ahead regional solar power forecasting with hierarchical temporal convolutional neural networks using historical power generation and weather data},
  author={Perera, Maneesha and De Hoog, Julian and Bandara, Kasun and Senanayake, Damith and Halgamuge, Saman},
  journal={Applied Energy},
  volume={361},
  pages={122971},
  year={2024},
  publisher={Elsevier}
}
```

Folder structure: _Please note that you should include your own data in the relevant folders that are read in the below .py files._
```
src - This folder contains all source code related to the neural network architectures
run_benchmarks.py - Includes benchmark models: seasonal naive and ARIMA
run_benchmark_nns.py: Includes the starting point to run all benchmark neural network models: LSTM, 1D CNN and TCN
run_global_approach_for_swis.py: Includes the starting point to run the Direct Forecast Strategy with approach HTCNN A1 and A2 (please refer to the paper for an detailed explanation)
run_global_approach_for_swis_clustering_approach.py - Includes the starting point to run the SubRegionAGG Forecast Strategy with approach HTCNN A1 and A2 (please refer to the paper for an detailed explanation)
```

