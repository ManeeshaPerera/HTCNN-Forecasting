# File to run and store DNN results
# run with different configs then process and store

import sys
import run_lstm
import run_cnn
import process_dnn_output
import constants

if __name__ == '__main__':
    # getting the model and time series to run
    file_index = int(sys.argv[1])
    model_name = sys.argv[2]
    main_dir = sys.argv[3]

    if model_name == "lstm":
        cell_dims = [32, 64]
        num_layers = []
        if file_index > 6:
            num_layers = [3, 5]
        else:
            num_layers = [2, 3]

        learning_rate = [0.001, 0.0001]
        epochs = 1000

        lookback = [1, 3, 7]

        for cell_dim in cell_dims:
            for lstm_layer in num_layers:
                for lr in learning_rate:
                    for lag in lookback:
                        print("current model running")
                        print(cell_dim, lstm_layer, lr, lag)
                        model_dir = f'{cell_dim}_{lstm_layer}_{lr}_{lag}'
                        run_lstm.run_dnn(file_index, model_name, lstm_layer, cell_dim, epochs, lr, lag, main_dir,
                                         model_dir)
                        filepath = f'{main_dir}/{model_name}_{model_dir}'
                        process_dnn_output.run_process(filepath, lag, constants.TS[file_index])

    if model_name == "cnn":
        num_layers = [4, 6, 8, 10]
        n_filters = [32, 64, 128]
        learning_rate = [0.001, 0.0001]
        epochs = 1000
        lookback = [1, 3, 7]
        kernel_size = 2

        for filter_val in n_filters:
            for cnn_layer in num_layers:
                for lr in learning_rate:
                    for lag in lookback:
                        print("current model running")
                        print(filter_val, cnn_layer, lr, lag)
                        model_dir = f'{filter_val}_{cnn_layer}_{lr}_{lag}'
                        dilation_rate = 2
                        dilation_rates = [dilation_rate ** i for i in range(cnn_layer)]
                        run_cnn.run_dnn(file_index, model_name, cnn_layer, filter_val, epochs, lr, lag, dilation_rates,
                                        kernel_size, main_dir,
                                        model_dir)
                        filepath = f'{main_dir}/{model_name}_{model_dir}'
                        process_dnn_output.run_process(filepath, lag, constants.TS[file_index])
