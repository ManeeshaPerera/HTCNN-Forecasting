# File to run and store DNN results
# run with different configs then process and store

import sys
import run_lstm
import process_dnn_output

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
        model_num = 1

        for cell_dim in cell_dims:
            for lstm_layer in num_layers:
                for lr in learning_rate:
                    print("current model running")
                    print(cell_dim, lstm_layer, lr)
                    run_lstm.run_dnn(file_index, model_name, lstm_layer, cell_dim, epochs, lr, main_dir, model_num)
                    filepath = f'{main_dir}/{model_name}_{model_num}'
                    process_dnn_output.run_process(filepath)
