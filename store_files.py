import os
import pickle5 as pickle


def save_files(model_name, filename, num_iter, history, forecast, actual, main_dir, model_num):
    directory_path = f'{main_dir}/{model_name}_{model_num}/'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    print("\nsaving files ==>")

    with open(f'{directory_path}/training_loss_{filename}_iteration_{num_iter}', 'wb') as file_loss:
        pickle.dump(history.history, file_loss)

    with open(f'{directory_path}/forecast_{filename}_iteration_{num_iter}', 'wb') as file_fc:
        pickle.dump(forecast, file_fc)

    with open(f'{directory_path}/actual_{filename}_iteration_{num_iter}', 'wb') as file_ac:
        pickle.dump(actual, file_ac)