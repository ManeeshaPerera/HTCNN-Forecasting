# Examples from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import tensorflow.keras as keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size=32, dim=(173, 192, 18), n_channels=8, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 18, 1), dtype=float)

        # Generate data
        # for i, ID in enumerate(list_IDs_temp):
        #     # Store sample
        #     X[i,] = np.load(f'swis_ts_data/img_ts/train_{ID}.npy')
        #
        #     # Store class
        #     y[i] = np.load(f'swis_ts_data/img_ts/train_{ID}_label.npy')

        for i in range(len(list_IDs_temp)):
            # Store sample
            X[i,] = np.load(f'swis_ts_data/img_ts/train_{i}.npy')

            # Store class
            y[i, ] = np.load(f'swis_ts_data/img_ts/train_{i}_label.npy')

        return X, y
