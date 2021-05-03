import numpy as np
import tensorflow as tf


class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df, batch_size,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        # creates the column names in the dataframe and index
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        print('\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}']))

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data, batch_size):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=batch_size, )

        ds = ds.map(self.split_window)

        return ds

    def map_data(self, data, train_val_test):
        if train_val_test == 'train':
            data = np.array(data.train_df, dtype=np.float32)
        elif train_val_test == 'val':
            data = np.array(data.val_df, dtype=np.float32)
        else:
            data = np.array(data.test_df, dtype=np.float32)

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=1, )
        ds = ds.map(self.split_window)
        return ds

    def make_dataset_combine(self, data, window_array, train_val_test, batch_size):
        data_grid = np.array(data, dtype=np.float32)

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data_grid,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=1, )
        ds = ds.map(self.split_window)

        # all six postcodes
        ds1 = self.map_data(window_array[0], train_val_test)
        ds2 = self.map_data(window_array[1], train_val_test)
        ds3 = self.map_data(window_array[2], train_val_test)
        ds4 = self.map_data(window_array[3], train_val_test)
        ds5 = self.map_data(window_array[4], train_val_test)
        ds6 = self.map_data(window_array[5], train_val_test)

        return ds, ds1, ds2, ds3, ds4, ds5, ds6

    @property
    def train(self):
        print("creating tf train dataset")
        return self.make_dataset(self.train_df, self.batch_size)

    @property
    def val(self):
        print("creating tf val dataset")
        return self.make_dataset(self.val_df, self.batch_size)

    @property
    def test(self):
        print("creating tf test dataset")
        return self.make_dataset(self.test_df, self.batch_size)

    def train_combine(self, window_array):
        print("creating tf train dataset")
        return self.make_dataset_combine(self.train_df, window_array, 'train', self.batch_size)

    def val_combine(self, window_array):
        print("creating tf val dataset")
        return self.make_dataset_combine(self.val_df, window_array, 'val', self.batch_size)

    def test_combine(self, window_array):
        print("creating tf test dataset")
        return self.make_dataset_combine(self.test_df, window_array, 'test', self.batch_size)
