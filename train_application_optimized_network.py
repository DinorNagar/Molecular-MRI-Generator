from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import scipy.io
import time
import os
import re


def create_directory_if_not_exists(directory_path):
    """
    Checks if the specified directory exists, and creates it if it doesn't.

    Parameters:
    directory_path (str): The directory path to check and create.

    Returns:
    bool: True if the directory already exists or was successfully created, False otherwise.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory created: {directory_path}")
            return True
        except OSError as e:
            print(f"Failed to create directory: {e}")
            return False
    else:
        print(f"Directory already exists: {directory_path}")
        return True


def get_tp_Trec_values(string):
    numbers = re.findall(r'\d+\.\d+|\d+', string)
    B0 = numbers[-4]
    tp = numbers[-3]
    Trec = numbers[-2]
    n_pulses = numbers[-1]

    return float(B0), float(tp), float(Trec), float(n_pulses)


def from_dict_to_data_scheme(parameter_mat, signal_matrix, tp, Trec, B0, n_pulses, samples_size):
    tp_vec = np.repeat(tp, samples_size, axis=0).reshape((samples_size, 1))
    Trec_vec = np.repeat(Trec, samples_size , axis=0).reshape((samples_size, 1))
    B0_vec = np.repeat(B0, samples_size, axis=0).reshape((samples_size, 1))
    n_pulses_vec = np.repeat(n_pulses, samples_size, axis=0).reshape((samples_size, 1))
    input_data = np.hstack((parameter_mat, tp_vec, Trec_vec, B0_vec, n_pulses_vec))
    output_data = signal_matrix

    return input_data, output_data


def scale_data(input_data):
    num_of_parameters = 31
    scaler_x = np.array([1.6, 0.12, 1, 1, 0.1, 0.0045045, 90, 1, 0.1, 0.0014, 1000, 1, 0.1, 0.0045045, 5000, 1, 0.1,
                         0.0045045, 3500, 1.3, 0.005, 0.00675, 20, 1, 0.00004, 0.216, 30, 2.4, 4, 11.7, 30])  # 1

    input_data_scaled = np.array([input_data[:, i] / scaler_x[i] for i in range(num_of_parameters)]).T

    return np.round(input_data_scaled, 6)


def create_model_architecture():
    final_model = Sequential()
    final_model.add(Dense(256, input_dim=31, activation='sigmoid', kernel_initializer='he_uniform'))
    final_model.add(Dense(256, input_dim=31, activation='sigmoid', kernel_initializer='he_uniform'))
    final_model.add(Dense(34, activation='sigmoid'))

    return final_model


def train(input_gen, validation_gen, input_model, checkpoint_path, result_graph_path, epochs=100, epochs_patience=5):

    tf.keras.backend.clear_session()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    filepath = "weights-improvements-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, filepath),
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        period=1)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs_patience, restore_best_weights=True)
    input_model.compile(loss='mse', optimizer='adam')

    # fit the model on the training dataset
    hist = input_model.fit(input_gen, epochs=epochs, validation_data=validation_gen,
                           callbacks=[callback, checkpoint_callback])

    plt.figure(1)
    epochs = np.arange(len(hist.history['val_loss']))
    plt.plot(epochs, hist.history['loss'], label='train loss')
    plt.plot(epochs, hist.history['val_loss'], label='validation loss')
    plt.title('losses vs epochs')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.legend()
    plt.savefig(os.path.join(result_graph_path,'loss_graph'))


def create_data_per_protocol(n, n_sample, n_dicts, dicts_path, tr):

    vl=round(1-tr, 8)

    start_time = time.time()
    total_time = time.time()

    for count, index in enumerate(range(n_dicts)):
        if index != 0 and index % 10 == 0:
            print(f"{int(index)} Dictionaries has been imported")
            print(f"Running time of {int(index)} dicts are: {time.time() - start_time}")
            start_time = time.time()

        dict_name = os.listdir(dicts_path)[index]
        dict_path = os.path.join(dicts_path, dict_name)
        B0_i, tp_i, Trec_i, n_pulses_i = get_tp_Trec_values(dict_name)

        """Creating the parameter_matrix"""
        parameters_matrix = scipy.io.loadmat(dict_path)

        signal_vec = parameters_matrix['sig']
        del parameters_matrix['__header__']
        del parameters_matrix['__version__']
        del parameters_matrix['__globals__']
        del parameters_matrix['sig']

        array_parameters_list = list(parameters_matrix.values())
        parameter_mat = np.vstack(array_parameters_list).T
        parameter_mat = parameter_mat.astype('float32')

        '''Selecting random indices'''
        random_idx = np.random.choice(np.arange(n), n_sample, replace=False)

        train_idx = random_idx[:int(n_sample * tr)]
        valid_idx = random_idx[int(n_sample * tr):]

        X_samples_train = parameter_mat[train_idx]
        y_samples_train = signal_vec[train_idx]

        X_samples_valid = parameter_mat[valid_idx]
        y_samples_valid = signal_vec[valid_idx]

        X_train, y_train = from_dict_to_data_scheme(X_samples_train, y_samples_train, tp_i, Trec_i, B0_i,
                                                    n_pulses_i, int(n_sample*tr))

        X_valid, y_valid = from_dict_to_data_scheme(X_samples_valid, y_samples_valid, tp_i, Trec_i, B0_i,
                                                    n_pulses_i, int(n_sample * vl))

        number_train = int(n_sample * tr)
        number_valid = int(n_sample * vl)

        X_train_scaled = scale_data(X_train)
        X_valid_scaled = scale_data(X_valid)

        train_data[int(count) * number_train:int(count + 1) * number_train, :] = X_train_scaled
        train_label[int(count) * number_train:int(count + 1) * number_train, :] = y_train

        val_data[int(count) * number_valid:int(count + 1) * number_valid, :] = X_valid_scaled
        val_labels[int(count) * number_valid:int(count + 1) * number_valid, :] = y_valid

    print(f"Loading the data took: {time.time() - total_time}")


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def preprocess_dataset(parameter_combinations_number, samples, num_dicts, data_path, batch_size,tr):
    """Creating the dataset"""

    create_data_per_protocol(n=parameter_combinations_number, n_sample=samples, n_dicts=num_dicts,
                             dicts_path=data_path, tr=tr)

    train_gen = DataGenerator(train_data, train_label, batch_size)
    valid_gen = DataGenerator(val_data, val_labels, batch_size)

    return train_gen, valid_gen


def main():
    global train_data, train_label, val_data, val_labels

    # Change to your dirs

    #loss_graph_dir = 'D:\\Thesis\\cest-mrf-main-or-github\\nn3\\results\\graph\\loss.png'
    #checkpoint_dir = 'D:\\Thesis\\cest-mrf-main-or-github\\nn3\\results\\checkpoints'
    #data_dir = 'D:\\Thesis\\cest-mrf-main-or-github\\nn3\\dataset'

    loss_graph_dir = '.\\train_results-application-optimized\\graphs'
    checkpoint_dir = '.\\train_results-application-optimized\\checkpoints'
    data_dir = '.\\dataset-application-optimized'

    create_directory_if_not_exists(loss_graph_dir)
    create_directory_if_not_exists(checkpoint_dir)

    # Hyper Parameters - Change according to your case
    num_dicts = 5 # 21
    num_samples = 259200 # 259200
    train_size = 0.9
    num_parameters = 31
    signal_length = 34
    parameter_combinations_num = 259200

    batch_size = 64
    epochs = 3 # 7
    epochs_for_patience = 3

    train_data = np.zeros((int(num_dicts * num_samples * train_size), num_parameters),
                          dtype=np.float16)
    train_label = np.zeros((int(num_dicts * num_samples * train_size), signal_length),
                           dtype=np.float16)
    val_data = np.zeros((int(num_dicts * num_samples * round(1-train_size,8)), num_parameters),
                        dtype=np.float16)
    val_labels = np.zeros((int(num_dicts * num_samples * round(1-train_size, 8)), signal_length),
                          dtype=np.float16)

    """Creating the dataset"""
    train_set, validation_set = preprocess_dataset(parameter_combinations_num, num_samples,
                                                   num_dicts, data_dir, batch_size,train_size)

    """Creating the final data from all the protocols"""

    loaded_model = create_model_architecture()
    train(train_set, validation_set, loaded_model, checkpoint_dir,  loss_graph_dir,
          epochs=epochs, epochs_patience=epochs_for_patience)


if __name__ == '__main__':
    main()
