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


def get_tp_Trec_values(dictionary_name):
    numbers_dict = re.findall('\d+', dictionary_name)
    tp, Trec = numbers_dict[:2]
    angle = re.findall('\d+', dictionary_name)[-1]
    if len(numbers_dict) == 6:
        B0 = re.findall('\d+', dictionary_name)[3] + '.' + re.findall('\d+', dictionary_name)[4]
    else:
        B0 = re.findall('\d+', dictionary_name)[3]
    return int(tp), int(Trec), float(B0), int(angle)


def read_parameters_from_txt(file_path):
    with open(file_path) as f:
        line = f.readlines()[0].split(',')
    B1 = np.array([np.round(float(x), 2) for x in line])
    return B1


def from_dict_to_data_scheme(parameter_mat, signal_matrix, tp, Trec, B1, B0, angle, samples_size, dw, offset_ppm):
    repeat_parameter_mat = np.repeat(parameter_mat, B1.shape[1], axis=0)
    signal_vec_rolled = np.roll(signal_matrix, 1, axis=1)
    signal_vec_rolled[:, 0] = 1
    signal_vec_squeezed = np.reshape(signal_vec_rolled, (samples_size * B1.shape[1], 1))

    """Constant vectors"""
    B1_vec = np.repeat(B1, samples_size, axis=0).reshape((samples_size * B1.shape[1], 1))
    offset_ppm_vec = np.repeat(offset_ppm, samples_size, axis=0).reshape((samples_size * B1.shape[1], 1))

    tp_vec = np.repeat(tp, samples_size * B1.shape[1], axis=0).reshape((samples_size * B1.shape[1], 1))
    Trec_vec = np.repeat(Trec, samples_size * B1.shape[1], axis=0).reshape((samples_size * B1.shape[1], 1))
    B0_vec = np.repeat(B0, samples_size * B1.shape[1], axis=0).reshape((samples_size * B1.shape[1], 1))
    angle_vec = np.repeat(angle, samples_size * B1.shape[1], axis=0).reshape((samples_size * B1.shape[1], 1))
    dw_vec = np.repeat(dw, samples_size * B1.shape[1], axis=0).reshape((samples_size * B1.shape[1], 1))

    input_data = np.hstack((repeat_parameter_mat, B1_vec, tp_vec, Trec_vec,
                            B0_vec, angle_vec, dw_vec, offset_ppm_vec, signal_vec_squeezed))
    output_data = np.reshape(signal_matrix, (samples_size * B1.shape[1], 1))

    return input_data, output_data


def scale_data(input_data):
    num_of_parameters = 13
    min_val = np.array([1.3, 0.04, 1.3, 0.00004, 20 / 111000, 5, 0.25, 1, 1, 3, 60, -3.5, -3.5])
    max_val = np.array([3.4, 1.2, 3.4, 0.04, 30000 / 110000, 1500, 6, 10, 10, 11.7, 90, 4.3, 20])

    input_data_scaled = np.array([(input_data[:, i] - min_val[i]) / (max_val[i] - min_val[i])
                                  for i in range(num_of_parameters)]).T

    input_data_scaled = np.hstack((input_data_scaled, np.expand_dims(input_data[:, num_of_parameters], axis=1)))

    return input_data_scaled


def create_model_architecture():
    final_model = Sequential()
    final_model.add(Dense(256, input_dim=14, activation='sigmoid', kernel_initializer='he_uniform'))
    final_model.add(Dense(256, input_dim=14, activation='sigmoid', kernel_initializer='he_uniform'))
    final_model.add(Dense(1, activation='sigmoid'))

    return final_model


def create_data_per_protocol(n, n_sample, len_B1, n_dicts, dicts_path, pos, dw_i, tr):
    vl = round(1 - tr, 10)

    start_time = time.time()
    total_time = time.time()
    for count, index in enumerate(range(0, 2 * n_dicts, 2)):  # len(os.listdir(dicts_path))
        if index != 0 and index % 1000 == 0:
            print(f"{int(index / 2)} Dictionaries has been imported")
            print(f"Running time of {int(index / 2)} dicts are: {time.time() - start_time}")
            start_time = time.time()
        dict_name = os.listdir(dicts_path)[index]

        B1_path = os.path.join(dicts_path, os.listdir(dicts_path)[index + 1])
        B1_i = read_parameters_from_txt(B1_path).reshape((1, len_B1))

        dict_path = os.path.join(dicts_path, dict_name)
        tp_i, Trec_i, B0_i, angle_i = get_tp_Trec_values(dict_name)
        offset_ppm_i = np.ones((1, len_B1)) * dw_i

        parameters_matrix = scipy.io.loadmat(dict_path)

        t1w_i = parameters_matrix['dict']['t1w'][0][0]
        t2w_i = parameters_matrix['dict']['t2w'][0][0]
        t1s_i = parameters_matrix['dict']['t1s'][0][0]
        t2s_i = parameters_matrix['dict']['t2s'][0][0]
        fs_i = parameters_matrix['dict']['fs'][0][0]
        ksw_i = parameters_matrix['dict']['ksw'][0][0]

        parameter_mat = np.hstack((t1w_i, t2w_i, t1s_i, t2s_i, fs_i, ksw_i))

        '''Signal'''
        signal_vec = parameters_matrix['dict']['sig'][0][0].T

        # Selecting random indices
        random_idx = np.random.choice(np.arange(n), n_sample, replace=False)

        train_idx = random_idx[:int(n_sample * tr)]
        valid_idx = random_idx[int(n_sample * tr):]

        X_samples_train = parameter_mat[train_idx]
        y_samples_train = signal_vec[train_idx]

        X_samples_valid = parameter_mat[valid_idx]
        y_samples_valid = signal_vec[valid_idx]

        X_train, y_train = from_dict_to_data_scheme(X_samples_train, y_samples_train, tp_i,
                                                    Trec_i, B1_i, B0_i, angle_i, int(n_sample * tr), dw_i, offset_ppm_i)

        X_valid, y_valid = from_dict_to_data_scheme(X_samples_valid, y_samples_valid, tp_i,
                                                    Trec_i, B1_i, B0_i, angle_i, int(n_sample * vl), dw_i, offset_ppm_i)

        number_train = int(len_B1 * n_sample * tr)
        number_valid = int(len_B1 * n_sample * vl)

        X_train_scaled = scale_data(X_train)
        X_valid_scaled = scale_data(X_valid)

        train_data[int(count + pos) * number_train:int(count + pos + 1) * number_train, :] = X_train_scaled
        train_label[int(count + pos) * number_train:int(count + pos + 1) * number_train, :] = y_train

        val_data[int(count + pos) * number_valid:int(count + pos + 1) * number_valid, :] = X_valid_scaled
        val_labels[int(count + pos) * number_valid:int(count + pos + 1) * number_valid, :] = y_valid

    print(f"Loading the data took: {time.time() - total_time}")
    pos += count + 1
    return pos


def create_data_per_protocol_changing_ppm(n, n_sample, len_B1, n_dicts, dicts_path, pos, dw_i, tr):
    vl = round(1 - tr, 10)

    start_time = time.time()
    total_time = time.time()
    for count, index in enumerate(range(0, 3 * n_dicts, 3)):  # len(os.listdir(dicts_path))
        if index != 0 and index % 1500 == 0:
            print(f"{int(index / 3)} Dictionaries has been imported")
            print(f"Running time of {int(index / 3)} dicts are: {time.time() - start_time}")
            start_time = time.time()
        dict_name = os.listdir(dicts_path)[index]

        B1_path = os.path.join(dicts_path, os.listdir(dicts_path)[index + 1])
        B1_i = read_parameters_from_txt(B1_path).reshape((1, len_B1))

        ppm_path = os.path.join(dicts_path, os.listdir(dicts_path)[index + 2])
        offset_ppm_i = read_parameters_from_txt(ppm_path)

        if len(offset_ppm_i) == 303:
            offset_ppm_i = offset_ppm_i[1:]
            offset_ppm_i = offset_ppm_i.reshape((1, len_B1))
        else:
            offset_ppm_i = offset_ppm_i.reshape((1, len_B1))

        dict_path = os.path.join(dicts_path, dict_name)
        tp_i, Trec_i, B0_i, angle_i = get_tp_Trec_values(dict_name)

        parameters_matrix = scipy.io.loadmat(dict_path)

        t1w_i = parameters_matrix['dict']['t1w'][0][0]
        t2w_i = parameters_matrix['dict']['t2w'][0][0]
        t1s_i = parameters_matrix['dict']['t1s'][0][0]
        t2s_i = parameters_matrix['dict']['t2s'][0][0]
        fs_i = parameters_matrix['dict']['fs'][0][0]
        ksw_i = parameters_matrix['dict']['ksw'][0][0]

        parameter_mat = np.hstack((t1w_i, t2w_i, t1s_i, t2s_i, fs_i, ksw_i))

        '''Signal'''
        signal_vec = parameters_matrix['dict']['sig'][0][0].T

        random_idx = np.random.choice(np.arange(n), n_sample, replace=False)

        train_idx = random_idx[:int(n_sample * tr)]
        valid_idx = random_idx[int(n_sample * tr):]

        X_samples_train = parameter_mat[train_idx]
        y_samples_train = signal_vec[train_idx]

        X_samples_valid = parameter_mat[valid_idx]
        y_samples_valid = signal_vec[valid_idx]

        X_train, y_train = from_dict_to_data_scheme(X_samples_train, y_samples_train, tp_i, Trec_i, B1_i, B0_i,
                                                    angle_i, int(n_sample * tr), dw_i, offset_ppm_i)

        X_valid, y_valid = from_dict_to_data_scheme(X_samples_valid, y_samples_valid, tp_i,
                                                    Trec_i, B1_i, B0_i, angle_i, int(n_sample * vl), dw_i, offset_ppm_i)

        number_train = int(len_B1 * n_sample * tr)
        number_valid = int(len_B1 * n_sample * vl)

        X_train_scaled = scale_data(X_train)
        X_valid_scaled = scale_data(X_valid)

        train_data[int(count + pos) * number_train:int(count + pos + 1) * number_train, :] = X_train_scaled
        train_label[int(count + pos) * number_train:int(count + pos + 1) * number_train, :] = y_train

        val_data[int(count + pos) * number_valid:int(count + pos + 1) * number_valid, :] = X_valid_scaled
        val_labels[int(count + pos) * number_valid:int(count + pos + 1) * number_valid, :] = y_valid

    print(f"Loading the data took: {time.time() - total_time}")
    pos += count + 1
    return pos


def train(input_gen, validation_gen, input_model, checkpoint, graph_path, epochs=100, patient_epochs=10):
    # Limits the memory growth of the GPUs while training
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
        filepath=os.path.join(checkpoint, filepath),
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        period=1)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patient_epochs,
                                                restore_best_weights=True)
    input_model.compile(loss='mse', optimizer='adam')  # mse and adam
    # fit the model on the training dataset
    history = input_model.fit(input_gen, epochs=epochs, validation_data=validation_gen,
                              callbacks=[callback, checkpoint_callback], shuffle=True)

    plt.figure(1)
    epochs_list = np.arange(len(history.history['val_loss']))
    plt.plot(epochs_list, history.history['loss'], label='train loss')
    plt.plot(epochs_list, history.history['val_loss'], label='validation loss')
    plt.title('losses vs epochs')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.legend()
    plt.savefig(os.path.join(graph_path,'loss_graph'))


def preprocess_dataset(protocol_names, protocol_paths, parameter_combination, dw_values,
                       batch_size, num_dicts, samples, train_size=0.9):

    index = 0

    for protocol_path, protocol_name, protocol_dw in zip(protocol_paths, protocol_names,
                                                         dw_values):

        protocol_parameter_combinations = parameter_combination[protocol_name]
        if protocol_name == 'cest':
            index = create_data_per_protocol(n=protocol_parameter_combinations, n_sample=int(samples/3),
                                             len_B1=30, n_dicts=num_dicts,
                                             dicts_path=protocol_path, pos=index, dw_i=protocol_dw, tr=train_size)

        elif protocol_name == 'mt':
            index = create_data_per_protocol_changing_ppm(n=protocol_parameter_combinations, n_sample=int(samples/3),
                                                          len_B1=30, n_dicts=num_dicts, dicts_path=protocol_path,
                                                          pos=index, dw_i=protocol_dw, tr=train_size)

        else:
            index = create_data_per_protocol(n=protocol_parameter_combinations, n_sample=samples,
                                             len_B1=10, n_dicts=num_dicts,
                                             dicts_path=protocol_path, pos=index, dw_i=protocol_dw, tr=train_size)

    train_gen = DataGenerator(train_data, train_label, batch_size)
    valid_gen = DataGenerator(val_data, val_labels, batch_size)

    return train_gen, valid_gen


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size_train):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size_train

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def main():
    """
    Change the paths according to your specific folders.
    For this specific acquisition protocols we have two protocols with signals of lengths 30 and the others with
    signal length of 10. In order to sample the data equally between the two protocols, we samples each protocol
    with signal length of 30 by sample/3 to keep the sampling relative to signals of length 10.
    """

    global train_data, train_label, val_data, val_labels


    datasets_dir = '.\\dataset-dynamic'
    loss_graph_path = '.\\train_results-dynamic\\graphs'
    checkpoint_path = '.\\train_results-dynamic\\checkpoints'

    create_directory_if_not_exists(loss_graph_path)
    create_directory_if_not_exists(checkpoint_path)

    # Hyper Parameters - Change according to your case
    num_of_dicts = 1000
    num_of_samples = 22650
    num_protocols = 9
    train_size = 0.9

    num_parameters = 13
    protocol_names_list = ['cest', 'p212', 'p216', 'p218', 'p222', 'p224', 'p248', 'p254', 'mt']
    parameter_combination_per_protocol = {'cest': 665873, 'p212': 176800, 'p216': 397800,
                                          'p218': 397800, 'p222': 286520, 'p224': 286520,
                                          'p248': 140790, 'p254': 140790, 'mt': 26400}
    dw_values_per_protocol = [3, -3.5, 2.75, 4.3, 2.75, 3.5, 3.5, 4.3, -2.5]
    num_epochs = 100
    epochs_patient = 10
    batch_size_dataloader = 512


    # Allocating memory to the train and test datasets and labels for efficient computations
    train_data = np.zeros((int(num_of_dicts * num_of_samples * num_protocols * 10 * train_size), num_parameters+1),
                          dtype=np.float16)
    train_label = np.zeros((int(num_of_dicts * num_of_samples * num_protocols * 10 * train_size), 1),
                           dtype=np.float16)
    val_data = np.zeros((int(num_of_dicts * num_of_samples * num_protocols * 10 * round(1-train_size,8)), num_parameters+1),
                        dtype=np.float16)
    val_labels = np.zeros((int(num_of_dicts * num_of_samples * num_protocols * 10 * round(1-train_size,8)), 1),
                          dtype=np.float16)

    acquisition_protocol_paths = [os.path.join(datasets_dir, protocol) for protocol in protocol_names_list]

    """Creating the dataset"""
    train_set, validation_set = preprocess_dataset(protocol_names_list,
                                                   acquisition_protocol_paths,
                                                   parameter_combination_per_protocol,
                                                   dw_values_per_protocol,
                                                   batch_size_dataloader,
                                                   num_dicts=num_of_dicts,
                                                   samples=num_of_samples,
                                                   train_size=train_size)

    """Load and train the model"""
    loaded_model = create_model_architecture()
    train(train_set, validation_set, loaded_model, checkpoint_path, loss_graph_path,
          epochs=num_epochs, patient_epochs=epochs_patient)


if __name__ == '__main__':
    main()
