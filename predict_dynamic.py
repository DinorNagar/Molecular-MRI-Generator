from matplotlib import pyplot as plt
from keras.models import Sequential
from scipy.stats import pearsonr
from keras.layers import Dense
from scipy.io import savemat
import matplotlib as mpl
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


def create_model_architecture():
    """
    Defines 
    """
    final_model = Sequential()
    final_model.add(Dense(256, input_dim=14, activation='sigmoid', kernel_initializer='he_uniform'))
    final_model.add(Dense(256, input_dim=14, activation='sigmoid', kernel_initializer='he_uniform'))
    final_model.add(Dense(1, activation='sigmoid'))

    return final_model


def scale_data(input_data):
    num_of_parameters = 13
    min_val = np.array([1.3, 0.04, 1.3, 0.00004, 20/111000, 5, 0.25, 1, 1, 3, 60, -3.5, -3.5])
    max_val = np.array([3.4, 1.2, 3.4, 0.04, 30000/110000, 1500, 6, 10, 10, 11.7, 90, 4.3, 20])

    input_data_scaled = np.array([(input_data[:, i] - min_val[i])/(max_val[i] - min_val[i])
                                  for i in range(num_of_parameters)]).T

    input_data_scaled = np.hstack((input_data_scaled, np.expand_dims(input_data[:, num_of_parameters], axis=1)))

    return input_data_scaled


def read_file_values(filename):
    with open(filename, 'r') as f:
        contents = f.read()

    # Split the contents of the file by commas and newlines
    values = []
    for line in contents.split('\n'):
        for value in line.split(','):
            if value.strip():  # Skip empty values
                values.append(float(value.strip()))

    return np.array(values)


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


def create_dict(X, model, B1_length):
    start_time = time.time()
    samples = int(X.shape[0]/B1_length)
    parameters = X[:, :13]
    iterable_mat = np.reshape(parameters, (samples, B1_length, 13))
    dict_mat = np.zeros((B1_length, samples))
    for index in range(iterable_mat.shape[1]):
        if index == 0:
            temp_mat = iterable_mat[:, index, :]
            ones = np.ones((samples, 1))
            temp_mat = np.hstack((temp_mat, ones))
            prev = model.predict(temp_mat, batch_size=5000)
            prev_flat = np.array(prev).flatten()
            dict_mat[index, :] = prev_flat

        else:
            temp_mat = iterable_mat[:, index, :]
            temp_mat = np.hstack((temp_mat, prev))
            prev = model.predict(temp_mat, batch_size=5000)
            prev_flat = np.array(prev).flatten()
            dict_mat[index, :] = prev_flat

    print(f"Creating a dict took: {time.time()-start_time}")
    return dict_mat


def create_predict_signal_figures(X, y, model, len_B1, save_graphs_path):
    tf.keras.backend.clear_session()

    """Signal prediction Example"""
    test_par, test_signal_pred = X[-len_B1:], y[-len_B1:]
    predicted_signal = []
    for ts in test_par:
        y_hat = model.predict(np.expand_dims(ts, axis=0))[0][0]
        predicted_signal.append(y_hat)

    plt.figure(figsize=(15, 10))
    mpl.rcParams['xtick.labelsize'] = 25
    mpl.rcParams['ytick.labelsize'] = 25
    plt.plot(np.arange(1, len(test_signal_pred) + 1), test_signal_pred, 'k', label='Ground Truth ')
    plt.plot(np.arange(1, len(predicted_signal) + 1), predicted_signal, '--bo', label='NN-predicted')
    plt.xlabel('Signal acquisition no.', fontsize=35)
    plt.ylabel('2-norm normalized signal(a.u.)', fontsize=35)
    plt.ylim([0, 1])
    plt.legend(loc='upper left', fontsize="25")

    plt.savefig(os.path.join(save_graphs_path, 'trajectories.png'))
    plt.close()


def calc_nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    range_signals = np.max(y_true) - np.min(y_true)
    nrmse = rmse/range_signals
    return nrmse


def dict_statistics(predicted_dict, sig_values,  B1_length, save_results_path):
    """Calculate the statistic parameters of the result"""

    pearson_val, p_val = pearsonr(predicted_dict.T.reshape(B1_length*predicted_dict.shape[1]), sig_values.squeeze())
    nrmse = calc_nrmse(sig_values.squeeze(), predicted_dict.T.reshape(B1_length*predicted_dict.shape[1]))
    nrmse = format(nrmse * 100, '.2f')

    with open(os.path.join(save_results_path, 'stats.txt'), 'w') as file:
        file.write('Statistical Coefficients of the whole data: \n')
        file.write(f"Pearson Coefficient: {np.round(pearson_val,4)} \n")
        file.write(f"The p-value is: {p_val} \n")
        file.write(f"NRMSE = {nrmse}%\n")

    """ Creating the graph of predicted values vs actual values """
    plt.figure(figsize=(10, 10))
    mpl.rcParams['xtick.labelsize'] = 25
    mpl.rcParams['ytick.labelsize'] = 25
    plt.plot(predicted_dict.T.flatten(), sig_values.flatten(), 'o', markerfacecolor='none',
             markersize=8, color='#093FCF')
    plt.plot(np.linspace(0, 0.8, 10), np.linspace(0, 0.8, 10), '-', color='k', linewidth=4)
    plt.text(0, 0.8, f'pearson > 0.99\n'
                     f'p value < 0.0001\n'
                     f'NRMSE = {nrmse}%', fontsize=25, ha='left', va='top')

    plt.xlabel('Ground Truth', fontsize=35)
    plt.ylabel('NN-predicted', fontsize=35)

    plt.savefig(os.path.join(save_results_path, "statistical_graph.png"))


def create_results_per_protocol(model, dw, dict_path, save_results_path, protocol, is_given_ppm=False):
    dict_name = os.listdir(dict_path)[0]
    B1_txt = os.path.join(dict_path, os.listdir(dict_path)[1])
    """Importing the device parameters"""
    numbers = re.findall(r'\d+\.*\d*', dict_name)
    tp, Trec, B0, angle = [float(num) for num in numbers if num != '0']
    B1 = read_file_values(B1_txt)
    len_B1 = len(B1)
    B1 = B1.reshape((1, len_B1))
    if is_given_ppm:
        offset_ppm_path = os.path.join(dict_path, os.listdir(dict_path)[2])
        offset_ppm = read_file_values(offset_ppm_path).reshape((1, len_B1))

    else:
        offset_ppm = dw * np.ones((1, len_B1))

    """Importing the tissue parameters"""
    mat_path = os.path.join(dict_path, dict_name)
    parameters_matrix = scipy.io.loadmat(mat_path)
    t1w_i = parameters_matrix['dict']['t1w'][0][0]
    t2w_i = parameters_matrix['dict']['t2w'][0][0]
    t1s_i = parameters_matrix['dict']['t1s'][0][0]
    t2s_i = parameters_matrix['dict']['t2s'][0][0]
    fs_i = parameters_matrix['dict']['fs'][0][0]
    ksw_i = parameters_matrix['dict']['ksw'][0][0]
    parameter_mat = np.hstack((t1w_i, t2w_i, t1s_i, t2s_i, fs_i, ksw_i))
    number_of_samples = parameter_mat.shape[0]

    """Importing the signal"""
    signal_vec = parameters_matrix['dict']['sig'][0][0].T

    """Creating the test dataset"""
    data_model, signal_model = from_dict_to_data_scheme(parameter_mat, signal_vec, tp,
                                                        Trec, B1, B0, angle, number_of_samples,
                                                        dw, offset_ppm)

    data_model_scaled = scale_data(data_model)

    """Creating and saving the predicted dictionary dictionary"""
    predicted_dict = create_dict(data_model_scaled, model, len_B1)
    dict_save_name = 'predicted_dict.mat'
    savemat(os.path.join(save_results_path, dict_save_name), {'signal': predicted_dict})

    """Creating the figures and the statistics"""
    dict_statistics(predicted_dict, signal_model, len_B1, save_results_path)
    create_predict_signal_figures(data_model_scaled, signal_model, model, len_B1, save_results_path)


def main():
    best_weight_path = '.\\example_scenarios\\dynamic_network\\weights\\dynamic_weights.hdf5'
    final_dict_save_path = '.\\example_scenarios\\dynamic_network\\stats'
    test_dicts_path = '.\\example_scenarios\\dynamic_network\\test_scenario'

    const_ppm_protocols = ['cest_7T']
    const_ppm_dw = [3]

    create_directory_if_not_exists(directory_path=final_dict_save_path)

    # Importing the model
    loaded_model = create_model_architecture()
    loaded_model.load_weights(best_weight_path)

    for protocol, dw_i in zip(const_ppm_protocols, const_ppm_dw):
        create_results_per_protocol(loaded_model, dw_i, test_dicts_path,
                                    final_dict_save_path, protocol, is_given_ppm=False)
        print(f"Completed protocol {protocol}")


if __name__ == '__main__':
    main()





