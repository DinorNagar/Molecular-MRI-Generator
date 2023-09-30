from matplotlib import pyplot as plt
from keras.models import Sequential
from scipy.stats import pearsonr
from keras.layers import Dense
import matplotlib as mpl
import numpy as np
import scipy.io
import time
import re
import os


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


def create_model_architecture():
    final_model = Sequential()
    final_model.add(Dense(256, input_dim=31, activation='sigmoid', kernel_initializer='he_uniform'))
    final_model.add(Dense(256, input_dim=31, activation='sigmoid', kernel_initializer='he_uniform'))
    final_model.add(Dense(34, activation='sigmoid'))

    return final_model


def scale_data(input_data):
    num_of_parameters = 31
    scaler_x = np.array([1.6, 0.12, 1, 1, 0.1, 0.0045045, 90, 1, 0.1, 0.0014, 1000, 1, 0.1, 0.0045045, 5000, 1, 0.1,
                         0.0045045, 3500, 1.3, 0.005, 0.00675, 20, 1, 0.00004, 0.216, 30, 2.4, 3.1, 11.7, 30, 1])

    input_data_scaled = np.array([input_data[:, i] / scaler_x[i] for i in range(num_of_parameters)]).T

    return np.round(input_data_scaled, 6)


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


def from_dict_to_data_scheme(parameter_mat, signal_matrix, tp, Trec, B0, n_pulses, samples_size):

    tp_vec = np.repeat(tp, samples_size, axis=0).reshape((samples_size, 1))
    Trec_vec = np.repeat(Trec, samples_size, axis=0).reshape((samples_size, 1))
    B0_vec = np.repeat(B0, samples_size, axis=0).reshape((samples_size, 1))
    n_pulses_vec = np.repeat(n_pulses, samples_size, axis=0).reshape((samples_size, 1))
    input_data = np.hstack((parameter_mat, tp_vec, Trec_vec, B0_vec, n_pulses_vec))
    output_data = signal_matrix

    return input_data, output_data


def create_dict(X, model):
    start_time = time.time()
    dict_mat = model.predict(X, batch_size=5000)
    print(f"Creating a dict took: {time.time()-start_time}")
    return dict_mat


def calc_nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    range_signals = np.max(y_true) - np.min(y_true)

    return np.round(rmse/range_signals, decimals=4)


def create_predict_signal_figures(y_predict, y_test, save_graphs_path):

    plt.figure(figsize=(15, 10))
    mpl.rcParams['xtick.labelsize'] = 25
    mpl.rcParams['ytick.labelsize'] = 25
    plt.plot(y_test[0, 1:], 'k', label='Ground Truth ')
    plt.plot(y_predict[0, 1:], 'bo', label='NN-predicted')
    plt.xlabel('Signal acquisition no.', fontsize=35)
    plt.ylabel('2-norm normalized signal(a.u.)', fontsize=35)
    plt.ylim([0, 1])
    plt.legend(loc='upper right', fontsize="25")
    plt.savefig(os.path.join(save_graphs_path, 'trajectories.png'))
    plt.close()


def dict_statistics(predicted_dict, sig_values, save_results_path):

    """Calculate the statistic parameters of the result"""
    pearson_val, p_val = pearsonr(predicted_dict.flatten(), sig_values.flatten())
    nrmse = calc_nrmse(predicted_dict.flatten(), sig_values.flatten())
    with open(os.path.join(save_results_path, 'stats.txt'), 'w') as file:
        file.write('Statistical Coefficients of the whole data: \n')
        file.write(f"Pearson Coefficient: {np.round(pearson_val,4)} \n")
        file.write(f"The p-value is: {p_val} \n")
        file.write(f"MSE:{nrmse}\n")

    """Creating the graph of predicted values vs actual values"""
    plt.figure(figsize=(10, 10))
    mpl.rcParams['xtick.labelsize'] = 25
    mpl.rcParams['ytick.labelsize'] = 25
    plt.plot(predicted_dict.flatten(), sig_values.flatten(), 'o', markerfacecolor='none',
             markersize=8, color='#093FCF')
    plt.plot(np.linspace(0, 0.8, 10), np.linspace(0, 0.8, 10), '-', color='k', linewidth=4)
    plt.text(0.1, 0.85, f'pearson > 0.9999\n'
                        f'p value < 0.0001\n'
                        f'NRMSE = {nrmse*100}%', fontsize=25, ha='left', va='top')

    plt.xlabel('Ground Truth', fontsize=35)
    plt.ylabel('NN-predicted', fontsize=35)
    plt.xlim([0, 0.9])
    plt.ylim([0, 0.9])
    plt.savefig(os.path.join(save_results_path, "statistical_graph.png"))


def create_results_per_protocol(model, dict_path, save_results_path):
    dict_name = os.listdir(dict_path)[0]
    test_dict_path = os.path.join(dict_path, dict_name)

    # Importing the device parameters
    B0_i, tp_i, Trec_i, n_pulses_i = get_tp_Trec_values(dict_name)

    # Creating the parameter_matrix
    parameters_matrix = scipy.io.loadmat(test_dict_path)

    signal_vec = parameters_matrix['sig']

    del parameters_matrix['__header__']
    del parameters_matrix['__version__']
    del parameters_matrix['__globals__']
    del parameters_matrix['sig']

    array_parameters_list = list(parameters_matrix.values())
    parameter_mat = np.vstack(array_parameters_list).T
    parameter_mat = parameter_mat.astype('float32')

    # Creating the test dataset
    data_model, signal_model = from_dict_to_data_scheme(parameter_mat, signal_vec, tp_i,
                                                        Trec_i, B0_i, n_pulses_i, parameter_mat.shape[0])

    data_model_scaled = scale_data(data_model)

    # Creating and saving the predicted dictionary dictionary
    predicted_dict = create_dict(data_model_scaled, model)

    scipy.io.savemat(os.path.join(save_results_path, 'predicted_dict.mat'), {'signal': predicted_dict})

    # Creating the figures and the statistics
    dict_statistics(predicted_dict, signal_model, save_results_path)
    create_predict_signal_figures(predicted_dict, signal_model, save_results_path)


def main():
    best_weight_path = '.\\example_scenarios\\application_optimized_network\\weights\\application_weights.hdf5'
    final_dict_save_path = '.\\example_scenarios\\application_optimized_network\\stats'
    test_dict_protocol_path = '.\\example_scenarios\\application_optimized_network\\test_scenario'

    create_directory_if_not_exists(final_dict_save_path)
    """Importing the model"""
    loaded_model = create_model_architecture()
    loaded_model.load_weights(best_weight_path)

    create_results_per_protocol(loaded_model, test_dict_protocol_path, final_dict_save_path)


if __name__ == '__main__':
    main()




