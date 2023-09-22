from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import matplotlib.pyplot as plt

import sympy
from time import time
import os
import sys
import datetime
import pathlib
import logging
import argparse
import configparser
import ast
import shutil

print(os.getcwd())
sys.path[0] = os.getcwd()

from utils import ASL, MLP, create_dataset

torch.autograd.set_detect_anomaly(False)

msg = 'Train for one specific method'

# initialize parser
parser = argparse.ArgumentParser(description=msg)
parser.add_argument('-c', '--config', help='Name of the config file:', default='debug.ini')

args = parser.parse_args()

config_name = args.config
config = configparser.ConfigParser()
config.read(os.path.join('trainingOnSyntheticData', 'config', config_name))
results_path = config['META']['results_path']
experiment_name = config['META']['experiment_name']

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def train(a_training, y_training, a_validation, y_validation, h, n_batches, n_epochs, criterion, early_stopping):
    net = MLP(x.shape[0], n_params=a_training.shape[1], h=h).to(device)

    # create your optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    running_loss = 0
    training_loss_list = []
    validation_loss_list = []
    # best_coeff = 0
    # best_loss = torch.inf

    batch_size = int(training_set_size / n_batches)
    logging.info(f'Batch size: {batch_size}')

    for i in range(n_epochs):
        permutation = np.random.permutation(training_set_size)
        y_training_permute = y_training[permutation]
        a_training_permute = a_training[permutation]

        for batch in range(n_batches):
            y_training_batch = y_training_permute[batch * batch_size: (batch + 1) * batch_size]
            a_training_batch = a_training_permute[batch * batch_size: (batch + 1) * batch_size]

            optimizer.zero_grad()  # zero the gradient buffers
            a_pred = net(y_training_batch)
            loss = criterion(a_pred, a_training_batch.squeeze(2))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        if i % 10 == 9:
            a_pred = net(y_validation)
            validation_loss = criterion(a_pred, a_validation.squeeze(2))
            validation_loss_list.append(validation_loss.cpu().detach().numpy())
            training_loss_list.append(running_loss / 10 / n_batches)
            running_loss = 0.0

            # Early stopping
            if early_stopping and i > 1000:
                if validation_loss_list[-1] > validation_loss_list[-2]:
                    break

        if i % 100 == 99:
            logging.info(
                f'[{i + 1:5d}] Training loss: {training_loss_list[-1]:.8f}, Validation loss: {validation_loss_list[-1]:.8f}')

    plt.plot(training_loss_list, label='training loss')
    plt.plot(validation_loss_list, label='validation loss')
    plt.legend()
    plt.yscale('log')
    plt.savefig(
        os.path.join(directory, f'Loss_training_set_size_{training_set_size}_n_batches_{n_batches}_h_{h}.png'))
    plt.close()
    return net


def evaluate_prec_rec_F1(y, a_target, name, results_dict):
    a_pred = (net(y).cpu().detach().numpy() > 0.5).astype(int)
    # a_target = (a_training.detach().numpy().squeeze(-1) + 1) / 2
    a_target = a_target.cpu().detach().numpy().squeeze(-1)
    average = 'micro'

    results_dict[f'Precision score ({name} data)'].append(
        np.round(precision_score(a_target, a_pred, average=average), decimals=5))
    results_dict[f'Recall score ({name} data)'].append(
        np.round(recall_score(a_target, a_pred, average=average), decimals=5))
    results_dict[f'F1 score ({name} data)'].append(np.round(f1_score(a_target, a_pred, average=average), decimals=5))

    logging.info(f'Precision score ({name} data): {results_dict[f"Precision score ({name} data)"][-1]}')
    logging.info(f'Recall score ({name} data): {results_dict[f"Recall score ({name} data)"][-1]}')
    logging.info(f'F1 score ({name} data): {results_dict[f"F1 score ({name} data)"][-1]}')


def evaluate_covering(y, a_target, name, results_dict, quantile, net):
    # Get the score of how many formulas got a complete covering (all important coefficients have been marked as
    # important).
    # Furthermore, calculate the average number of coefficients, in the case that a full covering has been reached.
    prediction = net(y).cpu().detach().numpy()
    if quantile:
        cutoff = np.quantile(prediction, 0.7)
        description = 'quantile cutoff'
    else:
        cutoff = 0.5
        description = '0.5 cutoff'
    a_pred = (prediction > cutoff).astype(int)
    a_target = a_target.cpu().detach().numpy().squeeze(-1)

    complete_coverings = np.sum(a_target > a_pred, axis=1) == 0
    covering_score = np.round(sum(complete_coverings) / y.shape[0], decimals=5)

    results_dict[f'Covering score ({name} data) ({description})'].append(covering_score)
    logging.info(f'Covering score ({name} data) ({description}): {covering_score}')

    coverings = a_pred[complete_coverings]
    avg_complete_cover_size = coverings.mean()

    results_dict[f'Complete cover size ({name} data) ({description})'].append(avg_complete_cover_size)
    logging.info(f'Complete cover size ({name} data) ({description}): {avg_complete_cover_size}')


def evaluate(net, a_training, y_training, a_test, y_test, results_dict):
    evaluate_prec_rec_F1(y_training, a_training, 'training', results_dict)
    evaluate_prec_rec_F1(y_test, a_test, 'test', results_dict)

    evaluate_covering(y_training, a_training, 'training', results_dict, True, net)
    evaluate_covering(y_test, a_test, 'test', results_dict, True, net)
    evaluate_covering(y_training, a_training, 'training', results_dict, False, net)
    evaluate_covering(y_test, a_test, 'test', results_dict, False, net)

    criterion = nn.MSELoss()
    mse_test = np.round(criterion(net(y_test), a_test.squeeze(-1)).item(), decimals=5)
    results_dict['mse_test'].append(mse_test)
    logging.info(f'MSE (test data): {mse_test}')

    criterion = nn.BCELoss()
    mse_test = np.round(criterion(net(y_test), a_test.squeeze(-1)).item(), decimals=5)
    results_dict['bce_test'].append(mse_test)
    logging.info(f'BCE (test data): {mse_test}')


def construct_result_dict(entry_names):
    results_dict = {}
    for entry_name in entry_names:
        results_dict[entry_name] = []
    return results_dict


def append_results_dict(results_dict, test_set_size, validation_set_size, function_names, degree_input_polynomials,
                        degree_output_polynomials, width, training_set_size, h, n_batches, t_training, t_data_creation,
                        n_epochs, loss_name, early_stopping, ensure_uniqueness, gamma_neg, p_target, normalization,
                        degree_output_polynomials_specific, n_functions_max):
    results_dict['test_set_size'].append(test_set_size)
    results_dict['validation_set_size'].append(validation_set_size)
    results_dict['function_names'].append(function_names)
    results_dict['degree_input_polynomials'].append(degree_input_polynomials)
    results_dict['degree_output_polynomials'].append(degree_output_polynomials)
    results_dict['width'].append(width)
    results_dict['training_set_size'].append(training_set_size)
    results_dict['hidden_neurons'].append(h)
    results_dict['n_batches'].append(n_batches)
    results_dict['t_training'].append(t_training)
    results_dict['t_data_creation'].append(t_data_creation)
    results_dict['n_epochs'].append(n_epochs)
    results_dict['loss'].append(loss_name)
    results_dict['early_stopping'].append(early_stopping)
    results_dict['ensure_uniqueness'].append(ensure_uniqueness)
    results_dict['gamma_neg'].append(gamma_neg)
    results_dict['p_target'].append(p_target)
    results_dict['normalization'].append(normalization)
    results_dict['degree_output_polynomials_specific'].append(degree_output_polynomials_specific)
    results_dict['n_functions_max'].append(n_functions_max)


if __name__ == '__main__':
    datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
    directory = os.path.join(results_path, datetime + experiment_name)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    shutil.copy(os.path.join('trainingOnSyntheticData', 'config', config_name), directory)

    logging.basicConfig(filename=os.path.join(directory, 'experiment.log'), level=logging.INFO)
    logging.info('Starting the logger.')
    logging.debug(f'Directory: {directory}')
    logging.debug(f'File: {__file__}')

    logging.info(f'Using {device}.')

    logging.info(f'############### Starting experiment with config file {config_name} ###############')

    hidden_neurons = ast.literal_eval(config['PARAMETERS']['hidden_neurons'])
    n_batches_list = ast.literal_eval(config['PARAMETERS']['n_batches_list'])
    training_set_sizes = ast.literal_eval(config['PARAMETERS']['training_set_sizes'])

    test_set_size = int(config['PARAMETERS']['test_set_size'])
    validation_set_size = int(config['PARAMETERS']['validation_set_size'])

    x = np.arange(-10, 10, 0.1)
    x = x.reshape(len(x), 1)
    functions = [np.sin, lambda x: np.sqrt(np.abs(x))]
    function_names = [sympy.sin, lambda x: sympy.sqrt(sympy.Abs(x))]
    degree_output_polynomials_specific = [1, 1]
    degree_input_polynomials = int(config['PARAMETERS']['degree_input_polynomials'])
    degree_output_polynomials = int(config['PARAMETERS']['degree_output_polynomials'])
    width = int(config['PARAMETERS']['width'])
    n_epochs = int(config['PARAMETERS']['n_epochs'])
    n_functions_max = int(config['PARAMETERS']['n_functions_max'])
    loss_name = config['PARAMETERS']['loss']
    early_stopping = config['PARAMETERS']['early_stopping'] == 'True'
    ensure_uniqueness = config['PARAMETERS']['ensure_uniqueness'] == 'True'
    normalization = config['PARAMETERS']['normalization']
    p_target = float(config['PARAMETERS']['p_target'])
    gamma_neg = ast.literal_eval(config['PARAMETERS']['gamma_neg'])

    if loss_name == 'MSE':
        criterion = nn.MSELoss()
    elif loss_name == 'BCE':
        criterion = nn.BCELoss()
    elif loss_name == 'ASL':
        update = False
        if update:
            logging.info('Activated automatic update rule for ASL.')
        else:
            logging.info('Deactivated automatic update rule for ASL.')
        asl_obj = ASL(gamma_negative=gamma_neg, p_target=p_target, update=True)
        criterion = asl_obj.asl
    else:
        raise NotImplementedError(
            f'Wrong loss specification: expected one out of "MSE", "BCE" and "ASL", but received {loss_name}.')

    entry_names = ['ensure_uniqueness', 'loss', 'test_set_size', 'validation_set_size', 'function_names',
                   'degree_output_polynomials_specific', 'n_functions_max',
                   'degree_input_polynomials', 'gamma_neg', 'p_target', 'normalization',
                   'degree_output_polynomials', 'width', 'n_epochs', 'training_set_size', 'hidden_neurons', 'n_batches',
                   't_training', 't_data_creation', 'Precision score (training data)', 'Recall score (training data)',
                   'F1 score (training data)', 'Precision score (test data)', 'Recall score (test data)',
                   'F1 score (test data)', 'mse_test', 'bce_test', 'early_stopping',
                   'Covering score (training data) (quantile cutoff)',
                   'Covering score (training data) (0.5 cutoff)',
                   'Covering score (test data) (quantile cutoff)',
                   'Covering score (test data) (0.5 cutoff)',
                   'Complete cover size (training data) (quantile cutoff)',
                   'Complete cover size (training data) (0.5 cutoff)',
                   'Complete cover size (test data) (quantile cutoff)',
                   'Complete cover size (test data) (0.5 cutoff)',
                   ]
    results_dict = construct_result_dict(entry_names)

    for training_set_size in training_set_sizes:
        logging.info(
            f'######## Starting with training set size {training_set_size} out of {training_set_sizes} now. ########')

        dataset_size = training_set_size + validation_set_size + test_set_size
        t_0 = time()
        y, a, params = create_dataset(dataset_size, ensure_uniqueness, degree_input_polynomials,
                                      degree_output_polynomials,
                                      width, functions, function_names, normalization, n_functions_max=n_functions_max,
                                      degree_output_polynomials_specific=degree_output_polynomials_specific,
                                      device=device, enforce_function=False)
        t_1 = time()
        t_data_creation = np.round(t_1 - t_0, 3)
        logging.info(f'Creating the dataset took {t_data_creation}s.')
        y_training = y[:training_set_size]
        a_training = a[:training_set_size]
        y_validation = y[training_set_size:training_set_size + validation_set_size]
        a_validation = a[training_set_size:training_set_size + validation_set_size]
        y_test = y[training_set_size + validation_set_size:]
        a_test = a[training_set_size + validation_set_size:]

        for h in hidden_neurons:
            logging.info(f'###### Starting with {h} hidden neurons out of {hidden_neurons} now. ######')
            for n_batches in n_batches_list:
                logging.info(f'#### Starting with {n_batches} batches out of {n_batches_list} now. ####')
                t_0 = time()
                net = train(a_training=a_training, y_training=y_training, a_validation=a_validation,
                            y_validation=y_validation, h=h, n_batches=n_batches, n_epochs=n_epochs, criterion=criterion,
                            early_stopping=early_stopping)
                t_1 = time()
                t_training = np.round(t_1 - t_0, 3)
                logging.info(f'Training the model took {t_training}s.')
                evaluate(net=net, a_training=a_training, y_training=y_training, a_test=a_test, y_test=y_test,
                         results_dict=results_dict)
                append_results_dict(results_dict, test_set_size, validation_set_size, function_names,
                                    degree_input_polynomials, degree_output_polynomials, width, training_set_size, h,
                                    n_batches, t_training, t_data_creation, n_epochs, loss_name, early_stopping,
                                    ensure_uniqueness, gamma_neg, p_target, normalization,
                                    degree_output_polynomials_specific, n_functions_max)
                results_pd = pd.DataFrame(results_dict)
                results_pd.to_csv(os.path.join(directory, 'test.csv'))
                torch.save(net, os.path.join(directory,
                                             f'model_training_set_size_{training_set_size}_hidden_neurons_{h}_n_batches'
                                             f'_{n_batches}.pt'))
                if loss_name == 'ASL' and asl_obj.update:
                    logging.info(f'The last gamma_negative of ASL was {asl_obj.gamma_negative}.')
