import copy

import logging
import random

import numpy as np
import os
import pathos.multiprocessing as mp
import sympy
import sys
import time
import torch
from scipy.optimize import differential_evolution, basinhopping, dual_annealing, minimize
from sklearn.model_selection import train_test_split
import ast
import pandas as pd

from parfam_torch import ParFamTorch, Evaluator
# from reparfam import ReParFam
from utils import add_noise, relative_l2_distance, r_squared, is_tuple_sorted, function_dict, function_name_dict
import utils
from pathlib import Path
import configparser

from utils import add_noise, relative_l2_distance, r_squared, get_complete_model_parameter_list, SetTransformer, tensor_to_model_parameters
from trainingOnSyntheticData.test_dlparfam import get_topk_predictions, filter_predictions
sys.path.append(os.path.dirname(os.getcwd()))
from multiprocessing import Process, Event, Manager, Lock
from time import time, sleep
from datetime import datetime
import itertools

# import tracemalloc
# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'
#
# # device = 'cpu'
#
# print(f'Using {device} (file: {__file__}))')


def extend_function_dict(extension_function_dict, extension_function_name_dict):
    global function_dict
    global function_name_dict
    function_dict = {**function_dict, **extension_function_dict}
    function_name_dict = {**function_name_dict, **extension_function_name_dict}
# import torch.nn.functional as F

# # import tracemalloc
# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'


# print(f'Using {device} (file: {__file__}))')

# function_dict = {'sqrt': lambda x: torch.sqrt(torch.abs(x)),
#                  'exp': lambda x: torch.minimum(torch.exp(x), np.exp(10) + torch.abs(x)),
#                  'cos': torch.cos, 'sin': torch.sin}
# function_name_dict = {'sqrt': lambda x: sympy.sqrt(sympy.Abs(x)), 'exp': sympy.exp, 'cos': sympy.cos, 'sin': sympy.sin}


def make_unpickable(model_parameters):
    function_names_str = model_parameters['functions']
    model_parameters['functions'] = []
    model_parameters['function_names'] = []
    for function_name_str in function_names_str:
        model_parameters['functions'].append(function_dict[function_name_str])
        model_parameters['function_names'].append(function_name_dict[function_name_str])
    return model_parameters


# Evaluates the model performance
def evaluate(model, coefficients, x_train, y_train, x_val, y_val, target_expr="", custom_loss=None):
    if target_expr != "":
        print(f'Target expression: {target_expr}')
    if custom_loss is None:
        relative_l2_distance_train = relative_l2_distance(model, coefficients, x_train, y_train)
        relative_l2_distance_val = relative_l2_distance(model, coefficients, x_val, y_val)
        r_squared_val = r_squared(model, coefficients, x_val, y_val)

        print(f'Relative l_2-distance train: {relative_l2_distance_train}')
        print(f'Relative l_2-distance validation: {relative_l2_distance_val}')
    else:
        y_pred_train = model.predict(coefficients, x_train)
        custom_loss_train = custom_loss(y_pred_train, y_train, None)
        y_pred_val = model.predict(coefficients, x_val)
        custom_loss_val = custom_loss(y_pred_val, y_val, None)
        
        print(f'Custom loss train: {custom_loss_train}')
        print(f'Custom loss val: {custom_loss_val}')
        
        relative_l2_distance_train, relative_l2_distance_val, r_squared_val = custom_loss_train, custom_loss_val, custom_loss_val
        
    return relative_l2_distance_train, relative_l2_distance_val, r_squared_val


# trains a model with a given model and evaluator
def training(model, evaluator, x, y, n_params_active, maxiter, initial_temp, maxfun, no_local_search, optimizer,
             maxiter_per_dim_local_minimizer, local_minimizer, x0=None):
    lw = [-10] * n_params_active
    up = [10] * n_params_active
    model.prepare_input_monomials(x)
    # Train with chosen optimizer
    if optimizer == 'dual_annealing':
        ret = dual_annealing(evaluator.loss_func, bounds=list(zip(lw, up)), maxiter=maxiter, maxfun=maxfun,
                             initial_temp=initial_temp, no_local_search=no_local_search, x0=x0,
                             minimizer_kwargs={'options': {'maxiter': maxiter_per_dim_local_minimizer * x.shape[1]},
                                               'method': local_minimizer,
                                               'jac': evaluator.gradient})
        coefficients = ret.x
    elif optimizer == 'basinhopping':
        if x0 is None:
            x0 = np.random.randn(n_params_active)

        ret = basinhopping(evaluator.loss_func, niter=int(maxiter / 10), x0=x0,
                           minimizer_kwargs={'options': {'maxiter': maxiter_per_dim_local_minimizer * x.shape[1]},
                                             'method': local_minimizer,
                                             'jac': evaluator.gradient})
        # displaying the memory
        # logging.info(f'Memory usage (current, peak): {tracemalloc.get_traced_memory()}')
        coefficients = ret.x
    elif optimizer == 'differential_evolution':
        ret = differential_evolution(evaluator.loss_func, bounds=list(zip(lw, up)), maxiter=maxiter, x0=x0, popsize=100)
        coefficients = ret.x
    elif optimizer == 'lbfgs':
        coefficients = train_lbfgs(evaluator, n_params_active, maxiter)
    elif optimizer == 'bfgs':
        coefficients = train_bfgs(evaluator, n_params_active, maxiter)
    model.testing_mode()
    return coefficients


def train_bfgs(evaluator, n_params_active, maxiter, coefficients=None):
    best_coefficients = None
    best_loss = float('inf')
    for i in range(maxiter):
        if coefficients is None:
            coefficients = torch.randn(n_params_active, dtype=torch.double) * 5
        ret = minimize(evaluator.loss_func, x0=coefficients, method='BFGS', jac=evaluator.gradient,
                       options={'maxiter': 200 * n_params_active})
        coefficients = ret.x
        loss = evaluator.loss_func(coefficients)
        if loss < best_loss:
            best_loss = loss
            best_coefficients = coefficients
            if loss < 10 ** (-5):
                break
        coefficients = None
    if best_coefficients is None:
        logging.warning(f'best_coefficients is None, something that should not happen. \n'
                        f'Settings: best_loss={best_loss}, '
                        f'loss={loss}, maxiter={maxiter}, coefficients={coefficients}.'
                        f' \nThe coefficients will be set to 0, to avoid an error.')
        best_coefficients = torch.zeros(n_params_active, dtype=torch.double)
    return best_coefficients


def train_lbfgs(evaluator, n_params_active, maxiter, coefficients=None):
    best_coefficients = None
    best_relative_l2_distance = float('inf')
    for i in range(maxiter):
        if coefficients is None:
            coefficients = torch.randn(n_params_active, dtype=torch.double) * 5
        coefficients, relative_l2_distance = evaluator.fit_lbfgs(coefficients, verbose=False)
        if relative_l2_distance < best_relative_l2_distance:
            best_relative_l2_distance = relative_l2_distance
            best_coefficients = coefficients
            if relative_l2_distance < 10 ** (-5):
                break
        coefficients = None
    if best_coefficients is None:
        logging.warning(f'best_coefficients is None, something that should not happen. \n'
                        f'Settings: best_relative_l2_distance={best_relative_l2_distance}, '
                        f'relative_l2_distance={relative_l2_distance}, maxiter={maxiter}, coefficients={coefficients}.'
                        f' \nThe coefficients will be set to 0, to avoid an error.')
        best_coefficients = torch.zeros(n_params_active, dtype=torch.double)
    return best_coefficients


# trains a model with given hyperparameters
def hyperparameter_training(x, y, target_expr, training_parameters, model_parameters, seed, verbose=True, custom_loss=None):
    if verbose:
        print('##### Training #####')
    # Hyperparameters
    maxiter1 = training_parameters['maxiter1']
    maxiter2 = training_parameters['maxiter2']
    optimizer = training_parameters['optimizer']
    maxiter_per_dim_local_minimizer = training_parameters['maxiter_per_dim_local_minimizer']
    lambda_1 = training_parameters['lambda_1']
    lambda_1_piecewise = training_parameters['lambda_1_piecewise']
    lambda_1_cut = training_parameters['lambda_1_cut']
    # Split into training and validation data
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=seed)
    # Some of the data sets are too big
    max_dataset_length = training_parameters['max_dataset_length']
    if len(y_train) > max_dataset_length:
        subset = np.random.choice(len(y_train), max_dataset_length, replace=False)
    else:
        subset = np.arange(len(y_train))
    x_train = x_train[subset]
    y_train = y_train[subset]

    # Set up model
    model = setup_model(model_parameters, n_input=x.shape[1], device=training_parameters['device'],
                        model=training_parameters['model'])
    t_0 = time()
    n_params = model.get_number_parameters()
    if training_parameters['classifier'] is None:
        mask = None
        n_params_active = n_params
    else:
        net_path = os.path.join('trainingOnSyntheticData/trained_models', training_parameters['classifier'])
        net = torch.load(net_path, map_location=torch.device(training_parameters['device']))
        prediction = net(torch.tensor(y, dtype=torch.float, device=training_parameters['device']))
        mask = prediction >= 0.2
        n_params_active = sum(mask)
    logging.info(f'Number parameters: {n_params}')
    print(f'Number parameters: {n_params}')
    logging.info(f'Number parameters active: {n_params_active}')
    print(f'Number parameters active: {n_params_active}')
    evaluator = Evaluator(x_train, y_train, lambda_0=0, lambda_1=lambda_1, model=model, mask=mask,
                          n_params=n_params, lambda_1_cut=lambda_1_cut, lambda_1_piecewise=lambda_1_piecewise,
                          custom_loss=custom_loss)
    coefficients_ = training(model, evaluator, x_train, y_train, n_params_active, maxiter=maxiter1, initial_temp=50000,
                             maxfun=10 ** 9, no_local_search=False, optimizer=optimizer,
                             maxiter_per_dim_local_minimizer=maxiter_per_dim_local_minimizer,
                             local_minimizer=training_parameters['local_minimizer'])
    logging.info(f'Number of evaluations: {evaluator.evaluations}')
    coefficients = torch.zeros(n_params, device=training_parameters['device'], dtype=torch.double)
    coefficients[mask] = torch.tensor(coefficients_, device=training_parameters['device'], dtype=torch.double)
    # For Dual_annealing: Run model for a second time with local search
    # if maxiter2 > 0 and optimizer == 'dual_annealing':
    #    coefficients = training(model, evaluator, x_train, y_train, n_params_active, maxiter=maxiter2, initial_temp=500,
    #                            maxfun=10 ** 9, maxiter_per_dim_local_minimizer=maxiter_per_dim_local_minimizer,
    #                            no_local_search=False, x0=coefficients, optimizer=optimizer)
    for _ in range(training_parameters['pruning_iterations'] - 1):
        coefficients = model.get_normalized_coefficients(coefficients)
        mask = torch.abs(coefficients) > training_parameters['pruning_cut_off']
        evaluator = Evaluator(x_train, y_train, lambda_0=0, lambda_1=lambda_1, model=model, mask=mask,
                              n_params=n_params, lambda_1_cut=lambda_1_cut, lambda_1_piecewise=lambda_1_piecewise)
        active_params = sum(mask)
        print(f'Active parameters {active_params}')

        coefficients_ = training(model, evaluator, x_train, y_train, active_params, maxiter=maxiter1,
                                 initial_temp=50000, maxfun=10 ** 9, no_local_search=True, optimizer=optimizer,
                                 maxiter_per_dim_local_minimizer=maxiter_per_dim_local_minimizer,
                                 local_minimizer=training_parameters['local_minimizer'])
        coefficients = torch.zeros(n_params, device=training_parameters['device'], dtype=torch.double)
        coefficients[mask] = torch.tensor(coefficients_, device=training_parameters['device'], dtype=torch.double)
    t_1 = time()
    print(f'Training time: {t_1 - t_0}')
    relative_l2_distance_train, relative_l2_distance_val, r_squared_val = evaluate(model, coefficients, x_train,
                                                                                   y_train, x_val, y_val,
                                                                                   target_expr, custom_loss=custom_loss)
    return relative_l2_distance_train, relative_l2_distance_val, r_squared_val, t_1 - t_0, coefficients, evaluator.evaluations


stop_flag = 0  # global variable to check whether the wanted accuracy was reached, set to 1 if that is the case


def train_model_old_event(model_parameters, x, y, target_expr, training_parameters, seed, accuracy, found_event,
                          result_dict, lock):
    if found_event.is_set():
        logging.info('Event is set, but the process is still running. what happened?')
    relative_l2_distance_train, relative_l2_distance_val, r_squared_val, training_time, coefficients, n_evaluations = \
        hyperparameter_training(x, y, target_expr, training_parameters, copy.deepcopy(model_parameters), seed,
                                verbose=True)
    with lock:
        logging.info(f'Training parameters: {training_parameters}')
        logging.info(f'Model parameters: {model_parameters}')
        logging.info(f'Relative l2 distance train: {relative_l2_distance_train}')
        logging.info(f'Relative l2 distance validation: {relative_l2_distance_val}')
        logging.info(f'R squared validation: {r_squared_val}')
        logging.info(f'Training time: {training_time}')

        if result_dict['relative_l2_distance_val'] > relative_l2_distance_val:
            logging.info(f"New best relative l2 distance validation: {relative_l2_distance_val}")
            result_dict['relative_l2_distance_train'] = relative_l2_distance_train
            result_dict['relative_l2_distance_val'] = relative_l2_distance_val
            result_dict['r_squared_val'] = r_squared_val
            result_dict['coefficients'] = coefficients
            result_dict['training_time'] = training_time
            result_dict['model_parameters'] = model_parameters

        # Check whether wanted accuracy was reached
        if relative_l2_distance_val < accuracy:
            logging.info(f"Terminate proccess, wanted accuracy {accuracy} was reached")
            result_dict['relative_l2_distance_train'] = relative_l2_distance_train
            result_dict['relative_l2_distance_val'] = relative_l2_distance_val
            result_dict['r_squared_val'] = r_squared_val
            result_dict['coefficients'] = coefficients
            result_dict['training_time'] = training_time
            result_dict['model_parameters'] = model_parameters
            found_event.set()
        else:
            logging.info(f'Accuracy: {accuracy}; Relative l2 distance validation: {relative_l2_distance_val}')

        result_dict['counter'] -= 1
        logging.info(f'Counter: {result_dict["counter"]}')

        if result_dict['counter'] == 0:
            print(f'All models have been tested, no good solution was found.')
            found_event.set()


def train_model(model_parameters, x, y, target_expr, training_parameters, seed, result_dict, lock):
    relative_l2_distance_train, relative_l2_distance_val, r_squared_val, training_time, coefficients, n_evaluations = hyperparameter_training(
        x, y, target_expr, training_parameters, model_parameters, seed, verbose=True)
    with lock:
        logging.info(f'Model parameters: {model_parameters}')
        logging.info(f'Relative l2 distance train: {relative_l2_distance_train}')
        logging.info(f'Relative l2 distance validation: {relative_l2_distance_val}')
        logging.info(f'R squared validation: {r_squared_val}')
        logging.info(f'Training time: {training_time}')

        if result_dict['relative_l2_distance_val'] > relative_l2_distance_val:
            logging.info(f"New best relative l2 distance validation: {relative_l2_distance_val}")
            result_dict['relative_l2_distance_train'] = relative_l2_distance_train
            result_dict['relative_l2_distance_val'] = relative_l2_distance_val
            result_dict['r_squared_val'] = r_squared_val
            result_dict['coefficients'] = coefficients
            result_dict['training_time'] = training_time

        result_dict['counter'] -= 1
        logging.info(f'Counter: {result_dict["counter"]}')

def get_polynomial_model_parameters_list(degree_output_polynomials, maximal_potence):
    model_parameter_list = []
    # Start with checking if it is a polynomial
    model_parameters = {'degree_input_polynomials': 0, 'degree_output_polynomials': degree_output_polynomials,
                        'degree_output_denominator': 0, 'degree_input_denominator': 0,
                        'function_names': [], 'width': 1, 'functions': [],
                        'degree_output_polynomials_specific': None,
                        'degree_output_polynomials_denominator_specific': None,
                        'enforce_function': False, 'maximal_potence': maximal_potence}
    model_parameter_list.append(model_parameters)
    return model_parameter_list

def get_rational_model_parameters_list(max_deg_output_denominator, max_deg_output_numerator, maximal_potence):
    model_parameter_list = []
    for d_output_denom in range(1, max_deg_output_denominator + 1):
        for d_output in range(1, max_deg_output_numerator + 1):
            model_parameters = {'degree_input_polynomials': 0, 'degree_output_polynomials': d_output,
                                'degree_output_denominator': d_output_denom,'degree_input_denominator': 0,
                                'functions': [],
                                'function_names': [], 'width': 1, 
                                'degree_output_polynomials_specific': None,
                                'degree_output_polynomials_denominator_specific': None,
                                'enforce_function': False,
                                'maximal_potence': maximal_potence}
            model_parameter_list.append(model_parameters)
    return model_parameter_list

def get_complete_model_parameter_list(model_parameters_max, model_str, enforce_function_iterate):
    enforce_function_iterate_list = get_enforce_function_iterate_list(enforce_function_iterate)

    max_deg_input = model_parameters_max['max_deg_input']
    max_deg_output = model_parameters_max['max_deg_output']
    max_deg_input_denominator = model_parameters_max['max_deg_input_denominator']
    max_deg_output_denominator = model_parameters_max['max_deg_output_denominator']
    function_names = model_parameters_max['function_names']
    functions = model_parameters_max['function_names']
    maximal_n_functions = model_parameters_max['maximal_n_functions']
    max_deg_output_polynomials_specific = model_parameters_max['max_deg_output_polynomials_specific']
    max_deg_output_polynomials_denominator_specific = model_parameters_max[
        'max_deg_output_polynomials_denominator_specific']

    model_parameter_list = []
    # Start with checking if it is a polynomial
    model_parameter_list += get_polynomial_model_parameters_list(degree_output_polynomials=max_deg_output, maximal_potence=model_parameters_max['maximal_potence'])

    # Continue with checking if it is a rational function
    model_parameter_list += get_rational_model_parameters_list(max_deg_output_denominator=max_deg_output_denominator,
                                                                 max_deg_output_numerator=max_deg_output, maximal_potence=model_parameters_max['maximal_potence'])

    # Try one function at a time while building up the polynomial degrees
    for n_functions in range(1, maximal_n_functions + 1):
        for d_output_denom in range(max_deg_output_denominator + 1):
            for d_output in range(1, max_deg_output + 1):
                for d_input_denom in range(max_deg_input_denominator + 1):
                    for d_input in range(1, max_deg_input + 1):
                        for function_indices in set(itertools.combinations(list(range(len(functions))) * n_functions,
                                                                           n_functions)):
                            if not is_tuple_sorted(function_indices):
                                # This ensures that we do not check any combination more than once
                                continue
                            for enforce_function in enforce_function_iterate_list:
                                if model_str == 'ReParFam' and enforce_function:
                                    continue
                                functions_ = []
                                function_names_ = []
                                deg_output_polynomials_specific = []
                                deg_output_polynomials_denominator_specific = []
                                for i in function_indices:
                                    functions_.append(functions[i])
                                    function_names_.append(function_names[i])
                                    deg_output_polynomials_specific.append(max_deg_output_polynomials_specific[i])
                                    deg_output_polynomials_denominator_specific.append(
                                        max_deg_output_polynomials_denominator_specific[i])
                                model_parameters = {'degree_input_polynomials': d_input,
                                                    'degree_output_polynomials': d_output,
                                                    'functions': functions_, 'function_names': function_names_,
                                                    'width': 1,
                                                    'degree_input_denominator': d_input_denom,
                                                    'degree_output_denominator': d_output_denom,
                                                    'degree_output_polynomials_specific': deg_output_polynomials_specific,
                                                    'degree_output_polynomials_denominator_specific': deg_output_polynomials_denominator_specific,
                                                    'enforce_function': enforce_function,
                                                    'maximal_potence': model_parameters_max['maximal_potence']
                                                    }
                                model_parameter_list.append(model_parameters)

    return model_parameter_list


def get_max_model_parameter_list(model_parameters_max):
    max_deg_input = model_parameters_max['max_deg_input']
    max_deg_output = model_parameters_max['max_deg_output']
    max_deg_input_denominator = model_parameters_max['max_deg_input_denominator']
    max_deg_output_denominator = model_parameters_max['max_deg_output_denominator']
    function_names = model_parameters_max['function_names']
    functions = model_parameters_max['function_names']
    max_deg_output_polynomials_specific = model_parameters_max['max_deg_output_polynomials_specific']
    max_deg_output_polynomials_denominator_specific = model_parameters_max[
        'max_deg_output_polynomials_denominator_specific']

    model_parameter_list = []
    # Start with checking if it is a polynomial
    model_parameter_list += get_polynomial_model_parameters_list(degree_output_polynomials=max_deg_output, maximal_potence=model_parameters_max['maximal_potence'])
    
    # Continue with checking if it is a rational function
    model_parameters = {'degree_input_polynomials': 0, 'degree_output_polynomials': max_deg_output, 'functions': [],
                        'function_names': [], 'width': 1, 'degree_input_denominator': 0,
                        'degree_output_denominator': max_deg_output_denominator,
                        'degree_output_polynomials_specific': None,
                        'degree_output_polynomials_denominator_specific': None,
                        'enforce_function': False, 'maximal_potence': model_parameters_max['maximal_potence']}
    model_parameter_list.append(model_parameters)
    # Try one function at a time while building up the polynomial degrees
    for i in range(len(functions)):
        for enforce_function in [False, True]:
            function = functions[i]
            function_name = function_names[i]
            deg_output_polynomials_specific = [max_deg_output_polynomials_specific[i]]
            deg_output_polynomials_denominator_specific = [
                max_deg_output_polynomials_denominator_specific[
                    i]]
            model_parameters = {'degree_input_polynomials': max_deg_input,
                                'degree_output_polynomials': max_deg_output,
                                'functions': [function], 'function_names': [function_name], 'width': 1,
                                'degree_input_denominator': max_deg_input_denominator,
                                'degree_output_denominator': max_deg_output_denominator,
                                'degree_output_polynomials_specific': deg_output_polynomials_specific,
                                'degree_output_polynomials_denominator_specific': deg_output_polynomials_denominator_specific,
                                'enforce_function': enforce_function,
                                'maximal_potence': model_parameters_max['maximal_potence']
                                }
            model_parameter_list.append(model_parameters)

    return model_parameter_list

def filter_predictions(predictions, function):
    mask = torch.zeros(predictions.shape[0], dtype=torch.bool)
    for i in range(predictions.shape[0]):
        if function == 'all':
            rational = (predictions[i,-3:] == 0).all()
        else:
            rational = predictions[i,-1] == 0
        active_input = predictions[i,0]>0 or predictions[i,1]>0 
        if (rational and not active_input) or (not rational and active_input):
            mask[i] = True
    return predictions[mask]

def get_enforce_function_iterate_list(enforce_function_iterate):
    if enforce_function_iterate == 'False': 
        enforce_function_iterate_list = [False]
    elif enforce_function_iterate == 'True':
        enforce_function_iterate_list = [True]
    elif enforce_function_iterate == 'Both':
        enforce_function_iterate_list = [False, True]
    else:
        raise NotImplementedError(f'enforce_function_iterate has to be "False", "True", or "Both" but not {enforce_function_iterate}')
    return enforce_function_iterate_list

def get_pretrained_model_parameter_list(model_parameters_max, path_nn, x, y, enforce_function_iterate, repetitions, k=10):
    path = os.path.join('trainingOnSyntheticData', 'results')
    config_path = Path(path_nn).parent
    print(os.path.join(path, config_path, 'train_model_parameters.ini'))
    config_nn = configparser.ConfigParser()  # Create a ConfigParser object
    config_nn.read(os.path.join(path, config_path, 'train_model_parameters.ini'))  # Read the configuration file
    assert config_nn['META']['objective'] == 'model_parameters'
    
    hyper_parameters_nn = dict(config_nn['TRAININGPARAMETERS'])
    hyper_parameters_nn = {key: ast.literal_eval(hyper_parameters_nn[key]) for key in hyper_parameters_nn.keys()}
    hyper_parameters_nn = {key: hyper_parameters_nn[key][0] if isinstance(hyper_parameters_nn[key], list) else hyper_parameters_nn[key]
                           for key in hyper_parameters_nn.keys()}
    
    data_parameters_nn = dict(config_nn['DATAPARAMETERS'])
    data_parameters_nn = {key: ast.literal_eval(data_parameters_nn[key]) for key in data_parameters_nn.keys()}  
    data_parameters_nn = {key: data_parameters_nn[key][0] if (isinstance(data_parameters_nn[key], list) and key!='function_names_str')
                          else data_parameters_nn[key] for key in data_parameters_nn.keys()}
    
    nn = torch.load(os.path.join(path, path_nn), map_location=x.device)
    nn.eval()
    
    # First choose a subset and then normalize, to align with the training process
    subset = np.random.choice(range(x.shape[0]), 200, replace=False)
    x = x[subset]
    y = y[subset]
    
    normalization = data_parameters_nn['normalization']
    if normalization == 'No':
        pass
    elif normalization == 'Full':
        y = F.normalize(y)
    elif normalization == 'DataSetWise':
        y = F.normalize(y, dim=1)
    elif normalization == 'DataSetWiseXY-11':
        x = min_max_normalization(x, dim=0, new_min=-1, new_max=1)  # normalize over the features
        y = min_max_normalization(y, dim=0, new_min=-1, new_max=1)
    
    data = torch.concat([x, y.unsqueeze(1)], dim=1)
    data = data.float().unsqueeze(0)
    data = torch.cat([torch.zeros(data.shape[0], data.shape[1], 10-data.shape[2], device=data.device, dtype=data.dtype), data], dim=-1)
    prediction = nn(data, return_logits=True)
    
    topk_predictions = get_topk_predictions(data_parameters_nn, prediction[0], k=k)
    topk_predictions = filter_predictions(topk_predictions, 'all')  
    
    enforce_function_iterate_list = get_enforce_function_iterate_list(enforce_function_iterate)
    
    standard_model_parameter_list = []
    # Start with checking if it is a polynomial
    standard_model_parameter_list += get_polynomial_model_parameters_list(degree_output_polynomials=model_parameters_max['max_deg_output'], maximal_potence=model_parameters_max['maximal_potence'])

    # Continue with checking if it is a rational function
    standard_model_parameter_list += get_rational_model_parameters_list(max_deg_output_denominator=model_parameters_max['max_deg_output_denominator'],
                                                max_deg_output_numerator=model_parameters_max['max_deg_output'], maximal_potence=model_parameters_max['maximal_potence'])
    
    predicted_model_parameter_list = []
    model_parameters_max = {'function_names': data_parameters_nn['function_names_str']} # needs only to contain the function names
    for i in range(topk_predictions.shape[0]):    
        model_parameters = tensor_to_model_parameters(topk_predictions[i], model_parameters_max)
        
        model_parameters['functions'] = model_parameters['function_names']
        model_parameters['degree_output_polynomials_specific'] = [1 for _ in model_parameters['function_names']]
        model_parameters['degree_output_polynomials_denominator_specific'] = [1 for _ in model_parameters['function_names']]    
        model_parameters['maximal_potence'] = data_parameters_nn['maximal_potence']
        for enforce_function in enforce_function_iterate_list:
            model_parameters['enforce_function'] = enforce_function  
            predicted_model_parameter_list.append(copy.deepcopy(model_parameters))
    predicted_model_parameter_list = predicted_model_parameter_list * repetitions
    
    return standard_model_parameter_list + predicted_model_parameter_list


def get_model_parameter_list(model_parameters_max, config, training_parameters, x, y):
    enforce_function_iterate = training_parameters['enforce_function_iterate']
    
    if config['META']['model_parameter_search'] == 'No':
        # One specific set of model parameters, which will be used instead of the hyperparameter search
        # model_parameters_fix = dict(config.items("MODELPARAMETERSFIX"))
        model_parameters_fix = config["MODELPARAMETERSFIX"]
        model_parameters_fix = {key: ast.literal_eval(model_parameters_fix[key]) for key in
                                model_parameters_fix.keys()}
        model_parameters_fix['functions'] = model_parameters_fix['function_names']
        model_parameter_list = [model_parameters_fix]
    elif config['META']['model_parameter_search'] == 'Complete':
        model_parameter_list = get_complete_model_parameter_list(model_parameters_max, training_parameters['model'], enforce_function_iterate)
    elif config['META']['model_parameter_search'] == 'Max':
        model_parameter_list = get_max_model_parameter_list(model_parameters_max)
    elif config['META']['model_parameter_search'] == 'pretrained':
        assert training_parameters['model'] == 'ParFamTorch'
        model_parameter_list = get_pretrained_model_parameter_list(model_parameters_max, training_parameters['path_pretrained'], x, y, enforce_function_iterate,
                                                                   k=training_parameters['topk_predictions'], repetitions=training_parameters['repetitions'])
    elif config['META']['model_parameter_search'] == 'pretrained+full':
        assert training_parameters['model'] == 'ParFamTorch'
        model_parameter_list_pretrained = get_pretrained_model_parameter_list(model_parameters_max, training_parameters['path_pretrained'], x, y, 
                                                                              enforce_function_iterate, repetitions=1,
                                                                              k=training_parameters['topk_predictions'])
        model_parameter_list_complete = get_complete_model_parameter_list(model_parameters_max, training_parameters['model'])
        model_parameter_list = model_parameter_list_pretrained + model_parameter_list_complete
    else:
        raise NotImplementedError(
            f"config['META']['model_parameter_search'] must be either 'No', 'Complete', 'Max', 'pretrained' or 'pretrained+full'"
            f"but not '{config['META']['model_parameter_search']}'")
    if config['META']['model_parameter_search'] == 'pretrained':
        # in th
        return model_parameter_list
    else:
        return model_parameter_list * training_parameters['repetitions']


def run_jobs_in_parallel_new_pool(x, y, target_expr, model_parameter_list, seed, training_parameters, accuracy):
    workers = min(mp.cpu_count(), len(model_parameter_list), training_parameters['n_processes'])
    logging.info(f"Available workers:{mp.cpu_count()} ")
    logging.info(f"Used workers:{workers}")
    # Create a multiprocessing pool
    pool = mp.Pool(processes=workers)
    # Combinations which will be performed in parallel

    with Manager() as manager:
        event = manager.Event()
        lock = manager.Lock()
        return_dict = manager.dict()
        return_dict['relative_l2_distance_train'] = None
        return_dict['relative_l2_distance_val'] = np.inf
        return_dict['r_squared_val'] = None
        return_dict['training_time'] = None
        return_dict['coefficients'] = None
        return_dict['model_parameters'] = None
        return_dict['counter'] = len(model_parameter_list)

        combinations = [
            (model_parameters, x, y, target_expr, training_parameters, seed, accuracy, event, return_dict, lock) for
            model_parameters in model_parameter_list]
        results = pool.starmap_async(train_model_old_event, combinations,
                                     error_callback=lambda error: logging.error(error))
        # Close the pool and wait for the processes to finish
        logging.info(f'Waiting for the event')
        event.wait()
        logging.info('Computations finished')
        pool.terminate()

        return return_dict['relative_l2_distance_train'], return_dict['relative_l2_distance_val'], return_dict[
            'r_squared_val'], return_dict['training_time'], return_dict['coefficients'], return_dict['model_parameters']


def run_jobs_in_parallel_old_pool(x, y, target_expr, model_parameter_list, seed, training_parameters, accuracy):
    workers = min(mp.cpu_count(), len(model_parameter_list))
    logging.info(f"Available workers:{mp.cpu_count()} ")
    logging.info(f"Used workers:{workers} ")
    # Create a multiprocessing pool
    pool = mp.Pool(processes=workers)
    # Combinations which will be performed in parallel
    combinations = [(model_parameters, x, y, target_expr, training_parameters, seed, accuracy) for model_parameters in
                    model_parameter_list]
    results = pool.starmap(train_model, combinations)
    # Close the pool and wait for the processes to finish
    pool.close()
    pool.join()

    # Check the results and stop all processes if necessary
    for result in results:
        if result is not None:
            relative_l2_distance_train, relative_l2_distance_val, formula, training_time = result
            # Save best values over parameter search
            if best_relative_l2_distance_val > relative_l2_distance_val:
                best_relative_l2_distance_train, best_relative_l2_distance_val = relative_l2_distance_train, relative_l2_distance_val
            best_formula, best_training_time = formula, training_time
            logging.info(
                f"New best distance (train,val): {best_relative_l2_distance_train, best_relative_l2_distance_val}")
            logging.info(f"Best formula: {best_formula} best training time: {best_training_time}")


def run_jobs_in_parallel_old_event(x, y, target_expr, model_parameter_list, seed, training_parameters, accuracy):
    manager = Manager()
    return_dict = manager.dict()

    return_dict['counter'] = len(model_parameter_list)
    return_dict['relative_l2_distance_train'] = np.inf
    return_dict['relative_l2_distance_val'] = np.inf
    return_dict['formula'] = None
    return_dict['training_time'] = None
    found_event = Event()
    lock = Lock()

    pool = [Process(target=train_model_old_event,
                    args=(
                        model_parameters, x, y, target_expr, training_parameters, seed, accuracy, found_event,
                        return_dict, lock))
            for model_parameters in model_parameter_list]

    for p in pool:
        p.start()

    found_event.wait()  # <- blocks until condition met
    print('{} | terminating processes'.format(datetime.now()))
    for p in pool:
        p.terminate()
    for p in pool:
        p.join()
    print('{} | all processes joined'.format(datetime.now()))
    print(return_dict)
    return return_dict['relative_l2_distance_train'], return_dict['relative_l2_distance_val'], return_dict[
        'formula'], return_dict['training_time']


def run_jobs_in_parallel(x, y, target_expr, model_parameter_list, seed, training_parameters, accuracy):
    manager = Manager()
    return_dict = manager.dict()

    return_dict['counter'] = len(model_parameter_list)
    return_dict['relative_l2_distance_train'] = np.inf
    return_dict['relative_l2_distance_val'] = np.inf
    return_dict['formula'] = None
    return_dict['training_time'] = None
    lock = Lock()

    pool = [Process(target=train_model,
                    args=(model_parameters, x, y, target_expr, training_parameters, seed, return_dict, lock))
            for model_parameters in model_parameter_list]

    started_pool = []
    n_processes = training_parameters['n_processes']
    for i in range(n_processes):
        pool[i].start()
        started_pool.append(pool[i])

    # found_event.wait()  # <- blocks until condition met

    finished = False
    while not finished:
        sleep(5)
        with lock:
            logging.info(f'Checking if we are ready:')
            logging.info(f'Relative l2 distance validation: {return_dict["relative_l2_distance_val"]}')
            logging.info(f'Accuracy: {accuracy}')
            logging.info(f'Counter: {return_dict["counter"]}')
            if return_dict['relative_l2_distance_val'] < accuracy:
                logging.info('Main process terminates the others, since the true equation has been found.')
                finished = True
            if return_dict['counter'] == 0:
                logging.info('Main process terminates the others, since all model parameters have been tried.')
                finished = True
            if not finished:
                alive_pool = [p for p in started_pool if p.is_alive()]
                start_new = min(n_processes - len(alive_pool), len(pool) - len(started_pool))
                logging.info(f'Number alive processes: {len(alive_pool)}')
                logging.info(f'New processes: {start_new}')
                logging.info(f'Number started processes: {len(started_pool)}')
                num_started_pools = len(started_pool)
                for i in range(start_new):
                    logging.info(f'i: {i}')
                    logging.info(f'num_started_pools + i: {num_started_pools + i}')
                    pool[num_started_pools + i].start()
                    started_pool.append(pool[num_started_pools + i])

    print('{} | terminating processes'.format(datetime.now()))
    for p in started_pool:
        p.terminate()
    for p in started_pool:
        p.join()
    print('{} | all processes joined'.format(datetime.now()))
    print(return_dict)
    return return_dict['relative_l2_distance_train'], return_dict['relative_l2_distance_val'], return_dict[
        'formula'], return_dict['training_time']


def run_jobs_in_parallel_test(x, y, target_expr, model_parameter_list, seed, training_parameters, accuracy):
    manager = Manager()
    return_dict = manager.dict()

    return_dict['counter'] = len(model_parameter_list)
    return_dict['relative_l2_distance_train'] = np.inf
    return_dict['relative_l2_distance_val'] = np.inf
    return_dict['formula'] = None
    return_dict['training_time'] = None
    lock = Lock()

    pool = [Process(target=train_model,
                    args=(model_parameters, x, y, target_expr, training_parameters, seed, return_dict, lock))
            for model_parameters in model_parameter_list]

    active_pool = []
    n_processes = training_parameters['n_processes']
    for i in range(n_processes):
        pool[i].start()
        active_pool.append(pool[i])
    num_started_processes = n_processes

    # found_event.wait()  # <- blocks until condition met

    finished = False
    while not finished:
        sleep(5)
        with lock:
            logging.info(f'Checking if we are ready:')
            logging.info(f'Relative l2 distance validation: {return_dict["relative_l2_distance_val"]}')
            logging.info(f'Accuracy: {accuracy}')
            logging.info(f'Counter: {return_dict["counter"]}')
            if return_dict['relative_l2_distance_val'] < accuracy:
                logging.info('Main process terminates the others, since the true equation has been found.')
                finished = True
            if return_dict['counter'] == 0:
                logging.info('Main process terminates the others, since all model parameters have been tried.')
                finished = True
            if not finished:
                dead_pool = [p for p in active_pool if not p.is_alive()]
                alive_pool = [p for p in active_pool if p.is_alive()]
                start_new = min(n_processes - len(alive_pool), len(pool) - len(active_pool))
                logging.info(f'Number alive processes: {len(alive_pool)}')
                logging.info(f'New processes: {start_new}')
                logging.info(f'Number started processes: {num_started_processes}')
                for i in range(start_new):
                    logging.info(f'i: {i}')
                    logging.info(f'num_started_pools + i: {num_started_processes + i}')
                    pool[num_started_processes + i].start()
                    active_pool.append(pool[num_started_processes + i])
                num_started_processes += start_new
                for p in dead_pool:
                    p.terminate()
                    p.join()

    print('{} | terminating processes'.format(datetime.now()))
    for p in alive_pool:
        p.terminate()
    for p in alive_pool:
        p.join()
    print('{} | all processes joined'.format(datetime.now()))
    print(return_dict)
    return return_dict['relative_l2_distance_train'], return_dict['relative_l2_distance_val'], return_dict[
        'formula'], return_dict['training_time']

def formula_selection(x_val, y_val, formulas, start_with_x0, target_noise, accuracy):
    # subsample the sets since uDSR only yields a sympy formula for prediction
    if len(y_val) > 1000:
        subset = np.random.choice(len(y_val), 1000, replace=False)
    else:
        subset = np.arange(len(y_val))
    y_val = y_val[subset]
    x_val = x_val[subset]

    best_formula = None
    best_test_error = 10000000
    for formula in formulas:
        try:
            formula = sympy.sympify(formula)
            y_pred = utils.evaluate_sympy_expression_nd(formula, x_val, start_with_x0=start_with_x0).detach().cpu().numpy()
            relative_l2_distance = np.linalg.norm(y_val - y_pred, ord=2) / np.linalg.norm(
                y_val - y_val.mean(), ord=2)

            if (relative_l2_distance < max(2 * target_noise, accuracy)) and len(str(formula)) < 10:
                best_formula = formula
                break

            if relative_l2_distance < best_test_error:
                best_formula = formula
                best_test_error = relative_l2_distance
        except Exception as e:
            logging.info(f'While evaluating the formula, following exception occurred {e}. '
                         f'Skip to the next formula worst case')
    return best_formula

def train_aifeynman(x, y, target_expr, training_parameters, accuracy, dataset_name, config, target_noise):
    dataset_name = dataset_name.replace('-', '_').replace('.', '_').replace('i', 'I')

    x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()
    aif_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "AIFeynman")
    sys.path.append(os.path.join(aif_path, "AI-Feynman/Code/"))
    from S_run_aifeynman import run_aifeynman
    temp_dir = f'temp_data_{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    t0 = time()
    # f"../../pmlb/datasets/{dataset_name}/"
    run_aifeynman(os.path.join(config['META']['path_pmlb'], 'datasets', dataset_name), f"{dataset_name}.tsv.gz", BF_try_time=training_parameters['bf_try_time'],
                  BF_ops_file_type=f"{training_parameters['bf_ops_file_type']}.txt", polyfit_deg=3,
                  NN_epochs=training_parameters['aif_nn_epochs'], test_percentage=20, max_dataset_length=training_parameters['max_dataset_length'],
                  result_dir=f"temp/{temp_dir}",
                  time_limit=training_parameters['time_limit'])

    hof = pd.read_csv(os.path.join('temp', temp_dir, f'solution_{dataset_name}'), delimiter=' ',
                      names=['a', 'b', 'c', 'd', 'e', 'expression'])['expression']

    logging.info(f'HOF consists out {len(hof)} formulas.')

    best_formula = formula_selection(x, y, hof, True, target_noise, accuracy)
    
    t1 = time()
    # os.remove(os.path.join(save_path, 'dso_temp_data_0_hof.csv'))
    # shutil.rmtree(os.path.join('temp', temp_dir))
    return best_formula, t1 - t0

def train_aifeynman_srbench(x, y, target_expr, training_parameters, accuracy, dataset_name, config, target_noise):

    from aifeynman import AIFeynmanRegressor
    
    max_dataset_length = training_parameters['max_dataset_length']
    if len(y) > max_dataset_length:
        subset = np.random.choice(len(y), max_dataset_length, replace=False)
    else:
        subset = np.arange(len(y))
    x = x[subset]
    y = y[subset].reshape(len(subset), 1)
        
    x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()

    t0 = time()

    est = AIFeynmanRegressor(
        BF_try_time=training_parameters['bf_try_time'],
        polyfit_deg=training_parameters['polyfit_deg'],
        NN_epochs=training_parameters['aif_nn_epochs'],
        # max_time=7*60*60
        max_time=training_parameters['time_limit']
        )
    
    est.fit(x, y)


    best_formula = sympy.simplify(est.best_model_)
    
    t1 = time()
    # os.remove(os.path.join(save_path, 'dso_temp_data_0_hof.csv'))
    # shutil.rmtree(os.path.join('temp', temp_dir))
    return best_formula, t1 - t0

def train_udsr(x, y, target_expr, training_parameters, accuracy, dataset_name, target_noise):
    x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()

    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, test_size=0.2)

    from dso import DeepSymbolicRegressor, DeepSymbolicOptimizer

    max_dataset_length = training_parameters['max_dataset_length']
    if len(y_train) > max_dataset_length:
        subset = np.random.choice(len(y_train), max_dataset_length, replace=False)
    else:
        subset = np.arange(len(y_train))
    x_train = x_train[subset]
    y_train = y_train[subset].reshape(len(subset), 1)

    data = np.concatenate([x_train, y_train], axis=1)
    data = pd.DataFrame(data)
    rand_int = random.randint(1, 1000000)
    data_file = f'temp_data_{dataset_name}-{rand_int}.csv'
    data.to_csv(data_file, index=False, header=False)

    t0 = time()

    # If dso_config includes gp_meld, the deap package might throw an error. In that case
    # one has to add "from inspect import isclass" at the beginning of the file with the 
    # error in the deap site package in the environment!
    dso_config = {
        "task": {
            # Deep Symbolic Regression
            "task_type": "regression",
            # This can either be (1) the name of the benchmark dataset (see
            # benchmarks.csv for a list of supported benchmarks) or (2) a path to a
            # CSV file containing the data.
            "dataset": data_file,  # "Nguyen-1",
            # To customize a function set, edit this! See functions.py for a list of
            # supported functions. Note "const" will add placeholder constants that
            # will be optimized within the training loop. This will considerably
            # increase runtime.
            "function_set": ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "const", "poly"],
            # Metric to be used for the reward function. See regression.py for
            # supported metrics.
            "metric": "inv_nrmse",
            "metric_params": [1.0],
            # Optional alternate metric to be used at evaluation time.
            "extra_metric_test": None,
            "extra_metric_test_params": [],
            # NRMSE threshold for early stopping. This is useful for noiseless
            # benchmark problems when DSO discovers the True solution.
            "threshold": training_parameters['accuracy_udsr'],
            # With protected=False, floating-point errors (e.g. log of negative
            # number) will simply returns a minimal reward. With protected=True,
            # "protected" functions will prevent floating-point errors, but may
            # introduce discontinuities in the learned functions.      
            "protected": False,
            # You can add artificial reward noise directly to the reward function.
            # Note this does NOT add noise to the dataset.
            "reward_noise": 0.0,
            "reward_noise_type": "r",
            "normalize_variance": False,
            # Set of thresholds (shared by all input variables) for building
            # decision trees. Note that no StateChecker will be added to Library
            # if decision_tree_threshold_set is an empty list or None.
            "decision_tree_threshold_set": [],
            # Parameters for optimizing the "poly" token.
            # Note: poly_optimizer is turned on if and only if "poly" is in function_set.
            "poly_optimizer_params": {
                # The (maximal) degree of the polynomials used to fit the data
                "degree": 3,
                # Cutoff value for the coefficients of polynomials. Coefficients
                # with magnitude less than this value will be regarded as 0.
                "coef_tol": 1e-6,
                # linear models from sklearn: linear_regression, lasso,
                # and ridge are currently supported, or our own implementation
                # of least squares regressor "dso_least_squares".
                "regressor": "dso_least_squares",
                "regressor_params": {
                    # Cutoff value for p-value of coefficients. Coefficients with 
                    # larger p-values are forced to zero.
                    "cutoff_p_value": 1.0,
                    # Maximum number of terms in the polynomial. If more coefficients are nonzero,
                    # coefficients with larger p-values will be forced to zero.
                    "n_max_terms": None,
                    # Cutoff value for the coefficients of polynomials. Coefficients
                    # with magnitude less than this value will be regarded as 0.
                    "coef_tol": 1e-6
                }
            }
        },
        # Hyperparameters related to genetic programming hybrid methods.
        "gp_meld" : {
            "run_gp_meld" : True,
            "population_size" : 100,
            "generations" : 20,
            "crossover_operator" : "cxOnePoint",
            "p_crossover" : 0.5,
            "mutation_operator" : "multi_mutate",
            "p_mutate" : 0.5,   
            "tournament_size" : 5,
            "train_n" : 50,
            "mutate_tree_max" : 3,
            "verbose" : False,
            # Speeds up processing when doing expensive evaluations.
            "parallel_eval" : True
        },
        # Only the key training hyperparameters are listed here. See
        # config_common.json for the full list.
        "training": {
            "n_samples": training_parameters["iterations_udsr"],
            "batch_size": 500,
            "epsilon": 0.02,
            # Recommended to set this to as many cores as you can use! Especially if
            # using the "const" token.
            "n_cores_batch": 1# -1
        },
        # # # Only the key Policy Optimizer hyperparameters are listed here. See
        # # # config_common.json for the full list.
        # "policy_optimizer": {
        #     "learning_rate": 0.0005,
        #     "entropy_weight": 0.03,
        #     "entropy_gamma": 0.7,
        #     # EXPERIMENTAL: Proximal policy optimization hyperparameters.
        #     # "policy_optimizer_type" : "ppo",
        #     # "ppo_clip_ratio" : 0.2,
        #     # "ppo_n_iters" : 10,
        #     # "ppo_n_mb" : 4,
        #     # EXPERIMENTAL: Priority queue training hyperparameters.
        #     # "policy_optimizer_type" : "pqt",
        #     # "pqt_k" : 10,
        #     # "pqt_batch_size" : 1,
        #     # "pqt_weight" : 200.0,
        #     # "pqt_use_pg" : False
        # },
        
        # // Only the key RNN controller hyperparameters are listed here. See
        # // config_common.json for the full list.
        "controller" : {
            "learning_rate": 0.0025,
            "entropy_weight" : 0.03,
            "entropy_gamma" : 0.7,
            # // EXPERIMENTAL: Priority queue training hyperparameters.
            "pqt" : True,
            "pqt_k" : 10,
            "pqt_batch_size" : 1,
            "pqt_weight" : 200.0,
            "pqt_use_pg" : False
        },
        # Hyperparameters related to including in situ priors and constraints. Each
        # prior must explicitly be turned "on" or it will not be used. See
        # config_common.json for descriptions of each prior.
        "prior": {
            "length" : {
                "min_" : 4,
                "max_" : 100,
                "on" : True
            },
            "inverse" : {
                "on" : True
            },
            "trig" : {
                "on" : True
            },
            "const" : {
                "on" : True
            },
            "no_inputs" : {
                "on" : True
            },
            "uniform_arity" : {
                "on" : True
            },
            "soft_length" : {
                "loc" : 10,
                "scale" : 5,
                "on" : True
            },
            "domain_range" : {
                "on" : True
            }
        }
        }

    model = DeepSymbolicOptimizer(dso_config)

    # Change the model.train() method in dso/core.py

    model.train(training_parameters['time_limit'])

    # try:
    #     with time_limit(training_parameters['time_limit']):
    #         # Fit the model
    #         model.train()
    # except TimeoutException as e:
    #     logging.info("Timed out!")

    # # View the best expression
    # print(model.program_.pretty())
    t1 = time()
    training_time = t1 - t0
    os.remove(data_file)

    save_path = model.save_path
    pf = pd.read_csv(os.path.join(save_path, f'dso_{data_file[:-4]}_0_pf.csv'))['expression']

    best_formula = formula_selection(x, y, pf, False, target_noise, accuracy)

    # os.remove(os.path.join(save_path, 'dso_temp_data_0_hof.csv'))
    # shutil.rmtree(save_path)
    return best_formula, training_time


def train_endtoend(x, y, target_expr, training_parameters, accuracy):
    logging.info(f'{utils.using("Before starting the training: ")}')
    
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'symbolicregression'))
    import symbolicregression
    
    import requests
    t0 = time()
    
    model_path = "model_endtoend.pt" 
    try:
        if not os.path.isfile(model_path): 
            url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)
        model = torch.load(model_path, map_location=torch.device('cpu'))
        print(model.device)
        print("Model successfully loaded!")

    except Exception as e:
        print("ERROR: model not loaded! path was: {}".format(model_path))
        raise e
    
    logging.info(f'{utils.using("After loading the model: ")}')
    
    
    est = symbolicregression.model.SymbolicTransformerRegressor(
                    model=model,
                    max_input_points=training_parameters['max_input_points'], # 200,
                    max_number_bags=training_parameters['max_number_bags'], # 100,
                    n_trees_to_refine=training_parameters['n_trees_to_refine'], # 10,
                    rescale=True,
                    logging=logging
                    )
    
    max_dataset_length = training_parameters['max_dataset_length']
    if len(y) > max_dataset_length:
        subset = np.random.choice(len(y), max_dataset_length, replace=False)
    else:
        subset = np.arange(len(y))
    x = x[subset]
    y = y[subset]
    
    est.fit(x, y)
    
    t1 = time()
    
    logging.info(f'{utils.using("After finishing the training: ")}')
    
    
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
    model_str = est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()
    for op,replace_op in replace_ops.items():
        model_str = model_str.replace(op,replace_op)
    model_str = sympy.parse_expr(model_str)
    
    logging.info(f'{utils.using("After computing the formula: ")}')
    
    
    return est, model_str, t1 - t0


def train_nesymres(x, y, target_expr, training_parameters, accuracy):
        
    if x.shape[1] > 3:
        return sympy.sympify(0), -1
    
    from nesymres.architectures.model import Model
    from nesymres.utils import load_metadata_hdf5
    from nesymres.dclasses import FitParams, NNEquation, BFGSParams
    from pathlib import Path
    from functools import partial
    import json
    import omegaconf
    
    with open('nesymres/eq_setting.json', 'r') as json_file:
        eq_setting = json.load(json_file)
        
    cfg = omegaconf.OmegaConf.load("nesymres/config.yaml")
    
    ## Set up BFGS load rom the hydra config yaml
    bfgs = BFGSParams(
            activated= cfg.inference.bfgs.activated,
            n_restarts=cfg.inference.bfgs.n_restarts,
            add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
            normalization_o=cfg.inference.bfgs.normalization_o,
            idx_remove=cfg.inference.bfgs.idx_remove,
            normalization_type=cfg.inference.bfgs.normalization_type,
            stop_time=cfg.inference.bfgs.stop_time,
        )
    
    params_fit = FitParams(word2id=eq_setting["word2id"], 
                            id2word={int(k): v for k,v in eq_setting["id2word"].items()}, 
                            una_ops=eq_setting["una_ops"], 
                            bin_ops=eq_setting["bin_ops"], 
                            total_variables=list(eq_setting["total_variables"]),  
                            total_coefficients=list(eq_setting["total_coefficients"]),
                            rewrite_functions=list(eq_setting["rewrite_functions"]),
                            bfgs=bfgs,
                            beam_size=cfg.inference.beam_size #This parameter is a tradeoff between accuracy and fitting time
                            )
    
    weights_path = 'nesymres/100M.ckpt'
    
    ## Load architecture, set into eval mode, and pass the config parameters
    model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
    model.eval()
    if torch.cuda.is_available(): 
        model.cuda()
    
    fitfunc = partial(model.fitfunc,cfg_params=params_fit)
    
    max_dataset_length = 500  # Hardcoded because this is the number NeSymRes was trained on
    
    if len(y) > max_dataset_length:
        subset = np.random.choice(len(y), max_dataset_length, replace=False)
    else:
        subset = np.arange(len(y))
    x = x[subset]
    y = y[subset]
        
    t0 = time()
    
    output = fitfunc(x,y) 
    
    t1 = time()
    
    formula_str = output['best_bfgs_preds'][0]
    
    for i in range(x.shape[1]):
        formula_str = formula_str.replace(f'x_{i+1}', f'x{i}')
    
    return sympy.sympify(formula_str), t1 - t0

def train_pysr(x, y, target_expr, training_parameters, accuracy):
    from pysr import PySRRegressor

    max_dataset_length = training_parameters['max_dataset_length']
    if len(y) > max_dataset_length:
        subset = np.random.choice(len(y), max_dataset_length, replace=False)
    else:
        subset = np.arange(len(y))
    x = x[subset]
    y = y[subset]

    t0 = time()

    model = PySRRegressor(
        procs=training_parameters['procs'],
        populations=training_parameters['populations'],
        # ^ 2 populations per core, so one is always running.
        population_size=training_parameters['population_size'],
        ncycles_per_iteration=training_parameters['ncycles_per_iteration'],
        # ^ Generations between migrations.
        niterations=training_parameters['evaluations_limit'],  # Run forever
        early_stop_condition=(
            "stop_if(loss, complexity) = loss < 1e-5 && complexity < 10"
            # Stop early if we find a good and simple equation
        ),
        timeout_in_seconds=training_parameters['time_limit'],
        # ^ Alternatively, stop after 24 hours have passed.
        maxsize=training_parameters['max_size'],
        # ^ Allow greater complexity.
        maxdepth=training_parameters['max_depth'],
        # ^ But, avoid deep nesting.
        binary_operators=["*", "+", "-", "/"],
        unary_operators=["square", "cube", "exp", "sqrt", "sin"],
        constraints={
            "/": (-1, 9),
            "square": 9,
            "cube": 9,
            "exp": 9,
            "sin": 9,
            "sqrt": 9,
        },
        # ^ Limit the complexity within each argument.
        # "inv": (-1, 9) states that the numerator has no constraint,
        # but the denominator has a max complexity of 9.
        # "exp": 9 simply states that `exp` can only have
        # an expression of complexity 9 as input.
        nested_constraints={
            "square": {"square": 1, "cube": 1, "exp": 0, "sqrt": 0, "sin": 1},
            "cube": {"square": 1, "cube": 1, "exp": 0, "sqrt": 1, "sin": 1},
            "exp": {"square": 1, "cube": 1, "exp": 0, "sqrt": 1, "sin": 0},
            "sin": {"square": 1, "cube": 1, "exp": 0, "sqrt": 1, "sin": 0},
            "sqrt": {"square": 1, "cube": 1, "exp": 0, "sqrt": 1, "sin": 1},
        },
        # ^ Nesting constraints on operators. For example,
        # "square(exp(x))" is not allowed, since "square": {"exp": 0}.
        complexity_of_operators={"/": 1, "exp": 1},
        # ^ Custom complexity of particular operators.
        complexity_of_constants=1,
        # ^ Punish constants more than variables
        select_k_features=9,
        # ^ Train on only the 4 most important features
        progress=False,
        # ^ Can set to false if printing to a file.
        weight_randomize=0.1,
        # ^ Randomize the tree much more frequently
        cluster_manager=None,
        # ^ Can be set to, e.g., "slurm", to run a slurm
        # cluster. Just launch one script from the head node.
        precision=32,
        # ^ Higher precision calculations.
        warm_start=True,
        # ^ Start from where left off.
        turbo=True,
        # ^ Faster evaluation (experimental)
        julia_project=None,
        # ^ Can set to the path of a folder containing the
        # "SymbolicRegression.jl" repo, for custom modifications.
        update=False,
        # ^ Don't update Julia packages
        # extra_torch_mappings={sympy.cos: torch.cos},
        # ^ Not needed as cos already defined, but this
        # is how you define custom torch operators.
        # extra_jax_mappings={sympy.cos: "jnp.cos"},
        # ^ For JAX, one passes a string.
    )

    model.fit(x, y)
    t1 = time()
    training_time = t1 - t0

    return model, training_time


def run_jobs_sequentially(x, y, target_expr, model_parameter_list, seed, training_parameters, accuracy, custom_loss=None):
    best_relative_l2_distance_train, best_relative_l2_distance_val, best_formula, best_training_time = None, np.inf, None, None
    best_r_squared_val, best_coefficients, best_model_parameters = None, None, None
    cumulative_n_evaluations = 0
    cumulative_training_time = 0
    time_limit = training_parameters['time_limit']
    evaluations_limit = training_parameters['evaluations_limit']
    for model_parameters in model_parameter_list:
        t_0 = time()
        logging.info(f'Model parameters: {model_parameters}')
        relative_l2_distance_train, relative_l2_distance_val, r_squared_val, training_time, coefficients, n_evaluations = \
            hyperparameter_training(x, y, target_expr, training_parameters, copy.deepcopy(model_parameters), seed,
                                    verbose=True, custom_loss=custom_loss)
        t_1 = time()
        if cumulative_training_time + t_1 - t_0 > time_limit:
            logging.info(f"Terminate proccess, the time limit of {time_limit}s was reached")
            print(f"Terminate proccess, the time limit of {time_limit}s was reached")
            break

        if cumulative_n_evaluations + n_evaluations > evaluations_limit:
            logging.info(f"Terminate proccess, the limit of {evaluations_limit} evaulations  was reached")
            print(f"Terminate proccess, the limit of {evaluations_limit} evaulations  was reached")
            break

        cumulative_n_evaluations += n_evaluations
        cumulative_training_time += t_1 - t_0

        if custom_loss is None:
            logging.info(f'Relative l2 distance train: {relative_l2_distance_train}')
            logging.info(f'Relative l2 distance validation: {relative_l2_distance_val}')
        else:
            logging.info(f'Custom loss train: {relative_l2_distance_train}')
            logging.info(f'Custom loss val: {relative_l2_distance_val}')
            
        logging.info(f'Training time: {training_time}')
        logging.info(f'Cumulative training time: {cumulative_training_time}')
        logging.info(f'Cumulative number of evaluations: {cumulative_n_evaluations}')

        if relative_l2_distance_val < best_relative_l2_distance_val:
            if custom_loss is None:        
                logging.info(f"New best relative l2 distance validation: {relative_l2_distance_val}")
            else:
                logging.info(f"New best custom loss validation: {relative_l2_distance_val}")
                
            best_relative_l2_distance_train = relative_l2_distance_train
            best_relative_l2_distance_val = relative_l2_distance_val
            best_training_time = training_time
            best_coefficients = coefficients
            best_model_parameters = model_parameters
            best_r_squared_val = r_squared_val

            reached_accuracy = relative_l2_distance_val < accuracy
            simple_formula = sum(torch.abs(coefficients) > 0.01) < training_parameters['max_n_active_parameters']
            if reached_accuracy and simple_formula:
                logging.info(f"Terminate proccess, wanted accuracy {accuracy} and number of active parameters "
                             f"{sum(torch.abs(coefficients) > 0.01)} / {training_parameters['max_n_active_parameters']}"
                             f" was reached")
                print(f"Terminate proccess, wanted accuracy {accuracy} and number of active parameters "
                      f"{sum(torch.abs(coefficients) > 0.01)} / {training_parameters['max_n_active_parameters']} was "
                      f"reached")
                break

    return best_relative_l2_distance_train, best_relative_l2_distance_val, best_r_squared_val, best_training_time, \
        best_coefficients, best_model_parameters, cumulative_n_evaluations, cumulative_training_time


def setup_model(model_parameters, n_input, device, model):
    model_parameters = make_unpickable(model_parameters)
    # Set up model
    if model == 'ParFamTorch':
        model = ParFamTorch(n_input=n_input,
                            degree_input_numerator=model_parameters['degree_input_polynomials'],
                            degree_output_numerator=model_parameters['degree_output_polynomials'],
                            degree_input_denominator=model_parameters['degree_input_denominator'],
                            degree_output_denominator=model_parameters['degree_output_denominator'],
                            degree_output_numerator_specific=model_parameters['degree_output_polynomials_specific'],
                            degree_output_denominator_specific=model_parameters[
                                'degree_output_polynomials_denominator_specific'],
                            width=1, maximal_potence=model_parameters['maximal_potence'],
                            functions=model_parameters['functions'], device=device,
                            function_names=model_parameters['function_names'],
                            enforce_function=model_parameters['enforce_function'])
    else:
        model = ReParFam(n_input=n_input,
                         degree_input_numerator=model_parameters['degree_input_polynomials'],
                         degree_output_numerator=model_parameters['degree_output_polynomials'],
                         degree_input_denominator=model_parameters['degree_input_denominator'],
                         degree_output_denominator=model_parameters['degree_output_denominator'],
                         degree_output_numerator_specific=model_parameters['degree_output_polynomials_specific'],
                         degree_output_denominator_specific=model_parameters[
                             'degree_output_polynomials_denominator_specific'],
                         width=1, maximal_potence=model_parameters['maximal_potence'],
                         functions=model_parameters['functions'], device=device,
                         function_names=model_parameters['function_names'],
                         enforce_function=model_parameters['enforce_function'])
    return model


def get_formula(coefficients, model_parameters, n_input, device, decimals):
    model = setup_model(copy.deepcopy(model_parameters), n_input, device)
    return model.get_formula(torch.tensor(coefficients, device=device), decimals=decimals, verbose=False)

def finetune_coeffs(x_train, y_train, coefficients, model, cutoff, lambda_1, max_dataset_length=5000, custom_loss=None):
    # finetuning the reduced coefficients after the fitting if the user is still unhappy with the complexity of the formula
    
    n_active_coefficients = sum(torch.abs(coefficients) > 0)
    mask = torch.abs(coefficients) > cutoff
    coefficients = copy.deepcopy(coefficients)
    coefficients[~ mask] = 0
    n_active_coefficients_new = sum(mask)
    if (n_active_coefficients_new > 0) and (n_active_coefficients_new < n_active_coefficients):
        # Run this routine only in the case that
        # 1. The number of active coefficients would be still greater than 0
        # 2. Sum new parameters would have been set to 0
        # Otherwise, this routine is not helpful.
        if len(y_train) > max_dataset_length:
            subset = np.random.choice(len(y_train), max_dataset_length, replace=False)
        else:
            subset = np.arange(len(y_train))
        x_train = x_train[subset]
        y_train = y_train[subset]

        n_params = model.get_number_parameters()
        evaluator = Evaluator(x_train, y_train, lambda_0=0, lambda_1=lambda_1, model=model, mask=mask,
                                n_params=n_params, custom_loss=custom_loss)

        coefficients_ = train_lbfgs(evaluator, n_active_coefficients_new, maxiter=1,
                                    coefficients=coefficients[mask])
        if not coefficients_ is None:
            coefficients_.requires_grad = False
            if coefficients.dtype != coefficients_.dtype:
                coefficients_.type(coefficients.dtype)
                logging.warning(f'The type of coefficients {coefficients} (dtype: {coefficients.dtype}) '
                                f'and coefficients_ {coefficients_} (dtype: {coefficients_.dtype}) '
                                f'were not the same and we had to cast them.')
            coefficients[mask] = coefficients_
            return coefficients
        else:
            logging.info(f'Coefficients_ while finetuning are None. Mask: {mask}')
            return None
    else:
        if (n_active_coefficients_new == 0):
            logging.info(f'Further finetuning would set all coefficients to 0.')
        else:
            logging.info(f'Finetuning with this cut off would not change anything.')
        return None


def evaluate_test_iterative_finetuning(best_model_parameters, x_train, y_train, x_test, y_test, device,
                                       best_coefficients, retrain, lambda_1, model_str, custom_loss=None):
    model = setup_model(copy.deepcopy(best_model_parameters), x_test.shape[1], device, model_str)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2)
    if custom_loss is None:
        relative_l2_distance_val = relative_l2_distance(model, best_coefficients, x_val, y_val)
    else:
        y_pred_val = model.predict(best_coefficients, x_val)
        custom_loss_val = custom_loss(y_pred_val, y_val, None)                
        relative_l2_distance_val = custom_loss_val
    n_active_coefficients = sum(torch.abs(best_coefficients) > 0)
    finetune = True
    update = True
    if finetune:
        if model_str == 'ParFamTorch':
            iterations = range(5, 1, -1)
        else:
            iterations = range(9, 1, -2)
        for i in iterations:
            # iterate through the cutoffs 10**(-5) to 10**(-2)
            for j in range(2):
                if j == 1 and not update:
                    # If we didn't change anything for this level of cutoff, then we don't have to restart it again
                    continue
                # perform each cutoff + optimization twice, since this often helps to reduce the number
                # of coefficients further
                coefficients = finetune_coeffs(x_train, y_train, best_coefficients, model, cutoff=10 ** (-i), lambda_1=lambda_1, max_dataset_length=5000, 
                                               custom_loss=custom_loss)
                if coefficients is None:
                    continue
                
                if custom_loss is None:
                    relative_l2_distance_val_new = relative_l2_distance(model, coefficients, x_val, y_val)
                else:
                    y_pred_val = model.predict(best_coefficients, x_val)
                    custom_loss_val = custom_loss(y_pred_val, y_val, None)                
                    relative_l2_distance_val_new = custom_loss_val
                # Check if the formula found now is worth keeping or not
                n_active_coefficients_new = sum(torch.abs(coefficients) > 0)
                better_performance = relative_l2_distance_val_new < relative_l2_distance_val
                slightly_worse_performance = relative_l2_distance_val_new < relative_l2_distance_val * 1.5
                considerably_less_coefficients = n_active_coefficients_new < n_active_coefficients * 0.75
                update = better_performance or (slightly_worse_performance and considerably_less_coefficients)
                if update:
                    relative_l2_distance_val = relative_l2_distance_val_new
                    best_coefficients = coefficients
                    n_active_coefficients = n_active_coefficients_new

    best_coefficients = model.get_normalized_coefficients(best_coefficients)
    
    if custom_loss is None:
        relative_l2_distance_test = relative_l2_distance(model, best_coefficients, x_test, y_test)
        r_squared_test = r_squared(model, best_coefficients, x_test, y_test)
        r_squared_val = r_squared(model, best_coefficients, x_val, y_val)
    else:
        y_pred_test = model.predict(best_coefficients, x_test)
        custom_loss_test = custom_loss(y_pred_test, y_test, None)
        
        y_pred_val = model.predict(best_coefficients, x_val)
        custom_loss_val = custom_loss(y_pred_val, y_val, None) 

        relative_l2_distance_test, r_squared_test, r_squared_val = custom_loss_test, custom_loss_val, custom_loss_test
        
    if model_str == 'ParFamTorch':
        formula = model.get_formula(best_coefficients, decimals=3, verbose=False)
    else:
        for i in range(3, 10):
            formula = model.get_formula(best_coefficients, decimals=i, verbose=False)
            if str(formula) != '0':
                break
    return n_active_coefficients, relative_l2_distance_test, r_squared_test, formula, r_squared_val, best_coefficients


def evaluate_test(best_model_parameters, x_train, y_train, x_test, y_test, device, best_coefficients, cutoff, retrain,
                  model_str, custom_loss=None):
    model = setup_model(copy.deepcopy(best_model_parameters), x_test.shape[1], device, model_str)
    best_coefficients = model.get_normalized_coefficients(best_coefficients)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2)
    mask = torch.abs(best_coefficients) > cutoff
    best_coefficients[~ mask] = 0
    n_active_coefficients = sum(mask)
    if retrain and n_active_coefficients > 0:
        if len(y_train) > 5000:
            subset = np.random.choice(len(y_train), 5000, replace=False)
        else:
            subset = np.arange(len(y_train))
        x_train = x_train[subset]
        y_train = y_train[subset]

        n_params = model.get_number_parameters()
        evaluator = Evaluator(x_train, y_train, lambda_0=0, lambda_1=0.00001, model=model, mask=mask,
                              n_params=n_params)

        coefficients_ = train_lbfgs(evaluator, n_active_coefficients, maxiter=1,
                                    coefficients=best_coefficients[mask])
        if not coefficients_ is None:
            best_coefficients[mask] = coefficients_
        else:
            logging.info(f'Coefficients_ while finetuning are None. Mask: {mask}')
    
    if custom_loss is None:
        relative_l2_distance_test = relative_l2_distance(model, best_coefficients, x_test, y_test)
        r_squared_test = r_squared(model, best_coefficients, x_test, y_test)
        r_squared_val = r_squared(model, best_coefficients, x_val, y_val)
    else:
        y_pred_test = model.predict(best_coefficients, x_test)
        custom_loss_test = custom_loss(y_pred_test, y_test, None)
        relative_l2_distance_test, r_squared_test, r_squared_val = custom_loss_test, custom_loss_test, custom_loss_test
    if model_str == 'ParFamTorch':
        formula = model.get_formula(best_coefficients, decimals=3, verbose=False)
    else:
        for i in range(3, 10):
            formula = model.get_formula(best_coefficients, decimals=i, verbose=False)
            if str(formula) != '0':
                break
    return n_active_coefficients, relative_l2_distance_test, r_squared_test, formula, r_squared_val

def min_max_normalization(data, dim, new_min, new_max):
    data_min = data.min(dim=dim)[0].unsqueeze(dim)
    data_max = data.max(dim=dim)[0].unsqueeze(dim)

    data = (data - data_min) / (data_max - data_min) * (new_max - new_min) + new_min

    # avoiding nan for constant functions and instead set it to zero
    constant = data_min == data_max
    data[constant.squeeze(dim).all(-1)] = 0.0

    return data

def model_parameter_search(x, y, target_expr, model_parameters_max, training_parameters, accuracy, config,
                           logging_to_file=True, model_parameters_perfect=None, dataset_name=None, custom_loss=None):
    seed = training_parameters['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    # prepare the data
    x, y = torch.tensor(x, device=training_parameters['device']), torch.tensor(y, device=training_parameters['device'])
    if config['META']['separate_test_set'] == 'True':
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25)
    else:
        # Relevant in special cases when we are not interested in test set performance, but only in the symbolic
        # recovery rate (synthetic data)
        x_train, x_test, y_train, y_test = x, x, y, y
    x_train, y_train = add_noise(x_train, y_train, training_parameters["target_noise"],
                                 training_parameters["feature_noise"])
    if training_parameters['normalization']:
        # normalization
        # scaler_x = MinMaxScaler(feature_range=(1,10))
        # scaler_x.fit(x_train)
        # x_train = torch.tensor(scaler_x.transform(x_train), device=training_parameters['device'])
        # x_test = torch.tensor(scaler_x.transform(x_test), device=training_parameters['device'])

        # scaler_y = MinMaxScaler(feature_range=(0.26,29.5))
        # scaler_y.fit(y_train.reshape(-1, 1))
        # y_train = torch.tensor(scaler_y.transform(y_train.reshape(-1, 1)), device=training_parameters['device']).squeeze(-1)
        # y_test = torch.tensor(scaler_y.transform(y_test.reshape(-1, 1)), device=training_parameters['device']).squeeze(-1)

        x_max = x_train.abs().max(axis=0)[0]
        y_max = y_train.abs().max(axis=0)[0]

        x_train = x_train / x_max * 10
        x_test = x_test / x_max * 10
        y_train = y_train / y_max * 10
        y_test = y_test / y_max * 10

    # Set logging
    if logging_to_file:
        directory = 'trained_models'
        logging.basicConfig(filename=os.path.join(directory, 'experiment.log'), level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    if training_parameters['model'] in ['pysr', 'udsr', 'aifeynman', 'aifeynman-srbench', 'endtoend', 'nesymres']:
        if training_parameters['model'] == 'pysr':
            model, training_time = train_pysr(x_train, y_train, target_expr, training_parameters, accuracy)
            best_formula = model.sympy()
            relative_l2_distance_train, relative_l2_distance_test, r_squared_test = evaluate(model, coefficients=None,
                                                                                             x_train=x_train,
                                                                                             y_train=y_train,
                                                                                             x_val=x_test, y_val=y_test,
                                                                                             target_expr=target_expr)
            print(f'Best formula: {best_formula}')
        elif training_parameters['model'] == 'endtoend':
            x_train = x_train.numpy()
            x_test = x_test.numpy()
            y_train = y_train.numpy()
            y_test = y_test.numpy()
            model, best_formula, training_time = train_endtoend(x_train, y_train, target_expr, training_parameters, accuracy)
            relative_l2_distance_train, relative_l2_distance_test, r_squared_test = evaluate(model, coefficients=None,
                                                                                             x_train=x_train,
                                                                                             y_train=y_train,
                                                                                             x_val=x_test, y_val=y_test,
                                                                                             target_expr=target_expr)
            print(f'Best formula: {best_formula}')
        else:
            if training_parameters['model'] == 'aifeynman':
                best_formula, training_time = train_aifeynman(x_train, y_train, target_expr, training_parameters,
                                                              accuracy, dataset_name, config, training_parameters["target_noise"])
            elif training_parameters['model'] == 'aifeynman-srbench':
                best_formula, training_time = train_aifeynman_srbench(x_train, y_train, target_expr, training_parameters,
                                                                      accuracy, dataset_name, config, training_parameters["target_noise"])
            elif training_parameters['model'] == 'nesymres':
                best_formula, training_time = train_nesymres(x_train, y_train, target_expr, training_parameters,
                                                        accuracy)
            elif training_parameters['model'] == 'udsr':
                # dsr needs numpy arrays
                best_formula, training_time = train_udsr(x_train, y_train, target_expr, training_parameters, accuracy,
                                                         dataset_name, training_parameters["target_noise"])
            print(f'Best formula: {best_formula}')

            # subsample the sets since uDSR only yields a sympy formula for prediction
            if len(y_train) > 1000:
                subset_train = np.random.choice(len(y_train), 1000, replace=False)
            else:
                subset_train = np.arange(len(y_train))

            if len(y_test) > 1000:
                subset_test = np.random.choice(len(y_test), 1000, replace=False)
            else:
                subset_test = np.arange(len(y_test))
            try:
                start_with_x0 = training_parameters['model'] in ['aifeynman', 'aifeynman-srbench', 'nesymres']
                y_train_pred = utils.evaluate_sympy_expression_nd(best_formula, x_train[subset_train], start_with_x0)
                y_test_pred = utils.evaluate_sympy_expression_nd(best_formula, x_test[subset_test], start_with_x0)

                y_train = y_train[subset_train]
                y_test = y_test[subset_test]

                relative_l2_distance_train = torch.norm(y_train - y_train_pred, p=2) / torch.norm(
                    y_train - y_train.mean(), p=2)
                relative_l2_distance_test = torch.norm(y_test - y_test_pred, p=2) / torch.norm(y_test - y_test.mean(),
                                                                                               p=2)

                r_squared_test = 1 - sum((y_test - y_test_pred) ** 2) / sum((y_test - y_test.mean()) ** 2)

            except Exception as e:
                print(f'Formula: {best_formula}')
                raise e
                logging.info(f'While evaluating the formula, following exception occurred {e}. Set all metrics to the '
                             f'worst case')
                relative_l2_distance_train = torch.tensor([1])
                relative_l2_distance_test = torch.tensor([1])
                r_squared_test = torch.tensor([0])

        # unneccessary variables for pysr; choose some defaults
        n_active_coefficients = torch.tensor([0])
        relative_l2_distance_val = relative_l2_distance_train
        r_squared_val = r_squared_test
        n_active_coefficients_reduced = torch.tensor([0])
        relative_l2_distance_test_reduced = relative_l2_distance_test
        r_squared_test_reduced = r_squared_test
        best_formula_reduced = best_formula
        r_squared_val_reduced = r_squared_val
        n_evaluations = training_parameters['evaluations_limit']
        best_model_parameters = None
        best_coefficients = None
        best_coefficients_reduced = None
        return relative_l2_distance_train, relative_l2_distance_val, r_squared_val, best_formula, training_time, \
            n_active_coefficients, relative_l2_distance_test, r_squared_test, n_active_coefficients_reduced, \
            relative_l2_distance_test_reduced, r_squared_test_reduced, best_formula_reduced, r_squared_val_reduced, \
            n_evaluations, best_model_parameters, best_coefficients, best_coefficients_reduced
    t_0 = time()
    # generate the list of model parameters as specified in config['model_parameter_search']
    if model_parameters_perfect is None:
        model_parameter_list = get_model_parameter_list(model_parameters_max, config, training_parameters, x_train, y_train)
    else:
        model_parameter_list = [model_parameters_perfect] * training_parameters['repetitions']
    t_0 = time()
    try:
        # Start running jobs in parallel (not up to date, please choose the sequential computation)
        if training_parameters['parallel']:
            raise NotImplementedError(f'You are trying to run the computations sequentially, however, this is not up '
                                      f'to date with the rest of the benchmark. Please use the sequential computation')
            logging.info('Parallel computation')
            best_relative_l2_distance_train, best_relative_l2_distance_val, r_squared_val, best_training_time, best_coefficients, \
                best_model_parameters = run_jobs_in_parallel_new_pool(x_train, y_train, target_expr,
                                                                      model_parameter_list,
                                                                      training_parameters, accuracy)
        else:
            print('Sequential computation')
            logging.info('Sequential computation')
            best_relative_l2_distance_train, best_relative_l2_distance_val, r_squared_val, best_training_time, best_coefficients, \
                best_model_parameters, n_evaluations, cumulative_training_time = run_jobs_sequentially(x_train, y_train,
                                                                                                       target_expr,
                                                                                                       model_parameter_list,
                                                                                                       seed,
                                                                                                       training_parameters,
                                                                                                       accuracy, 
                                                                                                       custom_loss)
    except Exception as e:
        # print(f'Exception {e}')
        logging.critical(e, exc_info=True)
        raise e

    # pool.terminate()  # Terminate all remaining processes
    t_1 = time()
    logging.info(f"Time for multiprocessing in total: {t_1 - t_0}")

    if best_coefficients is None:
        logging.info(f'The time or evaluations limit was reached before a single set of coefficients has been reached.'
                     f'So, we will simply return the 0 function.')
        return (torch.ones(1), torch.ones(1), torch.zeros(1), '0', training_parameters['time_limit'],
                torch.zeros(1), torch.ones(1), torch.zeros(1), torch.zeros(1),
                torch.ones(1), torch.zeros(1), '0', torch.zeros(1), training_parameters['evaluations_limit'],
                None, None, None)
    # First evaluate without any finetuning
    n_active_coefficients, relative_l2_distance_test, r_squared_test, best_formula, r_squared_val = evaluate_test(
        best_model_parameters, x_train, y_train, x_test, y_test, training_parameters['device'], best_coefficients,
        0.0, False, training_parameters['model'], custom_loss=custom_loss)

    # Now evaluate after cutting off "large" parameters and finetuning
    if training_parameters['iterative_finetuning']:
        n_active_coefficients_reduced, relative_l2_distance_test_reduced, r_squared_test_reduced, best_formula_reduced, \
            r_squared_val_reduced, best_coefficients_reduced = evaluate_test_iterative_finetuning(best_model_parameters,
                                                                                                  x_train, y_train,
                                                                                                  x_test,
                                                                                                  y_test,
                                                                                                  training_parameters[
                                                                                                      'device'],
                                                                                                  best_coefficients,
                                                                                                  True,
                                                                                                  lambda_1=
                                                                                                  training_parameters[
                                                                                                      'lambda_1_finetuning'],
                                                                                                  model_str=
                                                                                                  training_parameters[
                                                                                                      'model'], 
                                                                                                  custom_loss=custom_loss)
    else:
        n_active_coefficients_reduced, relative_l2_distance_test_reduced, r_squared_test_reduced, best_formula_reduced, \
            r_squared_val_reduced = evaluate_test(best_model_parameters, x_train, y_train, x_test, y_test,
                                                  training_parameters['device'], best_coefficients, 0.01, True,
                                                  custom_loss=custom_loss)

    logging.info(
        f"Best distance (train, val, test): {best_relative_l2_distance_train, best_relative_l2_distance_val, relative_l2_distance_test}")
    logging.info(f"Best formula: {best_formula} best training time: {best_training_time}")

    # displaying the memory
    # logging.info(f'Memory usage (current, peak) (end): {tracemalloc.get_traced_memory()}')
    # stopping the library
    # tracemalloc.stop()

    return best_relative_l2_distance_train, best_relative_l2_distance_val, r_squared_val, best_formula, t_1 - t_0, \
        n_active_coefficients, relative_l2_distance_test, r_squared_test, n_active_coefficients_reduced, \
        relative_l2_distance_test_reduced, r_squared_test_reduced, best_formula_reduced, r_squared_val_reduced, \
        n_evaluations, best_model_parameters, best_coefficients, best_coefficients_reduced
