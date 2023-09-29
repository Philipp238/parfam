import copy

import logging
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

from parfam_torch import ParFamTorch, Evaluator
from reparfam import ReParFam
from utils import add_noise, relative_l2_distance, r_squared, is_tuple_sorted

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

function_dict = {'sqrt': lambda x: torch.sqrt(torch.abs(x)),
                 'exp': lambda x: torch.minimum(torch.exp(x), np.exp(10) + torch.abs(x)),
                 'cos': torch.cos, 'sin': torch.sin}
function_name_dict = {'sqrt': lambda x: sympy.sqrt(sympy.Abs(x)), 'exp': sympy.exp, 'cos': sympy.cos, 'sin': sympy.sin}


def make_unpickable(model_parameters):
    function_names_str = model_parameters['functions']
    model_parameters['functions'] = []
    model_parameters['function_names'] = []
    for function_name_str in function_names_str:
        model_parameters['functions'].append(function_dict[function_name_str])
        model_parameters['function_names'].append(function_name_dict[function_name_str])
    return model_parameters


# Evaluates the model performance
def evaluate(model, coefficients, x_train, y_train, x_val, y_val, target_expr=""):
    relative_l2_distance_train = relative_l2_distance(model, coefficients, x_train, y_train)
    relative_l2_distance_val = relative_l2_distance(model, coefficients, x_val, y_val)
    r_squared_val = r_squared(model, coefficients, x_val, y_val)
    if target_expr != "":
        print(f'Target expression: {target_expr}')
    print(f'Relative l_2-distance train: {relative_l2_distance_train}')
    print(f'Relative l_2-distance validation: {relative_l2_distance_val}')
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
def hyperparameter_training(x, y, target_expr, training_parameters, model_parameters, verbose=True):
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
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, test_size=0.2)
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
    print(f'Number parameters: {n_params}')
    print(f'Number parameters active: {n_params_active}')
    evaluator = Evaluator(x_train, y_train, lambda_0=0, lambda_1=lambda_1, model=model, mask=mask,
                          n_params=n_params, lambda_1_cut=lambda_1_cut, lambda_1_piecewise=lambda_1_piecewise)
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
                                                                                   y_train,
                                                                                   x_val, y_val, target_expr)
    return relative_l2_distance_train, relative_l2_distance_val, r_squared_val, t_1 - t_0, coefficients, evaluator.evaluations


stop_flag = 0  # global variable to check whether the wanted accuracy was reached, set to 1 if that is the case


def train_model_old_event(model_parameters, x, y, target_expr, training_parameters, accuracy, found_event,
                          result_dict, lock):
    if found_event.is_set():
        logging.info('Event is set, but the process is still running. what happened?')
    relative_l2_distance_train, relative_l2_distance_val, r_squared_val, training_time, coefficients, n_evaluations = \
        hyperparameter_training(x, y, target_expr, training_parameters, copy.deepcopy(model_parameters),
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


def train_model(model_parameters, x, y, target_expr, training_parameters, result_dict, lock):
    relative_l2_distance_train, relative_l2_distance_val, r_squared_val, training_time, coefficients, n_evaluations = hyperparameter_training(
        x, y, target_expr, training_parameters, model_parameters, verbose=True)
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

        result_dict['counter'] -= 1
        logging.info(f'Counter: {result_dict["counter"]}')


def get_complete_model_parameter_list(model_parameters_max, model_str):
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
    model_parameters = {'degree_input_polynomials': 0, 'degree_output_polynomials': max_deg_output, 'functions': [],
                        'function_names': [], 'width': 1, 'degree_input_denominator': 0,
                        'degree_output_denominator': 0,
                        'degree_output_polynomials_specific': None,
                        'degree_output_polynomials_denominator_specific': None,
                        'enforce_function': False, 'maximal_potence': model_parameters_max['maximal_potence']}
    model_parameter_list.append(model_parameters)

    # Continue with checking if it is a rational function
    for d_output_denom in range(1, max_deg_output_denominator + 1):
        for d_output in range(1, max_deg_output + 1):
            model_parameters = {'degree_input_polynomials': 0, 'degree_output_polynomials': d_output,
                                'functions': [],
                                'function_names': [], 'width': 1, 'degree_input_denominator': 0,
                                'degree_output_denominator': d_output_denom,
                                'degree_output_polynomials_specific': None,
                                'degree_output_polynomials_denominator_specific': None,
                                'enforce_function': False,
                                'maximal_potence': model_parameters_max['maximal_potence']}
            model_parameter_list.append(model_parameters)
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
                            for enforce_function in [False, True]:
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
    model_parameters = {'degree_input_polynomials': 0, 'degree_output_polynomials': max_deg_output, 'functions': [],
                        'function_names': [], 'width': 1, 'degree_input_denominator': 0,
                        'degree_output_denominator': 0,
                        'degree_output_polynomials_specific': None,
                        'degree_output_polynomials_denominator_specific': None,
                        'enforce_function': False, 'maximal_potence': model_parameters_max['maximal_potence']}
    model_parameter_list.append(model_parameters)

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


def get_model_parameter_list(model_parameters_max, config, training_parameters):
    if config['META']['model_parameter_search'] == 'No':
        # One specific set of model parameters, which will be used instead of the hyperparameter search
        model_parameters_fix = dict(config.items("MODELPARAMETERSFIX"))
        model_parameters_fix = {key: ast.literal_eval(model_parameters_fix[key]) for key in
                                model_parameters_fix.keys()}
        model_parameters_fix['functions'] = model_parameters_fix['function_names']
        model_parameter_list = [model_parameters_fix]
    elif config['META']['model_parameter_search'] == 'Complete':
        model_parameter_list = get_complete_model_parameter_list(model_parameters_max, training_parameters['model'])
    elif config['META']['model_parameter_search'] == 'Max':
        model_parameter_list = get_max_model_parameter_list(model_parameters_max)
    else:
        raise NotImplementedError(
            f"config['META']['model_parameter_search'] must be either 'No', 'Complete' or 'Max' but not"
            f" '{config['META']['model_parameter_search']}'")
    return model_parameter_list * training_parameters['repetitions']


def run_jobs_in_parallel_new_pool(x, y, target_expr, model_parameter_list, training_parameters, accuracy):
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
            (model_parameters, x, y, target_expr, training_parameters, accuracy, event, return_dict, lock) for
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


def run_jobs_in_parallel_old_pool(x, y, target_expr, model_parameter_list, training_parameters, accuracy):
    workers = min(mp.cpu_count(), len(model_parameter_list))
    logging.info(f"Available workers:{mp.cpu_count()} ")
    logging.info(f"Used workers:{workers} ")
    # Create a multiprocessing pool
    pool = mp.Pool(processes=workers)
    # Combinations which will be performed in parallel
    combinations = [(model_parameters, x, y, target_expr, training_parameters, accuracy) for model_parameters in
                    model_parameter_list]
    results = pool.starmap(train_model, combinations)
    # Close the pool and wait for the processes to finish
    pool.close()
    pool.join()

    # Check the trained_models and stop all processes if necessary
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


def run_jobs_in_parallel_old_event(x, y, target_expr, model_parameter_list, training_parameters, accuracy):
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
                        model_parameters, x, y, target_expr, training_parameters, accuracy, found_event,
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


def run_jobs_in_parallel(x, y, target_expr, model_parameter_list, training_parameters, accuracy):
    manager = Manager()
    return_dict = manager.dict()

    return_dict['counter'] = len(model_parameter_list)
    return_dict['relative_l2_distance_train'] = np.inf
    return_dict['relative_l2_distance_val'] = np.inf
    return_dict['formula'] = None
    return_dict['training_time'] = None
    lock = Lock()

    pool = [Process(target=train_model,
                    args=(model_parameters, x, y, target_expr, training_parameters, return_dict, lock))
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


def run_jobs_in_parallel_test(x, y, target_expr, model_parameter_list, training_parameters, accuracy):
    manager = Manager()
    return_dict = manager.dict()

    return_dict['counter'] = len(model_parameter_list)
    return_dict['relative_l2_distance_train'] = np.inf
    return_dict['relative_l2_distance_val'] = np.inf
    return_dict['formula'] = None
    return_dict['training_time'] = None
    lock = Lock()

    pool = [Process(target=train_model,
                    args=(model_parameters, x, y, target_expr, training_parameters, return_dict, lock))
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


def run_jobs_sequentially(x, y, target_expr, model_parameter_list, training_parameters, accuracy):
    best_relative_l2_distance_train, best_relative_l2_distance_val, best_formula, best_training_time = None, np.inf, None, None
    best_r_squared_val, best_coefficients, best_model_parameters = None, None, None
    cumulative_n_evaluations = 0
    cumulative_training_time = 0
    time_limit = training_parameters['time_limit']
    evaluations_limit = training_parameters['evaluations_limit']
    for model_parameters in model_parameter_list:
        t_0 = time()
        relative_l2_distance_train, relative_l2_distance_val, r_squared_val, training_time, coefficients, n_evaluations = \
            hyperparameter_training(x, y, target_expr, training_parameters, copy.deepcopy(model_parameters),
                                    verbose=True)
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
        logging.info(f'Training parameters: {training_parameters}')
        logging.info(f'Model parameters: {model_parameters}')
        logging.info(f'Relative l2 distance train: {relative_l2_distance_train}')
        logging.info(f'Relative l2 distance validation: {relative_l2_distance_val}')
        logging.info(f'Training time: {training_time}')
        logging.info(f'Cumulative training time: {cumulative_training_time}')
        logging.info(f'Cumulative number of evaluations: {cumulative_n_evaluations}')

        if relative_l2_distance_val < best_relative_l2_distance_val:
            logging.info(f"New best relative l2 distance validation: {relative_l2_distance_val}")
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
                            width=model_parameters['width'], maximal_potence=model_parameters['maximal_potence'],
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
                         width=model_parameters['width'], maximal_potence=model_parameters['maximal_potence'],
                         functions=model_parameters['functions'], device=device,
                         function_names=model_parameters['function_names'],
                         enforce_function=model_parameters['enforce_function'])
    return model


def get_formula(coefficients, model_parameters, n_input, device, decimals):
    model = setup_model(copy.deepcopy(model_parameters), n_input, device)
    return model.get_formula(torch.tensor(coefficients, device=device), decimals=decimals, verbose=False)


def evaluate_test_iterative_finetuning(best_model_parameters, x_train, y_train, x_test, y_test, device,
                                       best_coefficients, retrain, lambda_1, model_str):
    model = setup_model(copy.deepcopy(best_model_parameters), x_test.shape[1], device, model_str)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2)
    relative_l2_distance_val = relative_l2_distance(model, best_coefficients, x_val, y_val)
    n_active_coefficients = sum(torch.abs(best_coefficients) > 0)
    finetune = True
    update = True
    if finetune:
        if model_str == 'ParFamTorch':
            iterations = range(5, 1, -1)
        else:
            iterations = range(9, 1, -2)
        for i in iterations:
            # iterate through the cutoffs 10**5 to 10**2
            for j in range(2):
                if j == 1 and not update:
                    # If we didn't change anything for this level of cutoff, then we don't have to restart it again
                    continue
                # perform each cutoff + optimization twice, since this often helps to reduce the number
                # of coefficients further
                cutoff = 10 ** (-i)
                mask = torch.abs(best_coefficients) > cutoff
                coefficients = copy.deepcopy(best_coefficients)
                coefficients[~ mask] = 0
                n_active_coefficients_new = sum(mask)
                if retrain and (n_active_coefficients_new > 0) and (n_active_coefficients_new < n_active_coefficients):
                    if len(y_train) > 5000:
                        subset = np.random.choice(len(y_train), 5000, replace=False)
                    else:
                        subset = np.arange(len(y_train))
                    x_train = x_train[subset]
                    y_train = y_train[subset]

                    n_params = model.get_number_parameters()
                    evaluator = Evaluator(x_train, y_train, lambda_0=0, lambda_1=lambda_1, model=model, mask=mask,
                                          n_params=n_params)

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
                    else:
                        logging.info(f'Coefficients_ while finetuning are None. Mask: {mask}')
                        continue
                    relative_l2_distance_val_new = relative_l2_distance(model, coefficients, x_val, y_val)

                    # Check if the formula found now is worth keeping or not
                    better_performance = relative_l2_distance_val_new < relative_l2_distance_val
                    slightly_worse_performance = relative_l2_distance_val_new < relative_l2_distance_val * 1.5
                    considerably_less_coefficients = n_active_coefficients_new < n_active_coefficients * 0.75
                    update = better_performance or (slightly_worse_performance and considerably_less_coefficients)
                    if update:
                        relative_l2_distance_val = relative_l2_distance_val_new
                        best_coefficients = coefficients
                        n_active_coefficients = n_active_coefficients_new

    best_coefficients = model.get_normalized_coefficients(best_coefficients)
    relative_l2_distance_test = relative_l2_distance(model, best_coefficients, x_test, y_test)
    r_squared_test = r_squared(model, best_coefficients, x_test, y_test)
    r_squared_val = r_squared(model, best_coefficients, x_val, y_val)
    if model_str == 'ParFamTorch':
        formula = model.get_formula(best_coefficients, decimals=3, verbose=False)
    else:
        for i in range(3, 10):
            formula = model.get_formula(best_coefficients, decimals=i, verbose=False)
            if str(formula) != '0':
                break
    return n_active_coefficients, relative_l2_distance_test, r_squared_test, formula, r_squared_val


def evaluate_test(best_model_parameters, x_train, y_train, x_test, y_test, device, best_coefficients, cutoff, retrain,
                  model_str):
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

    relative_l2_distance_test = relative_l2_distance(model, best_coefficients, x_test, y_test)
    r_squared_test = r_squared(model, best_coefficients, x_test, y_test)
    r_squared_val = r_squared(model, best_coefficients, x_val, y_val)
    if model_str == 'ParFamTorch':
        formula = model.get_formula(best_coefficients, decimals=3, verbose=False)
    else:
        for i in range(3, 10):
            formula = model.get_formula(best_coefficients, decimals=i, verbose=False)
            if str(formula) != '0':
                break
    return n_active_coefficients, relative_l2_distance_test, r_squared_test, formula, r_squared_val


def model_parameter_search(x, y, target_expr, model_parameters_max, training_parameters, accuracy, config,
                           model_parameters_perfect=None):
    np.random.seed(training_parameters['seed'])
    torch.manual_seed(training_parameters['seed'])
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
    # generate the list of model parameters as specified in config['model_parameter_search']
    if model_parameters_perfect is None:
        model_parameter_list = get_model_parameter_list(model_parameters_max, config, training_parameters)
    else:
        model_parameter_list = [model_parameters_perfect] * training_parameters['repetitions']
    # Set logging
    directory = 'trained_models'
    logging.basicConfig(filename=os.path.join(directory, 'experiment.log'), level=logging.INFO)

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
                                                                                                       training_parameters,
                                                                                                       accuracy)
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
                torch.ones(1), torch.zeros(1), '0', torch.zeros(1), training_parameters['evaluations_limit'])

    # First evaluate without any finetuning
    n_active_coefficients, relative_l2_distance_test, r_squared_test, best_formula, r_squared_val = evaluate_test(
        best_model_parameters, x_train, y_train, x_test, y_test, training_parameters['device'], best_coefficients,
        0.0, False, training_parameters['model'])

    # Now evaluate after cutting off "large" parameters and finetuning
    if training_parameters['iterative_finetuning']:
        n_active_coefficients_reduced, relative_l2_distance_test_reduced, r_squared_test_reduced, best_formula_reduced, \
            r_squared_val_reduced = evaluate_test_iterative_finetuning(best_model_parameters, x_train, y_train, x_test,
                                                                       y_test, training_parameters['device'],
                                                                       best_coefficients, True,
                                                                       lambda_1=training_parameters[
                                                                           'lambda_1_finetuning'],
                                                                       model_str=training_parameters['model'])
    else:
        n_active_coefficients_reduced, relative_l2_distance_test_reduced, r_squared_test_reduced, best_formula_reduced, \
            r_squared_val_reduced = evaluate_test(best_model_parameters, x_train, y_train, x_test, y_test,
                                                  training_parameters['device'], best_coefficients, 0.01, True)

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
        n_evaluations
