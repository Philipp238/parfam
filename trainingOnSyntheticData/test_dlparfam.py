import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import dual_annealing, basinhopping, differential_evolution

from pmlb import fetch_data

import sympy
from sympy import symbols
from time import time
import numpy as np
import os
import sys
import pandas as pd
import configparser
import ast

# from srbench.utils import get_formula
sys.path[0] = os.getcwd()
print(os.getcwd())
from utils import create_dataset
from parfam_torch import ParFamTorch, Evaluator

import utils


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = 'cpu'
print(f'Using {device}.')


def round_expr(expr, num_digits):
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(sympy.Number)})


def evaluate(model, coefficients, x, y, target_expr, lambda_1):
    relative_l2_distance = torch.norm(y - model.predict(torch.tensor(coefficients).to(device), x), p=2) / torch.norm(y,
                                                                                                                     p=2)
    formula = model.get_formula(coefficients, verbose=True)
    print(f'Estimated expression: {formula}')
    print(f'Target expression: {target_expr}')
    print(f'Relative l_2-distance: {relative_l2_distance}')
    print(f'L_1 reg: {lambda_1 * np.linalg.norm(coefficients)}')
    print(f'Loss: {relative_l2_distance + lambda_1 * np.linalg.norm(coefficients)}')
    return relative_l2_distance, formula


def training(model, evaluator, x, y, n_params_active, maxiter, initial_temp, maxfun, no_local_search, optimizer,
             x0=None):
    lw = [-10] * n_params_active
    up = [10] * n_params_active
    t_0 = time()
    model.prepare_input_monomials(x)
    if optimizer == 'dual_annealing':
        ret = dual_annealing(evaluator.loss_func, bounds=list(zip(lw, up)), maxiter=maxiter, maxfun=maxfun,
                             initial_temp=initial_temp, no_local_search=no_local_search, x0=x0)
    elif optimizer == 'basinhopping':
        if x0 is None:
            x0 = np.random.randn(n_params_active)
        ret = basinhopping(evaluator.loss_func, niter=maxiter,
                           x0=x0, minimizer_kwargs={'method': 'BFGS', 'jac': evaluator.gradient})
    elif optimizer == 'differential_evolution':
        ret = differential_evolution(evaluator.loss_func, bounds=list(zip(lw, up)), maxiter=maxiter, x0=x0, popsize=100)
    t_1 = time()
    print(f'Training time: {t_1 - t_0}')
    model.testing_mode()
    return ret


def standard_training(x, y, target_expr, optimizer, iter1, functions, function_names,
                      degree_output_polynomials_specific, enforce_function,
                      degree_input_polynomials, degree_output_polynomials, width, degree_output_denominator,
                      degree_input_denominator, iter2=0, verbose=False, lambda_1=0.001):
    if verbose:
        print('##### Standard training #####')
    model = ParFamTorch(n_input=x.shape[1], degree_input_numerator=degree_input_polynomials,
                        degree_output_numerator=degree_output_polynomials, width=width,
                        functions=functions, degree_output_denominator=degree_output_denominator,
                        degree_input_denominator=degree_input_denominator,
                        function_names=function_names,
                        degree_output_numerator_specific=degree_output_polynomials_specific,
                        enforce_function=enforce_function)
    n_params = model.get_number_parameters()
    evaluator = Evaluator(x, y, lambda_0=0, lambda_1=lambda_1, model=model, n_params=n_params, lambda_05=0,
                          lambda_1_cut=0)
    t_0 = time()
    ret = training(model, evaluator, x, y, n_params, maxiter=iter1, initial_temp=50000, maxfun=10 ** 9,
                   no_local_search=True, optimizer=optimizer)
    if iter2 > 0:
        ret_loc = training(model, evaluator, x, y, n_params, maxiter=iter2, initial_temp=500, maxfun=10 ** 9,
                           no_local_search=False, x0=ret.x, optimizer=optimizer)
        ret = ret_loc
    t_1 = time()
    relative_l2_distance, formula = evaluate(model, ret.x, x, y, target_expr, lambda_1=lambda_1)
    return relative_l2_distance, formula, t_1 - t_0


def classifier_informed_training(x, y, target_expr, net, optimizer, iter1, functions, function_names,
                                 degree_output_polynomials_specific, enforce_function,
                                 degree_input_polynomials, degree_output_polynomials, width, degree_output_denominator,
                                 degree_input_denominator, iter2=0, verbose=False, lambda_1=0.001):
    if verbose:
        print('##### Classifier informed training ######')
    prediction = net(torch.tensor(y, dtype=torch.float, device='cpu'))
    # mask = prediction >= torch.quantile(prediction, 0.7)
    mask = prediction >= 0.2
    model = ParFamTorch(n_input=x.shape[1], degree_input_numerator=degree_input_polynomials,
                        degree_output_numerator=degree_output_polynomials, width=width,
                        functions=functions, degree_output_denominator=degree_output_denominator,
                        degree_input_denominator=degree_input_denominator,
                        function_names=function_names,
                        degree_output_numerator_specific=degree_output_polynomials_specific,
                        enforce_function=enforce_function)
    n_params = model.get_number_parameters()
    evaluator = Evaluator(x, y, lambda_0=0, lambda_1=lambda_1, model=model, n_params=n_params, mask=mask)
    symbolic_a = [symbols(f'a{i}') if mask[i] else 0 for i in range(n_params)]
    symbolic_expr = model.get_formula(symbolic_a, verbose=False)
    if verbose:
        print(f'Reduced formula: {symbolic_expr}')
    n_params_active = sum(mask)
    t_0 = time()
    ret = training(model, evaluator, x, y, n_params_active, maxiter=iter1, initial_temp=50000, maxfun=10 ** 9,
                   no_local_search=True, optimizer=optimizer)
    if iter2 > 0:
        ret_loc = training(model, evaluator, x, y, n_params_active, maxiter=iter2, initial_temp=500, maxfun=10 ** 9,
                           no_local_search=False, x0=ret.x, optimizer=optimizer)
        ret = ret_loc
    t_1 = time()
    coefficients = np.zeros(n_params)
    coefficients[mask] = ret.x
    relative_l2_distance, formula = evaluate(model, coefficients, x, y, target_expr, lambda_1=lambda_1)
    return relative_l2_distance, formula, t_1 - t_0


def data(function_name):
    x = np.arange(-10, 10, 0.1)
    x = x.reshape(len(x), 1)

    if function_name == '1':
        a = np.random.randn(5)
        func = lambda coeff, x, module: module.cos(coeff[0] * x)

    if function_name == '2':
        a = np.random.randn(5)
        func = lambda coeff, x, module: module.cos(coeff[0] * x) * coeff[1] + coeff[2] + module.cos(coeff[3] * x) ** 2 * \
                                        coeff[4]

    if function_name == '3':
        a = np.random.randn(3)
        func = lambda coeff, x, module: module.cos(coeff[0] * x) * coeff[1] + coeff[2]

    if function_name == '4':
        a = np.random.randn(5)
        func = lambda coeff, x, module: module.cos(coeff[0] * x) * coeff[1] + coeff[2] + module.cos(coeff[3] * x) ** 2 * \
                                        coeff[4]
    if function_name == '5':
        functions = [np.sin, np.cos]
        function_names = [sympy.sin, sympy.cos]
        target_model = ParFamNP(n_input=1, degree_input_polynomials=2, degree_output_polynomials=2, width=1,
                                functions=functions, function_names=function_names)

        n_params = target_model.get_number_parameters()
        a = np.random.randn(n_params)
        switch = target_model.get_random_coefficients_unique(n_functions_max=3)
        a = switch * a
        y = target_model.predict(a, x)

        target_expr = target_model.get_formula(a, verbose=False)
        target_expr = round_expr(target_expr, 3)

        return x, y, target_expr, switch

    if function_name == 'feynman_III_10_19':
        x = np.random.randn(1000, 4) * 10
        # x = np.random.uniform(1, 5, 4000).reshape(1000, 4)
        a = np.ones(4)
        func = lambda coeff, x, module: x[3] * module.sqrt(
            coeff[0] * x[0] ** 2 + coeff[1] * x[1] ** 2 + coeff[2] * x[2] ** 2)

    if function_name == 'feynman_III_13_18':
        x = np.random.randn(1000, 4) * 10
        # x = np.random.uniform(1, 5, 4000).reshape(1000, 4)
        a = np.ones(4)
        func = lambda coeff, x, module: 4 * np.pi * x[0] * x[1] ** 2 * x[2] / x[3]

    if function_name == 'feynman_I_6_2a':
        x = np.arange(1, 3, 0.1)
        x = x.reshape(len(x), 1)
        a = np.ones(4)
        func = lambda coeff, x, module: module.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)

    if function_name == 'feynman_I_6_2b_weakened':
        x_1 = np.arange(1, 3, 0.1)
        x_2 = np.arange(1, 3, 0.1)
        x_3 = np.arange(1, 3, 0.1)
        g = np.meshgrid(x_1, x_2, x_3)
        x = np.vstack(map(np.ravel, g)).T
        a = np.ones(4)
        func = lambda coeff, x, module: module.exp(-(x[0] - x[1]) ** 2 / x[2] ** 2 / 2) / np.sqrt(2 * np.pi)

    if function_name == 'feynman_III_14_14_weakened':
        x_1 = np.arange(1, 5, 0.2)
        x_2 = np.arange(1, 2, 0.2)
        x_3 = np.arange(1, 2, 0.2)
        x_4 = np.arange(1, 2, 0.2)
        x_5 = np.arange(1, 2, 0.2)
        g = np.meshgrid(x_1, x_2, x_3, x_4, x_5)
        x = np.vstack(map(np.ravel, g)).T
        a = np.ones(4)
        func = lambda coeff, x, module: module.exp(x[1] * x[2] / (x[3] * x[4]))

    if function_name == 'feynman_III_13_18':
        x_1 = np.arange(1, 5, 0.2)
        x_2 = np.arange(1, 2, 0.2)
        x_3 = np.arange(1, 2, 0.2)
        x_4 = np.arange(1, 2, 0.2)
        g = np.meshgrid(x_1, x_2, x_3, x_4)
        x = np.vstack(map(np.ravel, g)).T
        a = np.ones(4)
        func = lambda coeff, x, module: x[0] * x[1] ** 2 * x[3] / x[4]

    if function_name == '7':
        a = np.random.randn(3)

        def func(coeff, x, module):
            if module == np:
                abs = module.abs
            else:
                abs = module.Abs
            return module.sqrt(abs(coeff[0] * x ** 2 + coeff[1] * x + coeff[2]))

    if function_name == '8':
        x = np.random.randn(1000, 3) * 10
        a = np.random.randn(3)
        func = lambda coeff, x, module: coeff[0] * x[0] ** 2 + coeff[1] * x[1] ** 2 + coeff[2] * x[2] ** 2

    if function_name == '9':
        a = np.random.randn(5)
        func = lambda coeff, x, module: module.exp(a[0] * x)

    if function_name == '10':
        a = np.random.randn(5) / 100
        func = lambda coeff, x, module: module.exp(a[0] * x ** 2)

    if x.shape[1] == 1:
        y = func(a, x, np).squeeze(1)
        x_sym = symbols('x')
    else:
        y = func(a, x.T, np)
        x_sym = []
        for i in range(x.shape[1]):
            x_sym.append(symbols(f'x{i}'))
    target_expr = func(a, x_sym, sympy)
    target_expr = round_expr(target_expr, 3)

    switch = None

    return x, y, target_expr, switch


def data_feynmann(function_name, max_dataset_length=1000):
    path_pmlb = '/home/philipp/projects/phyiscalLawLearning/pmlb'
    path_pmlb_datasets = os.path.join(path_pmlb, 'datasets')
    x, y = fetch_data(function_name, return_X_y=True, local_cache_dir=path_pmlb_datasets)
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y).astype(np.float64)
    target_formula = utils.get_formula(function_name)
    if len(y) > max_dataset_length:
        subset = np.random.choice(len(y), max_dataset_length, replace=False)
    else:
        subset = np.arange(len(y))
    x = x[subset]
    y = y[subset]
    return x, y, target_formula, None


class MLP(nn.Module):
    def __init__(self, L, n_params, h):
        super().__init__()
        self.fc1 = nn.Linear(L, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, n_params)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Yann LeCun et al. "Efficient BackProp" 1.7159 * torch.tanh(2/3*self.fc3(x))
        return x


def learning(net):
    function_names = ['feynman_III_10_19']
    feynmann_data = False
    # function_names = ['feynman_III_10_19']
    successes_standard = 0
    successes_classifier = 0
    n_experiments = 10
    for _ in range(n_experiments):
        for function_name in function_names:
            if feynmann_data:
                x, y, target_expr, switch = data_feynmann(function_name)
            else:
                x, y, target_expr, switch = data(function_name)
            print(f'########## Learning the following function: {target_expr} ##########')
            # optimizer = 'dual_annealing'
            iter1 = 10
            iter2 = 0
            optimizer = 'basinhopping'
            relative_l2_distance = \
                standard_training(x, y, target_expr, iter1=iter1, iter2=iter2, optimizer=optimizer, lambda_1=0.0001)[0]
            if relative_l2_distance < 0.001:
                successes_standard += 1
            # relative_l2_distance = classifier_informed_training(x, y, target_expr, net, iter1=iter1, iter2=iter2,
            #                                                    optimizer=optimizer)[0]
            if relative_l2_distance < 0.001:
                successes_classifier += 1
    print(f'Standard training solved {successes_standard} out of {n_experiments}.')
    print(f'Classifier informed training solved {successes_classifier} out of {n_experiments}.')


def learning_random_functions(net):
    import datetime
    datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    test_set_size = 60
    functions = [torch.sin, lambda x: torch.sqrt(torch.abs(x))]
    function_names = [sympy.sin, lambda x: sympy.sqrt(sympy.Abs(x))]
    degree_output_polynomials = 2
    degree_input_polynomials = 2
    width = 1
    degree_output_polynomials_specific = [1, 1]
    degree_output_denominator = 0
    degree_input_denominator = 0
    enforce_function = False
    n_functions_max = 2

    iter_standard = 2000
    iter_classifier = 2000
    iter2 = 0
    optimizer = 'basinhopping'
    lambda_1 = 0.001
    accuracy = 0.0001

    torch.manual_seed(1234)
    np.random.seed(1234)
    y, a, params = create_dataset(test_set_size, True, degree_input_polynomials,
                                  degree_output_polynomials,
                                  width, functions, function_names, device='cpu', normalization=False,
                                  degree_output_polynomials_specific=degree_output_polynomials_specific,
                                  enforce_function=enforce_function, n_functions_max=n_functions_max)
    successes_standard = 0
    successes_classifier = 0
    results_dict = {}
    results_dict['formula'] = []
    results_dict['classifier'] = []
    results_dict['rel_l2_distance'] = []
    results_dict['success'] = []
    results_dict['estimated_formula'] = []
    results_dict['training_time'] = []
    for i in range(test_set_size):
        model = ParFamTorch(n_input=1, degree_input_numerator=degree_input_polynomials,
                            degree_output_numerator=degree_output_polynomials, width=width,
                            functions=functions, function_names=function_names, enforce_function=enforce_function,
                            degree_output_numerator_specific=degree_output_polynomials_specific)

        target_expr = model.get_formula(params[i], verbose=False)

        x = torch.arange(-10, 10, 0.1)
        x = x.reshape((len(x), 1))

        print(f'########## Learning the following function: {target_expr} ##########')
        # optimizer = 'dual_annealing'

        if True:
            relative_l2_distance_standard, estimated_formula, training_time = \
                standard_training(x, y[i], target_expr, optimizer, iter_standard, functions, function_names,
                                  degree_output_polynomials_specific, enforce_function,
                                  degree_input_polynomials, degree_output_polynomials, width, degree_output_denominator,
                                  degree_input_denominator, iter2=iter2, verbose=True, lambda_1=lambda_1)
            if relative_l2_distance_standard < accuracy:
                successes_standard += 1
            results_dict['formula'].append(target_expr)
            results_dict['estimated_formula'].append(estimated_formula)
            results_dict['training_time'].append(training_time)
            results_dict['classifier'].append(False)
            results_dict['success'].append((relative_l2_distance_standard < accuracy).item())
            results_dict['rel_l2_distance'].append(relative_l2_distance_standard.item())
            results_df = pd.DataFrame(results_dict)
            results_df.to_csv(os.path.join('trainingOnSyntheticData', f'{datetime}_results_iter_{iter_classifier}.csv'))
        if True:
            relative_l2_distance_classifier, estimated_formula, training_time = \
                classifier_informed_training(x, y[i], target_expr, net, optimizer, iter_classifier, functions,
                                             function_names,
                                             degree_output_polynomials_specific, enforce_function,
                                             degree_input_polynomials, degree_output_polynomials, width,
                                             degree_output_denominator,
                                             degree_input_denominator, iter2=iter2, verbose=True, lambda_1=lambda_1)
            if relative_l2_distance_classifier < accuracy:
                successes_classifier += 1
            results_dict['formula'].append(target_expr)
            results_dict['estimated_formula'].append(estimated_formula)
            results_dict['training_time'].append(training_time)
            results_dict['classifier'].append(True)
            results_dict['success'].append((relative_l2_distance_classifier < accuracy).item())
            results_dict['rel_l2_distance'].append(relative_l2_distance_classifier.item())
            results_df = pd.DataFrame(results_dict)
            results_df.to_csv(os.path.join('trainingOnSyntheticData', f'{datetime}_results_iter_{iter_classifier}.csv'))

    print(f'Standard training solved {successes_standard} out of {test_set_size}.')
    print(f'Classifier informed training solved {successes_classifier} out of {test_set_size}.')

def get_topk_predictions(data_parameters, prediction, k):
    n_functions = len(data_parameters['function_names_str'])

    # Use for each polynomial 1+max_degree variables, unless the max_degree is 0; Same for the functions (+1 in case there is no function used)
    lengths = [data_parameters['degree_input_numerator'], data_parameters['degree_input_denominator'], data_parameters['degree_output_numerator'], 
               data_parameters['degree_output_denominator'], len(data_parameters['function_names_str'])]
    lengths = [length + 1 for length in lengths if length > 0]
    
    list_probs = []
    index = 0
    for length in lengths:
        # list_probs.append(torch.softmax(prediction[i, index:index+length], dim=0))
        list_probs.append(torch.softmax(prediction[index:index+length], dim=0))
        index += length
    probabilities = utils.compute_comb_product(list_probs) 
    # probabilities = utils.compute_comb_product(list_probs) 
    v, indices = torch.topk(probabilities.flatten(), k=min(k, len(probabilities.flatten())))
    topk_predictions = torch.tensor(np.array([np.unravel_index(indices.numpy(), probabilities.shape)]).squeeze(0)).T
    if n_functions > 1:
        # functions in target are one hot encoded, why they are categorical in topk_predictions
        topk_predictions = torch.cat([topk_predictions[:, :-1], nn.functional.one_hot(topk_predictions[:,-1], num_classes=n_functions+1)[:, 1:]],
                                        dim=1)
    return topk_predictions

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

def setup_df():
    perfect_mp_pol = {'feynman_II_27_16': torch.tensor([0,0,4,0,0]),
                  'feynman_II_34_2': torch.tensor([0,0,3,0,0])
                  }

    perfect_mp_rat = {'feynman_III_15_14': torch.tensor([0,0,2,3,0]),
                    'feynman_III_15_27': torch.tensor([0,0,1,2,0]),
                    'feynman_III_7_38': torch.tensor([0,0,2,1,0]),
                    'feynman_II_10_9': torch.tensor([0,0,1,2,0])
                    }

    perfect_mp_cos = {'feynman_III_15_12': torch.tensor([2,0,2,0,1]),
                    'feynman_III_17_37': torch.tensor([1,0,3,0,1]),
                    'feynman_II_15_4': torch.tensor([1,0,3,0,1]),
                    'feynman_II_15_5': torch.tensor([1,0,3,0,1]),
                    'feynman_III_8_54': torch.tensor([2,1,1,0,1])} | \
                    perfect_mp_pol | perfect_mp_rat 

    perfect_mp_sqrt = {'feynman_II_13_23': torch.tensor([2,2,1,1,1]),
                    'feynman_II_13_34': torch.tensor([2,2,2,1,1]),
                    'feynman_I_10_7': torch.tensor([2,2,1,1,1]),
                    'feynman_I_47_23': torch.tensor([2,1,1,0,1]),
                    'feynman_I_48_2': torch.tensor([2,2,3,1,1])
                    } | \
                    perfect_mp_pol | perfect_mp_rat 
                    
    perfect_mp_exp = {'feynman_I_6_2b': torch.tensor([2,2,1,1,1]),
                    } | \
                    perfect_mp_pol | perfect_mp_rat 
                    

    perfect_mp_full = {key: torch.cat([perfect_mp_cos[key], torch.zeros(2)]) for key in perfect_mp_cos.keys()} | \
                    {key: torch.cat([perfect_mp_sqrt[key], torch.zeros(2)])[[0,1,2,3,5,4,6]] for key in perfect_mp_sqrt.keys()} | \
                    {key: torch.cat([perfect_mp_exp[key], torch.zeros(2)])[[0,1,2,3,5,6,4]] for key in perfect_mp_exp.keys()}

    perfect_mp_full_4 = {'feynman_III_10_19': torch.tensor([2,0,2,0,0,1,0]),
                        'feynman_III_13_18': torch.tensor([0,0,4,1,0,0,0]),
                        'feynman_III_21_20': torch.tensor([0,0,3,1,0,0,0]),
                        'feynman_III_4_32': torch.tensor([2,2,1,1,0,0,1]),
                        'feynman_III_4_33': torch.tensor([2,2,2,1,0,0,1]),
                        'feynman_II_11_27': torch.tensor([0,0,4,2,0,0,0]),
                        'feynman_II_34_11': torch.tensor([0,0,3,1,0,0,0]),
                        'feynman_II_6_11': torch.tensor([1,0,2,3,1,0,0])
                        }

    perfect_mp_full = perfect_mp_full | perfect_mp_full_4

def predict_feynman(net, dataset_name, config, k, max_dataset_length, function, dimension, verbose=True, perfect_mp_arr=None):
    data_parameters_dict = dict(config.items("DATAPARAMETERS"))
    data_parameters_dict = {key: ast.literal_eval(data_parameters_dict[key]) for key in
                            data_parameters_dict.keys()}
    
    if function == 'Full':
        assert data_parameters_dict['function_names_str'][0] == 'cos'
        assert data_parameters_dict['function_names_str'][1] == 'sqrt'
        assert data_parameters_dict['function_names_str'][2] == 'exp'
    
    device = 'cpu'
    normalization = 'DataSetWiseXY-11'

    x, y, target_expr, switch = data_feynmann(dataset_name, max_dataset_length=max_dataset_length)
    
    if normalization == 'DataSetWiseXY-11':
        y = utils.min_max_normalization(torch.tensor(y, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(-1), dim=1, new_min=-1, new_max=1)
        x = utils.min_max_normalization(torch.tensor(x, device=device, dtype=torch.float).unsqueeze(0), dim=1, new_min=-1, new_max=1)
    
    
    data = torch.cat([x, y], dim=-1) # (dataset_size=1, num_samples, dimension + 1)
    print(f'Data shape: {data.shape}')
    if dimension == '1-9':
        data = torch.cat([torch.zeros(data.shape[0], data.shape[1], 10-data.shape[2], device=data.device, dtype=data.dtype), data], dim=-1)
        print(f'Data shape: {data.shape}')
    
    prediction_one_hot_softmax = net(data, return_logits=False)
    prediction_one_hot_logits = net(data, return_logits=True)
    prediction = utils.one_hot_to_categorical(prediction_one_hot_logits, data_parameters_dict)
        
    index = 0
    for category in ['degree_input_numerator', 'degree_input_denominator', 'degree_output_numerator', 'degree_output_denominator', 'function_names_str']:
        if category != 'function_names_str':
            length = data_parameters_dict[category] + 1
        else:
            length = len(data_parameters_dict[category]) + 1
        print(f'{category}: {index}-{index+length-1}')
        index += length            
    
    
    if perfect_mp_arr is None:
        if dimension==3:
            if function == 'cos':
                perfect_mp = perfect_mp_cos 
            elif function == 'all':
                perfect_mp = perfect_mp_full
        else:
            perfect_mp = perfect_mp_full_4
        
        if dimension =='1-9':
            perfect_mp = perfect_mp_full
        
        if dataset_name in perfect_mp.keys():
            perfect_mp_arr = perfect_mp[dataset_name]
    
    print(f'Target expression: {target_expr}')
    if not (perfect_mp_arr is None):
        print(f'Perfect model parameters: \n {perfect_mp_arr}')
    print(f'Prediction one hot: \n {prediction_one_hot_softmax}')
    print(f'Prediction categorical: \n {prediction}')
    
    topk_predictions = get_topk_predictions(data_parameters_dict, prediction_one_hot_logits[0], k=k)
    topk_predictions = filter_predictions(topk_predictions, function)

    print(f'Top {k} predictions categorical:')

    top_k_index = np.inf
    for i in range(topk_predictions.shape[0]):
        if (not (perfect_mp_arr is None)) and (torch.tensor(perfect_mp_arr)==topk_predictions[i]).all():
            print(f'{i+1}. {topk_predictions[i]} ==> Perfect model parameters')
            top_k_index = i+1
        else:
            if verbose:
                print(f'{i+1}. {topk_predictions[i]}')
    return top_k_index
    
def single_prediction(net, config, function, dimension):
    # 3d: 
    # option 1
    # (containing cos) feynman_III_15_12 (top 8), feynman_III_17_37 (top 5), feynman_II_15_4 (top 5), feynman_II_15_5 (top 1), feynman_III_8_54 (top 2)
    # (purely rational) feynman_III_15_14 (top 2), feynman_III_15_27 (top 4), feynman_III_7_38 (top 3), feynman_II_10_9 (top 1)
    # (purely polonomial) feynman_II_27_16 (top 1), feynman_II_34_2 (top 1)
    # option 2
    # (containing cos) feynman_III_15_12 (top 10), feynman_III_17_37 (top 1), feynman_II_15_4 (top 7), feynman_II_15_5 (top 1), feynman_III_8_54 (top 2)    
    # (purely rational) feynman_III_15_14 (top 1), feynman_III_15_27 (top 1), feynman_III_7_38 (top 1), feynman_II_10_9 (top 1), 
    # (purely polonomial) feynman_II_27_16 (top 1), feynman_II_34_2 (top 1)
    # option 3
    # (containing cos) feynman_III_15_12 (top 20), feynman_III_17_37 (top 3), feynman_II_15_4 (top 3), feynman_II_15_5 (top 2), feynman_III_8_54 (top 4)    
    # (purely rational) feynman_III_15_14 (top 519), feynman_III_15_27 (top 59), feynman_III_7_38 (top 538), feynman_II_10_9 (top 497), 
    # (purely polonomial) feynman_II_27_16 (top 91), feynman_II_34_2 (top 253)
    # (containing sqrt) feynman_II_13_23 (top 59), feynman_II_13_34 (top 62), feynman_I_10_7 (top 72), feynman_I_47_23 (top 2), feynman_I_48_2 (top 58)
    # (containing exp) feynman_I_6_2b (top 40)
    # option 5
    # (containing cos) feynman_III_15_12 (top 19), feynman_III_17_37 (top 4), feynman_II_15_4 (top 8), feynman_II_15_5 (top 18), feynman_III_8_54 (top 2)    
    # (purely rational) feynman_III_15_14 (top 519), feynman_III_15_27 (top 59), feynman_III_7_38 (top 538), feynman_II_10_9 (top 497), 
    # (purely polonomial) feynman_II_27_16 (top 91), feynman_II_34_2 (top 253)
    # (containing sqrt) feynman_II_13_23 (top 59), feynman_II_13_34 (top 62), feynman_I_10_7 (top 72), feynman_I_47_23 (top 2), feynman_I_48_2 (top 58)
    # (containing exp) feynman_I_6_2b (top 40)
    
    # 4d
    # option 4
    # (containing cos) feynman_II_6_11 (top 251)
    # (containing sqrt) feynman_III_10_19 (top 7)
    # (containing exp) feynman_III_4_32 (top 244), feynman_III_4_33 (top 98)
    # (purely rational) feynman_III_13_18 (top 1), feynman_III_21_20 (top 16), feynman_II_11_27 (top 393), feynman_II_34_11 (top 1)
    dataset_name = 'feynman_II_15_5'
    predict_feynman(net, dataset_name, config, k=20, max_dataset_length=200, function=function, dimension=dimension)
    
def multiple_prediction(net, config, function, dimension):
    seed = 12345
    # The results will depend on the order and will be different from the results
    cos = ['feynman_III_15_12', 'feynman_III_17_37', 'feynman_II_15_4', 'feynman_II_15_5', 'feynman_III_8_54', 'feynman_II_6_11']
    cos_indices = []
    for dataset_name in cos:
        # Set a seed, since the results depend otherwise on the order and won't fit to the previously reported results anymore
        np.random.seed(seed)
        cos_indices.append(predict_feynman(net, dataset_name, config, k=700, max_dataset_length=200, function=function, dimension=dimension))
    
    sqrt = ['feynman_II_13_23', 'feynman_II_13_34', 'feynman_I_10_7', 'feynman_I_47_23', 'feynman_I_48_2', 'feynman_III_10_19']
    sqrt_indices = []
    for dataset_name in sqrt:
        # Set a seed, since the results depend otherwise on the order and won't fit to the previously reported results anymore
        np.random.seed(seed)
        sqrt_indices.append(predict_feynman(net, dataset_name, config, k=700, max_dataset_length=200, function=function, dimension=dimension))
        
    exp = ['feynman_I_6_2b', 'feynman_III_4_32', 'feynman_III_4_33']
    exp_indices = []
    for dataset_name in exp:
        # Set a seed, since the results depend otherwise on the order and won't fit to the previously reported results anymore
        np.random.seed(seed)
        exp_indices.append(predict_feynman(net, dataset_name, config, k=700, max_dataset_length=200, function=function, dimension=dimension))
    
    rationals = ['feynman_III_15_14', 'feynman_III_15_27', 'feynman_III_7_38', 'feynman_II_10_9', 'feynman_III_13_18', 'feynman_III_21_20', 'feynman_II_11_27', 'feynman_II_34_11']
    rationals_indices = []
    for dataset_name in rationals:
        # Set a seed, since the results depend otherwise on the order and won't fit to the previously reported results anymore
        np.random.seed(seed)
        rationals_indices.append(predict_feynman(net, dataset_name, config, k=700, max_dataset_length=200, function=function, dimension=dimension))
    
    polynomials = ['feynman_II_27_16', 'feynman_II_34_2']
    polynomials_indices = []
    for dataset_name in polynomials:
        # Set a seed, since the results depend otherwise on the order and won't fit to the previously reported results anymore
        np.random.seed(seed)
        polynomials_indices.append(predict_feynman(net, dataset_name, config, k=700, max_dataset_length=200, function=function, dimension=dimension))
    
    
    
    print(cos_indices) # (1M) [19, 4, 8, 18, 2, 22]  # (5M) [10, 1, 1, 1, 2, 1]
    print(sqrt_indices) # [93, 16, 104, 17, 53, 1] # (5M) [121, 46, 165, 1, 35, 1]
    print(exp_indices) # [137, 280, 137] # (5M) [274, 349, 125]
    print(rationals_indices) # [143, 194, 375, 277, 4, 55, 134, 185] # (5M) [1, 1, 1, 145, 1, 1, 386, 1]
    print(polynomials_indices) # [3, 117] # (5M) [8, 1]

# Define a custom function to handle the parsing of lists
def parse_list(s):
    # Remove square brackets and split the string by commas
    s = s.strip('[]')
    parts = s.split(',')
    # Convert each part to integer
    return [int(part) for part in parts]
    
def evaluate_on_whole_feynman(net, config):
    feynman = {'id': [], 'dataset_name': [], 'target_formula': [], 'model_parameter': [], 'top_k': [], 'covered': [], 'possible': []}
    
    with open('datasets_feynmann_with_modelparameters.csv', 'r') as file:
        lines = file.readlines()

    
    for line in lines:
        
        if len(line.split(',')) < 10:
            id, dataset_name, target_formula, model_parameter =  line.split(',')
            feynman['top_k'].append(np.inf)
            feynman['covered'].append(False)
            feynman['possible'].append(False)
        else:
            
            id, dataset_name, target_formula, model_parameter1, model_parameter2, model_parameter3, \
                model_parameter4, model_parameter5, model_parameter6, model_parameter7 = line.split(',')
                
            model_parameter = [model_parameter1[1], model_parameter2, model_parameter3, model_parameter4, model_parameter5, model_parameter6, model_parameter7[0]]
            model_parameter = [int(mp) for mp in model_parameter]
            
            k = predict_feynman(net, dataset_name, config, k=10000, max_dataset_length=200, function='all', dimension='1-9', verbose=False, perfect_mp_arr=model_parameter)
            
            covered = k < 10000
            
            feynman['top_k'].append(k)
            feynman['covered'].append(covered)
            feynman['possible'].append(True)
        
        feynman['id'].append(id)
        feynman['dataset_name'].append(dataset_name)
        feynman['target_formula'].append(target_formula)
        feynman['model_parameter'].append(model_parameter)
    
    return pd.DataFrame(feynman)

if __name__ == '__main__':
    np.random.seed(12345)
    
    setup_df()
    
    option = 6
    if option==1:
        # 3d cos 100,000
        dimension = 3
        function = 'cos'
        path = 'trainingOnSyntheticData/results/Set nd/3d/single function full/20240304_153746_train_model_parameters'
        net_name = 'Datetime_20240304_154136_Loss_training_set_size_96432_batch_size_512_hidden_dim_256.pt'
    elif option==2:
        # 3d cos 1,000,000
        dimension = 3
        function = 'cos'
        path = 'trainingOnSyntheticData/results/Set nd/3d/single function full/20240305_100534_cos_1000000'
        net_name = 'Datetime_20240305_103817_Loss_training_set_size_968880_batch_size_256_hidden_dim_256.pt'
    elif option==3:
        # 3d cos, sqrt, exp
        dimension = 3
        function = 'all'
        path = 'trainingOnSyntheticData/results/Set nd/20240306_175222_1_3_5_7_9_full_100000'
        net_name = 'Datetime_20240306_231521_Loss_training_set_size_83604_batch_size_256_hidden_dim_256.pt'
    elif option==4:
        # 4d cos, sqrt, exp
        dimension = 4
        function = 'all'
        path = 'trainingOnSyntheticData/results/Set nd/20240306_175022_4_full_1000000'
        net_name = 'Datetime_20240306_182904_Loss_training_set_size_803230_batch_size_256_hidden_dim_256.pt'
    elif option==5:
        # 4d cos, sqrt, exp
        dimension = '1-9'
        function = 'all'
        path = 'trainingOnSyntheticData/results/Flexible dimension/full_19_small_training_sets/20240424_145058_no_wu'
        net_name = 'Datetime_20240424_151254_Loss_training_set_size_77488_batch_size_256_hidden_dim_256.pt'
    elif option==6:
        # 4d cos, sqrt, exp
        dimension = '1-9'
        function = 'all'
        path = 'trainingOnSyntheticData/results/Flexible dimension/full_19_big_training_sets/20240427_070711_5M'
        net_name = 'Datetime_20240427_070717_Loss_training_set_size_797545_batch_size_341_hidden_dim_256.pt'
    # For the config file
    config_path = os.path.join(path, 'train_model_parameters.ini')
    config = configparser.ConfigParser()
    config.read(config_path)

    net_path = os.path.join(path, net_name)
    net = torch.load(net_path, map_location=torch.device('cpu'))
    net.eval()
    if not hasattr(net, 'use_linear_embedding'):
        net.use_linear_embedding = False

    # learning_random_functions(net)

    # single_prediction(net, config, function, dimension)
    # multiple_prediction(net, config, function, dimension)
    evaluate_on_whole_feynman(net, config)
