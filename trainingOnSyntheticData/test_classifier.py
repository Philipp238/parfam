import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import dual_annealing, basinhopping, differential_evolution

from pmlb import fetch_data

import sympy
from sympy import symbols
from time import time
import numpy as np
import os
import sys
import pandas as pd

# from srbench.utils import get_formula
sys.path[0] = os.getcwd()
print(os.getcwd())
from utils import create_dataset
from parfam_torch import ParFamTorch, Evaluator

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
    model = ParFamTorch(n_input=x.shape[1], degree_input_polynomials=degree_input_polynomials,
                        degree_output_polynomials=degree_output_polynomials, width=width,
                        functions=functions, degree_output_denominator=degree_output_denominator,
                        degree_input_denominator=degree_input_denominator,
                        function_names=function_names,
                        degree_output_polynomials_specific=degree_output_polynomials_specific,
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
    model = ParFamTorch(n_input=x.shape[1], degree_input_polynomials=degree_input_polynomials,
                        degree_output_polynomials=degree_output_polynomials, width=width,
                        functions=functions, degree_output_denominator=degree_output_denominator,
                        degree_input_denominator=degree_input_denominator,
                        function_names=function_names,
                        degree_output_polynomials_specific=degree_output_polynomials_specific,
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


def data_feynmann(function_name):
    path_pmlb_datasets = os.path.join('/', 'datasets_with_notes')
    x, y = fetch_data(function_name, return_X_y=True, local_cache_dir=path_pmlb_datasets)
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y).astype(np.float64)
    target_formula = get_formula(function_name)
    if len(y) > 1000:
        subset = np.random.choice(len(y), 1000, replace=False)
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


def covering(net, quantile):
    # function_names = ['1', '2', '3']
    function_names_data = ['5']
    n_iterations = 40
    successes = 0
    for _ in range(n_iterations):
        for function_name_data in function_names_data:
            x, y, target_expr, switch = data(function_name_data)
            print(f'########## Learning the following function: {target_expr} ##########')
            # standard_training(x, y, target_expr)
            prediction = net(torch.tensor(y, dtype=torch.float, device='cpu')).cpu().detach().numpy()
            if quantile:
                cutoff = np.quantile(prediction, 0.7)
            else:
                cutoff = 0.2
            switch_pred = (prediction > cutoff).astype(int)

            complete_covering = np.sum(switch > switch_pred) == 0

            if complete_covering:
                successes += 1
                print('Complete cover')
            else:
                print('Not complete cover')

            functions = [np.sin, np.cos]
            function_names = [sympy.sin, sympy.cos]
            model = ParFamNP(n_input=1, degree_input_polynomials=2, degree_output_polynomials=2, width=1,
                             functions=functions,
                             function_names=function_names)
            n_params = model.get_number_parameters()
            symbolic_a = [symbols(f'a{i}') if switch_pred[i] else 0 for i in range(n_params)]
            symbolic_expr = model.get_formula(symbolic_a, verbose=False)
            print(f'Reduced formula: {symbolic_expr}')
    print(f'Average amount of successes: {successes / n_iterations}')


def covering_new(net, quantile):
    test_set_size = 20
    functions = [np.sin, np.cos]
    function_names = [sympy.sin, sympy.cos]
    degree_output_polynomials = 2
    degree_input_polynomials = 2
    width = 1

    np.random.seed(12345)
    y, a = create_dataset(test_set_size, True, degree_input_polynomials,
                          degree_output_polynomials,
                          width, functions, function_names)
    successes = 0
    for i in range(test_set_size):
        model = ParFamNP(n_input=1, degree_input_polynomials=degree_input_polynomials,
                         degree_output_polynomials=degree_output_polynomials, width=width,
                         functions=functions, function_names=function_names)
        n_params = model.get_number_parameters()

        prediction = net(torch.tensor(y[i], dtype=torch.float, device='cpu')).cpu().detach().numpy()
        if quantile:
            cutoff = np.quantile(prediction, 0.7)
        else:
            cutoff = 0.2
        switch_pred = (prediction > cutoff).astype(int)

        symbolic_a = [symbols(f'a{i}') if switch_pred[i] else 0 for i in range(n_params)]
        symbolic_expr = model.get_formula(symbolic_a, verbose=False)
        target_expr = model.get_formula(a[i], verbose=False)
        print(f'Target formula: {target_expr}')
        print(f'Reduced formula: {symbolic_expr}')

        complete_covering = np.sum(a[i].squeeze(1).cpu().detach().numpy() > switch_pred) == 0

        if complete_covering:
            successes += 1
            print('Complete cover')
        else:
            print('Not complete cover')

    print(f'Average amount of successes: {successes / test_set_size}')


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


if __name__ == '__main__':
    np.random.seed(12345)

    # For the network
    dataset_size = 2000000
    hidden_neurons = 200
    n_batches = 500
    net_path = (f'trainingOnSyntheticData/trained_models/20230807_213046_train/model_training_set_size_{dataset_size}_'
                f'hidden_neurons_{hidden_neurons}_n_batches_{n_batches}.pt')

    # net_path = (f'trainingOnSyntheticData/trained_models/20230804_110046_train_local/'
    #            f'model_training_set_size_100000_hidden_neurons_100_n_batches_100.pt')

    # net_path = f'optimization/trainingOnSyntheticData/trained_models/20230510_171529_train/model_training_set_size_' \
    #            f'2000000_hidden_neurons_200_n_batches_100.pt'

    net = torch.load(net_path, map_location=torch.device('cpu'))

    learning_random_functions(net)
    # covering(net, quantile=False)
