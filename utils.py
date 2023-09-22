import os
import pandas as pd
import numpy as np
from itertools import product
import sympy
# from torcheval.metrics import R2Score
# from torchmetrics import R2Score

import torch
from torch import nn
import torch.nn.functional as F
from parfam_torch import ParFamTorch

# Gets real formula of regression problem: only works for feynman so far
# feynman_formulas = pd.read_csv(os.path.join(os.path.dirname(__file__), "feynman_formulas"), names=['Index', 'Name', 'Formula'])
feynman_formulas = pd.read_csv(os.path.join(os.path.dirname(__file__), "datasets"), names=['Index', 'Name', 'Formula'])


def is_tuple_sorted(t):
    for i in range(1, len(t)):
        # return False if the element is smaller than the previous element
        if t[i] < t[i - 1]:
            return False
    return True


def get_formula(name):
    formula = feynman_formulas[feynman_formulas["Name"] == name]["Formula"].values
    if formula.size > 0:
        formula = formula[0]
    else:
        print("No target expression given for " + name)  # only feynman so far
    return formula


# Gets relative l2 error with given model
def relative_l2_distance(model, coefficients, x, y):
    y_pred = model.predict(coefficients, x)
    relative_l2_distance = torch.norm(y - y_pred, p=2) / torch.norm(y - y.mean(), p=2)
    return relative_l2_distance


def r_squared(model, coefficients, x, y):
    y_pred = model.predict(coefficients, x)
    r2score = 1 - sum((y - y_pred) ** 2) / sum((y - y.mean()) ** 2)
    return r2score


# adds results to the results dictionary
def append_results_dict_model_training_parameters(results_dict, training_parameters, model_parameters_dict,
                                                  target_formula, estimated_formula,
                                                  relative_l2_distance_train, relative_l2_distance_val, r_squared_val,
                                                  success, training_time, dataset_name, n_active_coefficients,
                                                  relative_l2_distance_test,
                                                  r_squared_test, n_active_coefficients_reduced,
                                                  relative_l2_distance_test_reduced, r_squared_test_reduced,
                                                  best_formula_reduced, r_squared_val_reduced, n_evaluations):
    relative_l2_distance_train = relative_l2_distance_train.item()
    relative_l2_distance_val = relative_l2_distance_val.item()
    relative_l2_distance_test = relative_l2_distance_test.item()
    r_squared_val = r_squared_val.item()
    r_squared_test = r_squared_test.item()
    relative_l2_distance_test_reduced = relative_l2_distance_test_reduced.item()
    r_squared_test_reduced = r_squared_test_reduced.item()
    r_squared_val_reduced = r_squared_val_reduced.item()
    success = success.item()
    n_active_coefficients = n_active_coefficients.item()
    n_active_coefficients_reduced = n_active_coefficients_reduced.item()
    hyperparameters = {**training_parameters, **model_parameters_dict}
    append_results_dict(results_dict, hyperparameters, target_formula, estimated_formula, relative_l2_distance_train,
                        relative_l2_distance_val, r_squared_val, success, training_time, dataset_name,
                        n_active_coefficients, relative_l2_distance_test, r_squared_test, n_active_coefficients_reduced,
                        relative_l2_distance_test_reduced, r_squared_test_reduced,
                        best_formula_reduced, r_squared_val_reduced, n_evaluations)


# adds results to the results dictionary
def append_results_dict(results_dict, hyperparameters, target_formula, estimated_formula, relative_l2_distance_train,
                        relative_l2_distance_val, r_squared_val, success, training_time, dataset_name,
                        n_active_coefficients, relative_l2_distance_test, r_squared_test, n_active_coefficients_reduced,
                        relative_l2_distance_test_reduced, r_squared_test_reduced, best_formula_reduced,
                        r_squared_val_reduced, n_evaluations):
    for key in hyperparameters.keys():
        results_dict[key].append(hyperparameters[key])
    results_dict['target_formula'].append(str(target_formula))
    results_dict['estimated_formula'].append(estimated_formula)
    results_dict['relative_l2_train'].append(relative_l2_distance_train)
    results_dict['relative_l2_val'].append(relative_l2_distance_val)
    results_dict['r_squared_val'].append(r_squared_val)
    results_dict['success'].append(success)
    results_dict['training_time'].append(np.round(training_time, 3))
    results_dict['dataset'].append(dataset_name)
    results_dict['n_active_coefficients'].append(n_active_coefficients)
    results_dict['relative_l2_test'].append(relative_l2_distance_test)
    results_dict['r_squared_test'].append(r_squared_test)
    results_dict['n_active_coefficients_reduced'].append(n_active_coefficients_reduced)
    results_dict['relative_l2_distance_test_reduced'].append(relative_l2_distance_test_reduced)
    results_dict['r_squared_test_reduced'].append(r_squared_test_reduced)
    results_dict['r_squared_val_reduced'].append(r_squared_val_reduced)
    results_dict['best_formula_reduced'].append(best_formula_reduced)
    results_dict['n_evaluations'].append(n_evaluations)


# Adds noise accordingly to features x and target y
def add_noise(x, y, target_noise, feature_noise):
    # Add noise to target
    if target_noise > 0:
        print('adding', target_noise, 'noise to target')
        var = target_noise * torch.norm(y, p=2) / np.sqrt(len(y))
        y = y + torch.randn(y.shape, device=y.device) * var

    # if target_noise > 0:
    #    print('adding', target_noise, 'noise to target')
    #    var = target_noise * np.linalg.norm(y.numpy(), ord=2)
    #    y = y + np.random.randn(y.shape[0]) * var

    # Add noise to the features
    if feature_noise > 0:
        print('adding', feature_noise, 'noise to features')
        var = feature_noise * torch.norm(x, p=2) / np.sqrt(len(x))
        x = x + torch.randn(x.shape, device=x.device) * var
    return x, y


# Gets all combinations of hyperparameters given in hp_dict and return a list
def get_hyperparameters_combination(hp_dict):
    iterables = [value if isinstance(value, list) else [value] for value in hp_dict.values()]
    all_combinations = list(product(*iterables))
    # Create a list of dictionaries for each combination
    combination_dicts = [{param: value for param, value in zip(hp_dict.keys(), combination)} for combination in
                         all_combinations]
    return combination_dicts


def round_expr(expr, num_digits):
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(sympy.Number)})


def experiment_data(n_samples, experiment_name):
    if experiment_name == 'exp':
        x = 20 * np.random.randn(n_samples, 2)
        a = np.random.rand() * 5

        def func(x, module):
            if module == sympy:
                x0 = sympy.Symbol('x0')
                x1 = sympy.Symbol('x1')
            else:
                x0 = x[:, 0]
                x1 = x[:, 1]
            return module.exp(a * (x0 + x1))

        y = func(x, np)
        target_expr = func(x, sympy)
        target_expr = round_expr(target_expr, 3)
    elif experiment_name == 'expx0/x1':
        x = 2 * np.random.randn(n_samples, 2) + 5  # ensure they are positive and away from zero
        a = np.random.rand() * 5

        def func(x, module):
            if module == sympy:
                x0 = sympy.Symbol('x0')
                x1 = sympy.Symbol('x1')
            else:
                x0 = x[:, 0]
                x1 = x[:, 1]
            return module.exp(a * (x0 / x1))
    elif experiment_name == 'test':
        x = 3 * np.linspace(-2 * np.pi, 2 * np.pi, n_samples)
        x = x.reshape((n_samples, 1))
        a = np.random.rand()

        def func(x, module):
            if module == sympy:
                x0 = sympy.Symbol('x0')
                # x1 = sympy.Symbol('x1')
            else:
                x0 = x[:, 0]
                # x1 = x[:,1]
            return module.cos(3 * x0) + module.sin(5 * x0) - x0 ** 2

        y = func(x, np)
        target_expr = func(x, sympy)
        target_expr = round_expr(target_expr, 3)
    else:
        raise NotImplementedError(f'Experiment {experiment_name} not implemented yet.')

    return x, y, target_expr


class ASL:
    def __init__(self, gamma_negative, p_target, update):
        self.gamma_negative = gamma_negative
        self.p_target = p_target
        self.update = update
        self.stepsize = 0.01
        self.epsilon = 0.0001

    def asl(self, x, y):
        if self.update:
            # gamma_positive = 0
            p_neg = (1 - x[y == 0].mean())
            p_pos = x[y == 1].mean()
            p_del = p_pos - p_neg
            self.gamma_negative = self.gamma_negative + self.stepsize * (p_del.detach() - self.p_target)
            self.gamma_negative = max(0, self.gamma_negative)
        loss = - (y * torch.log(x + self.epsilon) + (1 - y) * (x) ** self.gamma_negative * torch.log(
            1 - x + self.epsilon)).mean()
        if torch.isnan(loss):
            raise ValueError('NAN')
        return loss


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


def covering_loss(prediction, target):
    loss = - (y * torch.log(x)).mean()
    # loss +=


def create_dataset(dataset_size, ensure_uniqueness, degree_input_polynomials, degree_output_polynomials, width,
                   functions, function_names, normalization, degree_output_polynomials_specific, n_functions_max,
                   device='cpu', enforce_function=False):
    target_model = ParFamTorch(n_input=1, degree_input_numerator=degree_input_polynomials,
                               degree_output_numerator=degree_output_polynomials, width=width,
                               functions=functions, function_names=function_names, enforce_function=enforce_function,
                               degree_output_numerator_specific=degree_output_polynomials_specific)
    n_params = target_model.get_number_parameters()

    x = np.arange(-10, 10, 0.1).reshape(-1, 1)
    a = np.zeros((dataset_size, n_params))
    params = torch.randn(dataset_size, n_params) * 3

    for i in range(dataset_size):
        if ensure_uniqueness:
            a[i] = target_model.get_random_coefficients_unique(n_functions_max=n_functions_max)
        else:
            a[i] = target_model.get_random_coefficients(n_functions_max=n_functions_max)

    params = params * a
    if dataset_size > 1:
        y = target_model.predict_batch(params.T, x).T
    else:
        y = target_model.predict(torch.tensor(params, dtype=torch.float, device=device),
                                 torch.tensor(x, dtype=torch.float, device=device))
    #    params[i] = params[i] * a[i]
    #    y[i] = target_model.predict(params[i], x)
    a = torch.tensor(a, dtype=torch.float, device=device).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float, device=device)
    if normalization == 'No':
        pass
    elif normalization == 'Full':
        y = F.normalize(y)
    return y, a, params
