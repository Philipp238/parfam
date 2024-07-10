import time
import os
import pandas as pd
import numpy as np
import itertools
from itertools import product
import torch
import sympy
# from torcheval.metrics import R2Score
# from torchmetrics import R2Score

import torch
from torch import nn
import torch.nn.functional as F
from parfam_torch import ParFamTorch
import math
import logging

import resource
import psutil
import copy

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    # you can convert that object to a dictionary 
    return f'{point}: mem (CPU python)={usage[2]/1024.0}MB; mem (CPU total)={dict(psutil.virtual_memory()._asdict())["used"] / 1024**2}MB'

# Gets real formula of regression problem: only works for feynman so far
# feynman_formulas = pd.read_csv(os.path.join(os.path.dirname(__file__), "feynman_formulas"), names=['Index', 'Name', 'Formula'])
feynman_formulas = pd.read_csv(os.path.join(os.path.dirname(__file__), "datasets"), names=['Index', 'Name', 'Formula'])

def sqrt(x):
    return torch.sqrt(torch.abs(x))

def exp(x):
    return torch.minimum(torch.exp(x), np.exp(10) + torch.abs(x))

def log(x):
    return torch.log(torch.abs(x) + 0.000001)

def sqrt_sympy(x):
    return sympy.sqrt(sympy.Abs(x))

def log_sympy(x):
    return sympy.log(sympy.Abs(x) + 0.000001)

function_dict = {'sqrt': sqrt, 'exp': exp, 'log': log, 'cos': torch.cos, 'sin': torch.sin}
function_name_dict = {'sqrt': sqrt_sympy, 'exp': sympy.exp, 'cos': sympy.cos, 'sin': sympy.sin, 'log': log_sympy}


def _list_index(len_list, len_prob, index):
    return [1 if i!=index else len_prob for i in range(len_list)]

def compute_comb_product(list_probs):
    # compute the product from a list of tensors in which each element in each tensor is multiplied with each element of the other tensors
    # Usage: Compute the probabilities of all different label sets
    n_probs = len(list_probs)
    list_probs_unsqueezed = []
    for i, prob in enumerate(list_probs):
        prob_unsqueezed = prob.reshape(_list_index(n_probs, len(prob), i))
        list_probs_unsqueezed.append(prob_unsqueezed)
    prod = 1
    for prob in list_probs_unsqueezed:
        prod = prod * prob
    return prod

def label_smoothing(labels, alpha, data_parameters):
    if alpha > 0:
        lengths = [data_parameters['degree_input_numerator'], data_parameters['degree_input_denominator'], data_parameters['degree_output_numerator'], 
            data_parameters['degree_output_denominator'], len(data_parameters['function_names_str'])]
        # Need for each polynomial 1+max_degree variables, unless the max_degree is 0; Same for the functions (+1 in case there is no function used)
        lengths = [length + 1 if length > 0 else 0 for length in lengths]  
        index = 0
        for i, length in enumerate(lengths):
            if length > 0:
                labels[:,index:index+length] = (1-alpha)*labels[:,index:index+length] + alpha / length
                index += length
    return labels

def categorical_to_one_hot(target_categorical, data_parameters):
    # degree_input_numerator, degree_input_denominator, degree_output_numerator, degree_output_denominator, function1 (cos), function2 (exp), function3 (sqrt)
    lengths = [data_parameters['degree_input_numerator'], data_parameters['degree_input_denominator'], data_parameters['degree_output_numerator'], 
               data_parameters['degree_output_denominator'], len(data_parameters['function_names_str'])]
    # Need for each polynomial 1+max_degree variables, unless the max_degree is 0; Same for the functions (+1 in case there is no function used)
    lengths = [length + 1 if length > 0 else 0 for length in lengths]  
    # Note that this assumes that there are never 2 functions used at the same time ==> The next check
    target_one_hot = torch.zeros(target_categorical.shape[0], sum(lengths), dtype=target_categorical.dtype, device=target_categorical.device)
    target_categorical = target_categorical.to(int)
    index = 0
    for i, length in enumerate(lengths[:-1]):
        if length > 0:
            target_one_hot[:,index:index+length] = F.one_hot(target_categorical[:, i], num_classes=length)
            index += length
    # target_one_hot[:,:sum(lengths[:-1])] = F.one_hot(target[:, :4][:, mask]).flatten(1)
    
    if len(data_parameters['function_names_str']):
        target_one_hot[:,sum(lengths[:-1])] = target_categorical[:, 4:].sum(dim=1) == 0
        target_one_hot[:,sum(lengths[:-1])+1:] = target_categorical[:, 4:]
    return target_one_hot
    
def one_hot_to_categorical(target_one_hot, data_parameters):
    n_functions = len(data_parameters['function_names_str'])
    target_categorical = torch.zeros(target_one_hot.shape[0], 4 + n_functions, dtype=target_one_hot.dtype, device=target_one_hot.device)
    # Use for each polynomial 1+max_degree variables, unless the max_degree is 0; Same for the functions (+1 in case there is no function used)
    lengths = [data_parameters['degree_input_numerator'], data_parameters['degree_input_denominator'], data_parameters['degree_output_numerator'], 
               data_parameters['degree_output_denominator'], len(data_parameters['function_names_str'])]
    lengths = [length + 1 if length > 0 else 0 for length in lengths]  

    index = 0
    for i, length in enumerate(lengths[:-1]):
        if length > 0:
            target_categorical[:,i] = torch.argmax(target_one_hot[:,index:index+length], dim=1)
        index += length
    
    if n_functions>0:
        active_function = torch.argmax(target_one_hot[:, -(n_functions+1):], dim=1)
        for i in range(n_functions):
            target_categorical[:, 4+i] = active_function == i+1
    return target_categorical

def get_complete_model_parameter_list(model_parameters_max, iterate_polynomials=False):
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
    if iterate_polynomials:
        for d_output in range(0, max_deg_output + 1):
            model_parameters = {'degree_input_polynomials': 0, 'degree_output_polynomials': d_output, 'functions': [],
                                'function_names': [], 'width': 1, 'degree_input_denominator': 0,
                                'degree_output_denominator': 0,
                                'degree_output_polynomials_specific': None,
                                'degree_output_polynomials_denominator_specific': None,
                                'enforce_function': False, 'maximal_potence': model_parameters_max['maximal_potence']}
            model_parameter_list.append(model_parameters)
    else:
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

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

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
    if isinstance(model, ParFamTorch):
        y_pred = model.predict(coefficients, x)
    else:
        y_pred = model.predict(x)
    
    if isinstance(x, np.ndarray):
        # numpy arrays
        relative_l2_distance = np.linalg.norm(y - y_pred, ord=2) / np.linalg.norm(y - y.mean(), ord=2)
    else:
        # torch tensors
        relative_l2_distance = torch.norm(y - y_pred, p=2) / torch.norm(y - y.mean(), p=2)
    return relative_l2_distance


def r_squared(model, coefficients, x, y):
    if isinstance(model, ParFamTorch):
        y_pred = model.predict(coefficients, x)
    else:
        y_pred = model.predict(x)
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
        # print('adding', target_noise, 'noise to target')
        var = target_noise * torch.norm(y, p=2) / np.sqrt(len(y))
        y = y + torch.randn(y.shape, device=y.device) * var

    # if target_noise > 0:
    #    print('adding', target_noise, 'noise to target')
    #    var = target_noise * np.linalg.norm(y.numpy(), ord=2)
    #    y = y + np.random.randn(y.shape[0]) * var

    # Add noise to the features
    if feature_noise > 0:
        # print('adding', feature_noise, 'noise to features')
        var = feature_noise * torch.norm(x, p=2) / np.sqrt(len(x))
        x = x + torch.randn(x.shape, device=x.device) * var
    return x, y


# Gets all combinations of hyperparameters given in hp_dict and return a list, 
# leaves the list of the keys of except_keys untouched though
def get_hyperparameters_combination(hp_dict, except_keys=[]):
    except_dict = {}
    for except_key in except_keys:
        except_dict[except_key] = hp_dict.pop(except_key)
    iterables = [value if isinstance(value, list) else [value] for value in hp_dict.values()]
    all_combinations = list(product(*iterables))
    # Create a list of dictionaries for each combination
    combination_dicts = [{**{param: value for param, value in zip(hp_dict.keys(), combination)}, **except_dict} for combination in
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


def classification_layer(op, data_parameters, sigmoid, one_hot):
    softmax = torch.nn.Softmax(dim=1)
    
    if one_hot:
        index = 0
        if data_parameters['degree_input_numerator']>0:
            op[:, index:index + data_parameters['degree_input_numerator']+1] = softmax(op[:, index:index + data_parameters['degree_input_numerator']+1])
            index += 1+data_parameters['degree_input_numerator']
        if data_parameters['degree_input_denominator']>0:
            op[:, index:index + data_parameters['degree_input_denominator']+1] = softmax(op[:, index:index + data_parameters['degree_input_denominator']+1])
            index += 1+data_parameters['degree_input_denominator']
        if data_parameters['degree_output_numerator']>0:
            op[:, index:index + data_parameters['degree_output_numerator']+1] = softmax(op[:, index:index + data_parameters['degree_output_numerator']+1])
            index += 1+data_parameters['degree_output_numerator']
        if data_parameters['degree_output_denominator']>0:
            op[:, index:index + data_parameters['degree_output_denominator']+1] = softmax(op[:, index:index + data_parameters['degree_output_denominator']+1])
            index += 1+data_parameters['degree_output_denominator']
        if len(data_parameters['function_names_str'])>0:    
            op[:, index:] = softmax(op[:, index:])
    else:
        op[:,0] = data_parameters['degree_input_numerator'] * sigmoid(op[:,0])
        op[:,1] = data_parameters['degree_input_denominator'] * sigmoid(op[:,1])
        op[:,2] = data_parameters['degree_output_numerator'] * sigmoid(op[:,2])
        op[:,3] = data_parameters['degree_output_denominator'] * sigmoid(op[:,3])
        op[:,4:] = sigmoid(op[:,4:])  # functions
    return op

class MLP(nn.Module):
    def __init__(self, L, output_size, h, objective='mask', data_parameters=None, one_hot=True):
        super().__init__()
        self.one_hot = one_hot
        if self.one_hot:
            self.softmax = nn.Softmax(dim=1)
        self.objective = objective
        self.data_parameters = data_parameters
        self.sigmoid = nn.Sigmoid()
        
        self.fc1 = nn.Linear(L, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, output_size)

    def forward(self, x):
        # x.shape = (batch_size, seq_length, dim_input)
        x = x.flatten(1, -1)  # x.shape = (batch_size, seq_length * dim_input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.objective == 'mask':
            x = self.sigmoid(x)
        elif self.objective == 'model_parameters':
            x = classification_layer(x, self.data_parameters, self.one_hot, self.softmax, self.sigmoid)
        return x


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, device):
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True, device=device)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # h0 = torch.zeros(self.n_layers,batch_size,self.hidden_size).to(device)
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        # print(f"Out: {out.shape}")
        # print(f"Hidden: {hidden.shape}")
        # out = out.contiguous().view(-1,self.hidden_size)
        out = out[:, -1, :]
        # print(f"Out: {out.shape}")
        out = torch.sigmoid(self.fc(out))
        return out  # , hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device)
        return hidden

class SetTransformer(nn.Module):
    def __init__(self, dim_input, dim_embedding, num_outputs, dim_output, sab_in_output=True, objective='mask',
            num_inds=32, dim_hidden=128, num_heads=6, ln=False, num_layers_enc=2, num_layers_dec=1, activation_fct='ReLU',
            data_parameters=None, dropout=0.0, one_hot=True, embedding=False, dim_hidden_embedding=None):
        super(SetTransformer, self).__init__()
        self.embedding_str = embedding
        if not dim_embedding:
            dim_embedding = dim_input
        if self.embedding_str=='linear':
            self.embedding = nn.Linear(dim_input, dim_embedding)
        elif self.embedding_str=='linear-relu':
            self.embedding = nn.Sequential(nn.Linear(dim_input, dim_embedding), 
                                           nn.ReLU())
        elif self.embedding_str=='linear-relu-linear':
            if not dim_hidden_embedding:
                dim_hidden_embedding=dim_embedding
            self.embedding = nn.Sequential(nn.Linear(dim_input, dim_hidden_embedding), 
                                           nn.ReLU(),
                                           nn.Linear(dim_hidden_embedding, dim_embedding))
        else:
            assert dim_input == dim_embedding
            assert self.embedding_str == 'no'
            
        self.one_hot = one_hot
        self.softmax = nn.Softmax(dim=1)
        self.objective = objective
        self.data_parameters = data_parameters
        self.dropout = nn.Dropout(p=dropout)
        encoder = [ISAB(dim_embedding, dim_hidden, num_heads, num_inds, ln=ln)]
        for i in range(num_layers_enc - 1):
            encoder.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.enc = nn.Sequential(*encoder)
        decoder = [PMA(dim_hidden, num_heads, num_outputs, ln=ln)]
        if sab_in_output:
            decoder.append(SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
            decoder.append(SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        for i in range(num_layers_dec - 1):
            decoder.append(nn.Linear(dim_hidden, dim_hidden))
            decoder.append(self.dropout)
            if activation_fct == 'ReLU':
                decoder.append(nn.ReLU())
            elif activation_fct == 'Sigmoid':
                decoder.append(nn.Sigmoid())
            elif activation_fct == 'ELU':
                decoder.append(nn.ELU())
            else:
                raise NotImplementedError(f'Activation function {activation_fct} is not implemented. Please choose among: ReLU, Sigmoid, and ELU.')
        decoder.append(nn.Linear(dim_hidden, dim_output))
        decoder.append(self.dropout)
        self.dec = nn.Sequential(*decoder)
        
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, X, return_logits=True):
        # X.shape = (batch_size, seq_length, dim_input)
        if self.embedding_str != 'no':
            X = self.embedding(X)
        # X.shape = (batch_size, seq_length, dim_embedding)
        op  = self.dec(self.enc(X)).squeeze(1) # X.shape[0], self.dim_output * self.num_outputs
        if not return_logits:
            if self.objective == 'mask':
                op = self.sigmoid(op)
            elif self.objective == 'model_parameters':
                op = classification_layer(op, self.data_parameters, self.sigmoid, self.one_hot)
        return op

class MAB(nn.Module):
    def __init__(self,dim_Q,dim_K,dim_V,num_heads,ln=False):
        super(MAB,self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads 
        self.fc_q = nn.Linear(dim_Q,dim_V)
        self.fc_k = nn.Linear(dim_K,dim_V)
        self.fc_v = nn.Linear(dim_K,dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V,dim_V)

    
    def forward(self,Q,K):
        Q = self.fc_q(Q)
        K,V = self.fc_k(K),self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split,2),0)
        K_ = torch.cat(K.split(dim_split,2),0)
        V_ = torch.cat(V.split(dim_split,2),0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V),2)
        O = torch.cat((Q_+A.bmm(V_)).split(Q.size(0),0),2)
        O = O if not getattr(self,'ln0',None) else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if not getattr(self,'ln1',None) else self.ln1(O)
        return O
    
class SAB(nn.Module):
    def __init__(self,dim_in,dim_out,num_heads,ln=False):
        super(SAB,self).__init__()
        self.mab = MAB(dim_in,dim_in,dim_out,num_heads,ln=ln)
    
    def forward(self,X):
        return self.mab(X,X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False): # (dim_input, dim_hidden, num_heads, num_inds, ln=ln)
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)
    
class PMA(nn.Module):
     def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

     def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class TransformerModel(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(
            self,
            in_features,
            embedding_size,
            out_features,
            nhead=1,  # 8,
            dim_feedforward=128,  # 2048,
            num_layers=1,  # 6,
            dropout=0.1,
            activation="relu",
            classifier_dropout=0.1,
            num_classifier_layer=2
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Linear(in_features, embedding_size)

        assert embedding_size % nhead == 0, "nheads must divide evenly into embedding_size"

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        classifier = []
        for _ in range(num_classifier_layer):
            classifier.append(nn.Linear(embedding_size, embedding_size))
            classifier.append(nn.Dropout(classifier_dropout))
        classifier.append(nn.Linear(embedding_size, out_features))
        classifier.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*classifier)
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.embedding_size)
        # x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


def covering_loss(prediction, target):
    loss = - (target * torch.log(prediction)).mean()
    # loss +=

def normalize(y):
    return y / y.abs().max(1, keepdim=True)[0]
    
def check_created_dataset(model, x, y, a, params, upperlimit, batch_size):
    assert y.isnan().sum() == 0
    assert (torch.abs(y) > upperlimit).sum() == 0
    assert torch.norm(params[a==0]) == 0
    assert (params[a==1]==0).sum() == 0
    if x.shape[2] > batch_size:
        y_test = predict_in_reduced_batches(x, model, batch_size, params)
    else:
        y_test = model.predict_batch_x_params(params, x)
    
    verbose = False
    if verbose:
        for i in range(y.shape[1]):
            if ((y / y_test) != 1).sum(dim=0)[i] > 0:
                print('Something went wrong:')
            else:
                print('All went well:')
            print(model.get_formula(params[:,i], verbose=False))
            
    if not torch.norm(y_test - y) / torch.norm(y) < 0.01:
        print(model)

    
    assert torch.norm(y_test - y) / torch.norm(y) < 0.01 or (torch.norm(y_test) == 0 and torch.norm(y) == 0)
    
    logging.info(f'Valid data sets')
    
def predict_in_reduced_batches(x, model, batch_size, params):
    num_samples = x.shape[0]
    dataset_size = x.shape[2]
    y = torch.zeros(num_samples, dataset_size, dtype=torch.float64, device=x.device)
    for i in range(int(np.ceil(dataset_size / batch_size))):
        y[:, i*batch_size:(i+1)*batch_size] = model.predict_batch_x_params(params[:, i*batch_size:(i+1)*batch_size], 
                                                                                    x[:, :, i*batch_size:(i+1)*batch_size])
    return y

def _is_valid(y, upperlimit=10**8):
    return (~ y.isnan()) & (torch.abs(y) < upperlimit)

def visualize_data(x, y, params, num_examples, model):
    import matplotlib.pyplot as plt
    for i in range(num_examples):
        x_test = x[:, 0, i]
        y_test = y[:, i]
        sorting = torch.sort(x_test)[1]
        plt.plot(x_test[sorting], y_test[sorting])
        expression = model.get_formula(params[:, i], verbose=False)
        print(f'Expression: {expression}')
        plt.title(f'{expression}') 
        plt.show()

def filter_singularities(x, y, a, params, num_samples, n_params, dimension, device, model, batch_size, upperlimit, param_distribution):
    invalid_points = ~ _is_valid(y, upperlimit)
    invalid_datasets = invalid_points.sum(dim=0) > 0
    too_invalid_datasets = invalid_points.sum(dim=0) > num_samples * (3/4) # more than 3/4 of the data points are Nan
    attempts = 0
    max_attempts = 6
    while (invalid_datasets.sum() > 0) and attempts < max_attempts:
        attempts += 1
        logging.info(f'Attempt {attempts}: Filling in {invalid_datasets.sum()} new data sets because of nan values.')
        #params[:, too_invalid_datasets] = torch.randn(n_params, too_invalid_datasets.sum(), dtype=torch.float64,
        #                                                    device=device) * 3 * a[:, too_invalid_datasets]
        params[:, too_invalid_datasets] = sample_parameters(a[:, too_invalid_datasets], param_distribution)
        logging.info(f'Attempt {attempts}: Changing {too_invalid_datasets.sum()} parameters because they produced too many nan values.')
        candidate_x = torch.zeros(num_samples * (2 * attempts), dimension, invalid_datasets.sum(), 
                                                    dtype=torch.float64, device=device).uniform_(-10, 10)
        if invalid_datasets.sum() > batch_size / (2 * attempts):
            candidate_y = predict_in_reduced_batches(candidate_x, model,
                                                        batch_size=int(batch_size / (2 * attempts)), 
                                                        params=params[:, invalid_datasets])
        else:
            candidate_y = model.predict_batch_x_params(params[:, invalid_datasets], candidate_x)
        valid_candidate_points = _is_valid(candidate_y, upperlimit)
        valid_candidate_sets = valid_candidate_points.sum(dim=0) >= num_samples 
        # Select the good data points
        indices=torch.multinomial(0.0 + valid_candidate_points[:, valid_candidate_sets].T, num_samples=num_samples).T
        
        masked_y = y[:, invalid_datasets]
        masked_y[:, valid_candidate_sets] = torch.gather(candidate_y[:, valid_candidate_sets], dim=0, index=indices)
        y[:, invalid_datasets] = masked_y
        
        masked_x = x[:,:, invalid_datasets]
        masked_x[:,:, valid_candidate_sets] = torch.gather(candidate_x[:,:, valid_candidate_sets], 
                                                                        dim=0, index=indices.repeat(dimension, 1, 1).permute(1,0,2))
        x[:,:, invalid_datasets] = masked_x

        invalid_points = ~ _is_valid(y, upperlimit)
        invalid_datasets = invalid_points.sum(dim=0) > 0
        too_invalid_datasets = invalid_points.sum(dim=0) > num_samples * (3/4) # more than 3/4 of the data points are Nan
        
    if attempts == max_attempts:
        logging.info(f'!!!!! {max_attempts} attempts were necessary for the nan values and there are still {invalid_datasets.sum()} nan values remaining... !!!!!')
        if invalid_datasets.sum() > 0:
            logging.info(f'Therefore, we remove these {invalid_datasets.sum()} data sets from the training set.')
            x = x[:,:, ~ invalid_datasets] 
            y = y[:, ~ invalid_datasets] 
            a = a[:, ~ invalid_datasets] 
            params = params[:, ~ invalid_datasets] 
    return x, y, a, params
    
def filter_singularities_domain(x, y, a, params, num_samples, n_params, device, model, batch_size, upperlimit, param_distribution):
    invalid_points = ~ _is_valid(y, upperlimit)
    invalid_datasets = invalid_points.sum(dim=0) > 0
    attempts = 0
    max_attempts = 6
    while (invalid_datasets.sum() > 0) and attempts < max_attempts:
        attempts += 1
        logging.info(f'Attempt {attempts}: Filling in {invalid_datasets.sum()} new data sets because of nan values.')
        #params[:, invalid_datasets] = torch.randn(n_params, invalid_datasets.sum(), dtype=torch.float64,
        #                                                    device=device) * 3 * a[:, invalid_datasets]
        # params = torch.bernoulli(a) * 2 - 1
        params[:, invalid_datasets] = sample_parameters(a[:, invalid_datasets], param_distribution)

        if invalid_datasets.sum() > batch_size:
            candidate_y = predict_in_reduced_batches(x[:, :, invalid_datasets], model, batch_size=batch_size, params=params[:, invalid_datasets])
        else:
            candidate_y = model.predict_batch_x_params(params[:, invalid_datasets], x[:, :, invalid_datasets])
        valid_candidate_points = _is_valid(candidate_y, upperlimit)
        valid_candidate_sets = valid_candidate_points.sum(dim=0) == num_samples 
        
        masked_y = y[:, invalid_datasets]
        masked_y[:, valid_candidate_sets] = candidate_y[:, valid_candidate_sets]
        y[:, invalid_datasets] = masked_y

        invalid_points = ~ _is_valid(y, upperlimit)
        invalid_datasets = invalid_points.sum(dim=0) > 0
        
    if attempts == max_attempts:
        logging.info(f'!!!!! {max_attempts} attempts were necessary for the nan values and there are still {invalid_datasets.sum()} nan values remaining... !!!!!')
        if invalid_datasets.sum() > 0:
            logging.info(f'Therefore, we remove these {invalid_datasets.sum()} data sets from the training set.')
            x = x[:,:, ~ invalid_datasets] 
            y = y[:, ~ invalid_datasets] 
            a = a[:, ~ invalid_datasets] 
            params = params[:, ~ invalid_datasets] 
    return x, y, a, params

def sample_parameters(a, param_distribution):
    if param_distribution == 'gaussian':
        return torch.randn(a.shape, dtype=torch.float64, device=a.device) * 3 * a
    elif param_distribution == 'bernoulli':
        return (torch.bernoulli(torch.ones(a.shape, dtype=torch.float64, device=a.device)) * 2 - 1) * a
    else:
        raise NotImplementedError(f'Choose as the parameter distribution one of the following:' + 
                                  f' "gaussian" or "bernoulli". You chose {param_distribution}')
    
    
def generate_batch_of_functions(model, dataset_size, num_samples, device, n_functions_max, ensure_uniqueness, num_visualizations,
                                normalization, debugging, most_complex_function=False, singularity_free_domain=True, param_distribution='gaussian'):
    dimension = model.n_input
    x = torch.zeros(num_samples, dimension, dataset_size, dtype=torch.float64, device=device)
    if singularity_free_domain:
        x = x.uniform_(1, 5)
    else:
        x = x.uniform_(-10, 10)
    n_params = model.get_number_parameters()

    a = np.zeros((n_params, dataset_size))

    for i in range(dataset_size):
        if ensure_uniqueness:
            a[:, i] = model.get_random_coefficients_unique(n_functions_max=n_functions_max)
        else:
            a[:, i] = model.get_random_coefficients(n_functions_max=n_functions_max, most_complex_function=most_complex_function)
    a = torch.tensor(a, dtype=torch.float64, device=device)
    # params = torch.randn(n_params, dataset_size, dtype=torch.float64, device=device) * 3 * a
    params = sample_parameters(a, param_distribution)
    batch_size = int(10**7 / n_params / num_samples) 
    if dataset_size > batch_size:
        y = predict_in_reduced_batches(x, model, batch_size, params)
    else:
        y = model.predict_batch_x_params(params, x)
    if not singularity_free_domain:
        logging.info(f'filter_singularities')
        upperlimit = 10**8
        x, y, a, params = filter_singularities(x, y, a, params, num_samples, n_params, dimension, device, model, 
                                                batch_size, upperlimit, param_distribution)
    else:
        logging.info(f'filter_singularities_domain')
        upperlimit = 10**3
        x, y, a, params = filter_singularities_domain(x, y, a, params, num_samples, n_params, device, model, batch_size, 
                                                        upperlimit, param_distribution)
        # y_single_prediction = torch.zeros(num_samples, dataset_size, dtype=torch.float, device=device)
        #for i in range(dataset_size):
        #    y_single_prediction[:, i] = model.predict(params[:, i], x[:, :, i])

    #    params[i] = params[i] * a[i]
    #    y[i] = target_model.predict(params[i], x)
    
    debugging = False
    if debugging:
        if x.shape[2] > 0:
            debug_nd_dataset(x, y, params, model)
    # if x.shape[2] > 0:
    #     formula = round_expr(model.get_formula(params[:, 0], verbose=False, decimals=3), 3)
    #     print(f'${sympy.latex(formula)}$ \\\\ ')
    if x.shape[2] > 0:
        check_created_dataset(model, x, y, a, params, upperlimit=upperlimit, batch_size=batch_size)
        visualize_data(x, y, params, num_visualizations, model)
    x = torch.tensor(x, dtype=torch.float, device=device)
    y = torch.tensor(y, dtype=torch.float, device=device)
    a = torch.tensor(a, dtype=torch.float, device=device)
    
    x = x.permute(2, 0, 1) # (dataset_size, num_samples, dimension)
    y = (y.T).unsqueeze(-1) # (dataset_size, num_samples, 1)
    
    if normalization == 'No':
        pass
    elif normalization == 'Full':
        y = F.normalize(y)
    elif normalization == 'DataSetWise':
        y = F.normalize(y, dim=1)
    elif normalization == 'DataSetWiseXY-11':
        y = min_max_normalization(y, dim=1, new_min=-1, new_max=1)
        x = min_max_normalization(x, dim=1, new_min=-1, new_max=1)
    
    data = torch.cat([x, y], dim=-1) # (dataset_size, num_samples, dimension + 1)
    a = a.T # (dataset_size, num_samples)
    params = params.T # (dataset_size, num_samples)
    
    return data, a, params

def min_max_normalization(data, dim, new_min, new_max):
    data_min = data.min(dim=dim)[0].unsqueeze(dim)
    data_max = data.max(dim=dim)[0].unsqueeze(dim)
        
    data = (data - data_min) / (data_max - data_min) * (new_max - new_min) + new_min
    
    # avoiding nan for constant functions and instead set it to zero
    constant = data_min == data_max    
    data[constant.squeeze(dim).all(-1)] = 0.0
    
    return data  

import random

def swap_rows(arr, i, j):
    tmp = copy.deepcopy(arr[i])
    arr[i] = copy.deepcopy(arr[j])
    arr[j] = tmp

def fisher_yates_shuffle(arr1, arr2):
    """
    Perform Fisher-Yates shuffle (in-place) on the given array.
    
    Parameters:
        arr (list): The input list to be shuffled.
    """
    n = len(arr1)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)  # Pick a random index from 0 to i
        
        # This swapping only works for 1-d arrays
        # arr1[i], arr1[j] = arr1[j], arr1[i]  # Swap elements at indices i and j
        # arr2[i], arr2[j] = arr2[j], arr2[i]  # Swap elements at indices i and j
        swap_rows(arr1, i, j)
        swap_rows(arr2, i, j)
    return arr1, arr2

def create_dataset_model_parameters(dataset_size, ensure_uniqueness, max_degree_input_numerator, max_degree_output_numerator, width,
                   max_function_names_str, normalization, n_functions_max, num_samples,
                   max_degree_output_denominator, d_min, d_max, maximal_n_active_base_functions,
                   maximal_potence, function_dict, function_name_dict, debugging,
                   max_degree_input_denominator, device='cpu', num_visualizations=0, most_complex_function=False, 
                   singularity_free_domain=True, param_distribution='gaussian', use_fisher_yates_shuffle=True, target_noise=0.0,
                   feature_noise=0.0):
    model_parameters_max = {}
    model_parameters_max['max_deg_output'] = max_degree_output_numerator
    model_parameters_max['max_deg_input'] = max_degree_input_numerator
    model_parameters_max['max_deg_input_denominator'] = max_degree_input_denominator
    model_parameters_max['max_deg_output_denominator'] = max_degree_output_denominator
    model_parameters_max['function_names'] = max_function_names_str
    model_parameters_max['maximal_n_functions'] = maximal_n_active_base_functions
    model_parameters_max['max_deg_output_polynomials_specific'] = [1 for _ in range(len(max_function_names_str))]
    model_parameters_max['max_deg_output_polynomials_denominator_specific'] = [1 for _ in range(len(max_function_names_str))]
    model_parameters_max['maximal_potence'] = maximal_potence
    
    
    model_parameter_list = get_complete_model_parameter_list(model_parameters_max, iterate_polynomials=True)
    
    datasets_per_modelparameter = int(dataset_size / len(model_parameter_list) / (d_max - d_min + 1)) + 1
    
    list_data = []
    list_a = []
    list_params = []
    list_model_parameter_encodings = []
    # For now we create a data set for a fixed dimension; later we will iterate over this
    for dimension in range(d_min, d_max+1):
        logging.info(f'Starting dimension {dimension+1} out of range ({[d_min, d_max]}).')
        
        for i, model_parameter in enumerate(model_parameter_list):
            logging.info(f'Starting with model parameter number {i}/{len(model_parameter_list)}')
            functions = []
            function_names = []
            for function_name_str in model_parameter['function_names']:
                functions.append(function_dict[function_name_str])
                function_names.append(function_name_dict[function_name_str])
            
            model = ParFamTorch(n_input=dimension,
                        degree_input_numerator=model_parameter['degree_input_polynomials'],
                        degree_output_numerator=model_parameter['degree_output_polynomials'],
                        degree_input_denominator=model_parameter['degree_input_denominator'],
                        degree_output_denominator=model_parameter['degree_output_denominator'],
                        degree_output_numerator_specific=model_parameter['degree_output_polynomials_specific'],
                        degree_output_denominator_specific=model_parameter[
                            'degree_output_polynomials_denominator_specific'],
                        width=model_parameter['width'], maximal_potence=model_parameter['maximal_potence'],
                        functions=functions, device=device,
                        function_names=function_names,
                        enforce_function=model_parameter['enforce_function'],
                        stabilize_denominator=False)
            
            data, a, params = generate_batch_of_functions(model, datasets_per_modelparameter, num_samples, device, n_functions_max, 
                                            ensure_uniqueness, num_visualizations, normalization, debugging, most_complex_function,
                                            singularity_free_domain, param_distribution)
            if target_noise > 0 or feature_noise > 0:
                for i in range(data.shape[0]):
                    data[i,:,:-1], data[i,:,-1] = add_noise(data[i,:,:-1], data[i,:,-1], target_noise, feature_noise)
            
            model_parameter_encoding = model_parameter_to_tensor(model_parameter, model_parameters_max, device)
            
            list_data.append(torch.cat([torch.zeros(data.shape[0], data.shape[1], d_max+1-data.shape[2], device=data.device, dtype=data.dtype), data], dim=-1))
            list_a.append(a)
            list_params.append(params)
            list_model_parameter_encodings.append(model_parameter_encoding.squeeze(0).repeat(data.shape[0], 1))

            logging.info(f'Target device: {device}')
            logging.info(f'Actual device: {data.device}')
            # logging.info(f'Total memory (GPU): {torch.cuda.get_device_properties(0).total_memory / 1024**2}MB')
            # logging.info(f'Reserved memory (GPU): {torch.cuda.memory_reserved(0) / 1024**2}MB')
            # logging.info(f'Allocated memory (GPU): {torch.cuda.memory_allocated(0) / 1024**2}MB')
            # logging.info(f'Allocated memory peak (GPU): {torch.cuda.max_memory_allocated(0) / 1024**2}MB')

            logging.info(f'{using("CPU Memory")}')
        
    data = torch.cat(list_data, dim=0)
    logging.info(f'{using("CPU Memory after concatenating the data")}')
    del list_data
    logging.info(using("CPU Memory after deleting the list of data"))
    
    model_parameter_encodings = torch.cat(list_model_parameter_encodings, dim=0)
    logging.info(f'{using("CPU Memory after concatenating the model parameter encodings")}')
    


    t0 = time.time()
    if use_fisher_yates_shuffle:
        logging.info(f'Using fisher yates to shuffle the data set.')
        data, model_parameter_encodings = fisher_yates_shuffle(data, model_parameter_encodings)
    else:
        logging.info(f'Using torch.randperm to shuffle the data set.')
        permutation = torch.randperm(data.shape[0])
        data = data[permutation]
        model_parameter_encodings = model_parameter_encodings[permutation]        
    t1 = time.time()
    logging.info(f'{using("CPU Memory after permuting x and y")}')
    logging.info(f'Permuting of {len(data)} rows took: {t1-t0:.3f}s.')
    # del permutation
    # logging.info(using('CPU memory after deleting the permutation'))

    return data, list_a, list_params, model_parameter_encodings
        
def model_parameter_to_tensor(model_parameter, model_parameters_max, device):
    # Transforms the model parameters to a tensor (numerical encoding for the degrees)
    encoding_input_numerator = torch.tensor([model_parameter['degree_input_polynomials']], dtype=torch.float, device=device)
    encoding_input_denominator = torch.tensor([model_parameter['degree_input_denominator']], dtype=torch.float, device=device)
    encoding_output_numerator = torch.tensor([model_parameter['degree_output_polynomials']], dtype=torch.float, device=device)
    encoding_output_denominator = torch.tensor([model_parameter['degree_output_denominator']], dtype=torch.float, device=device)

    encoding_functions = torch.tensor([function_str in model_parameter['function_names'] for function_str in model_parameters_max['function_names']],
                                      dtype=torch.float, device=device)
    list_encodings = [encoding_input_numerator, encoding_input_denominator, encoding_output_numerator, encoding_output_denominator, encoding_functions]
    encoding = torch.cat(list_encodings)
    return encoding

def tensor_to_model_parameters(tensor, model_parameters_max):
    # Transforms the model parameters to a tensor (numerical encoding for the degrees)
    model_parameter = {}
    model_parameter['degree_input_polynomials'] = tensor[0].cpu().detach().item()
    model_parameter['degree_input_denominator'] = tensor[1].cpu().detach().item()
    model_parameter['degree_output_polynomials'] = tensor[2].cpu().detach().item()
    model_parameter['degree_output_denominator'] = tensor[3].cpu().detach().item()
    
    model_parameter['function_names'] = []
    for i, function_name in enumerate(model_parameters_max['function_names']):
        if tensor[4+i] > 0.5:
            model_parameter['function_names'].append(function_name)
    return model_parameter

def setup_parfam_for_dlparfam(model_parameters, device, function_dict, function_name_dict, dimension):
    functions = []
    function_names = []
    for function_name_str in model_parameters['function_names_str']:
        functions.append(function_dict[function_name_str])
        function_names.append(function_name_dict[function_name_str])

    degree_output_numerator_specific = [1 for _ in functions]
    degree_output_denominator_specific = [1 for _ in functions]

    model = ParFamTorch(n_input=dimension,
                    degree_input_numerator=model_parameters['degree_input_numerator'],
                    degree_output_numerator=model_parameters['degree_output_numerator'],
                    degree_input_denominator=model_parameters['degree_input_denominator'],
                    degree_output_denominator=model_parameters['degree_output_denominator'],
                    degree_output_numerator_specific=degree_output_numerator_specific,
                    degree_output_denominator_specific=degree_output_denominator_specific,
                    width=model_parameters['width'], maximal_potence=model_parameters['maximal_potence'],
                    functions=functions, device=device,
                    function_names=function_names,
                    enforce_function=model_parameters['enforce_function'],
                    stabilize_denominator=False)
    return model

    
def create_dataset_flexible_grid(dataset_size, ensure_uniqueness, normalization, n_functions_max, num_samples,
                   dimension, function_dict, function_name_dict, model_parameters, debugging,
                   device='cpu', num_visualizations=0, singularity_free_domain=True, param_distribution='gaussian'):

    model = setup_parfam_for_dlparfam(model_parameters, device, function_dict, function_name_dict, dimension)
    
    return generate_batch_of_functions(model, dataset_size, num_samples, device, n_functions_max, ensure_uniqueness,
                                       num_visualizations, normalization, debugging, singularity_free_domain, param_distribution)


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
    a = torch.tensor(a, dtype=torch.float, device=device)  # .unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float, device=device)
    if normalization == 'No':
        pass
    elif normalization == 'Full':
        y = F.normalize(y)

    x = torch.tensor(x.repeat(y.shape[0], axis=1), dtype=torch.float, device=device).T
    data = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
    return data, a, params

def debug_1d_dataset(x, y, params, model):
        x_single_prediction = x[:, 0, 0]
        sorting = x_single_prediction.sort()[1]
        y_single_prediction = model.predict(params[:, 0], x_single_prediction.unsqueeze(-1))
        
        formula = model.get_formula(params[:, 0], verbose=False)

        y_batch_prediction_test = model.predict_batch_x_params(params[:, 0].unsqueeze(-1), x_single_prediction.unsqueeze(-1).unsqueeze(-1))
        
        y_sympy = evaluate_sympy_expression(formula, x_single_prediction)
    
        import matplotlib.pyplot as plt
        plt.plot(x_single_prediction[sorting], y_single_prediction[sorting], '+', label='Single prediction')
        plt.plot(x_single_prediction[sorting], y_sympy[sorting], '-', label='Sympy')
        plt.plot(x[sorting, 0, 0], y[sorting, 0], 'o', label='Batch prediction')
        plt.plot(x[sorting, 0, 0], y_batch_prediction_test[sorting, 0], '*', label='Batch prediction (test)')
        plt.legend()
        plt.show()
        
        print((y_sympy - y_single_prediction).norm() / y_sympy.norm())
        
def debug_nd_dataset(x, y, params, model, i=0):
    x_single_prediction = x[:, :, i]
    formula = model.get_formula(params[:, i], verbose=False, decimals=16)
    formula_to_print = model.get_formula(params[:, i], verbose=False, decimals=3)

    y_single_prediction = model.predict(params[:, i], x_single_prediction)
    y_batch_prediction_test = model.predict_batch_x_params(params[:, i].unsqueeze(-1), x_single_prediction.unsqueeze(-1)).squeeze(1)
    try:
        y_sympy = evaluate_sympy_expression_nd(formula, x_single_prediction)
    except Exception as e:
        print(f'Formula: {formula}')
        print(f'x: {x}')
        print(f'y: {y}')
        model.predict_batch_x_params(params[:, i].unsqueeze(-1), x_single_prediction.unsqueeze(-1)).squeeze(1)
        # raise e
        y_sympy = 0
    if (y_sympy - y_single_prediction).abs().median() > 1:
        # This can be caused by the changes in the function by cutting off the parameters after 3 decimals
        y_single_prediction = model.predict(params[:, i], x_single_prediction)
    
    print(f'\n \n')
    print(f'Model: \n{model}')
    print(f'Formula: {formula_to_print}')
    print(f'Sympy vs Single prediction ParFam: {(y_sympy - y_single_prediction).abs().median()}')
    print(f'Sympy vs Batch prediction ParFam (Test): {(y_sympy - y_batch_prediction_test).abs().median()}')
    print(f'Sympy vs Batch prediction ParFam: {(y_sympy - y[:,i]).abs().median()}')
    print(f'Batch prediction ParFam (Test) vs Single prediction ParFam: {(y_batch_prediction_test - y_single_prediction).abs().median()}')
    
    return
        

def evaluate_sympy_expression_nd(formula, x, start_with_x0=True):
    dimension = x.shape[1]
    y = torch.zeros(x.shape[0], dtype=torch.float)
    if start_with_x0:
        symbols = [sympy.symbols(f'x{i}') for i in range(dimension)]
    else:
        symbols = [sympy.symbols(f'x{i+1}') for i in range(dimension)]
    for i in range(len(x)):
        subs = {symbol: x[i, j].item() for j, symbol in enumerate(symbols)}
        y[i] = float(formula.evalf(subs=subs))
    return y 

def evaluate_sympy_expression_1d(formula, x):
    # x should be a 1-dimensional tensor
    y = torch.zeros(len(x), dtype=torch.float)
    x0 = sympy.symbols('x0')
    for i in range(len(x)):
        y[i] = float(formula.evalf(subs={x0: x[i].item()}))
    return y
