from train import model_parameter_search, extend_function_dict, setup_model, finetune_coeffs
from utils import relative_l2_distance
import sympy
import torch
from sklearn.model_selection import train_test_split
import os
import configparser 
import ast
import numpy as np

class ParFamWrapper:
        
    default_model_parameters = {}
    
    def __init__(self, iterate, config_name='small', degree_input_numerator=None, degree_output_numerator=None, degree_input_denominator=None, degree_output_denominator=None,
                 functions=None, function_names=None, input_names=None, degree_output_numerator_specific=None, maximal_n_active_base_functions=None,
                 maximal_potence=None, degree_output_denominator_specific=None, enforce_function=None, device='cpu', separate_test_set=True):
        
        self.input_names = input_names
        
        root_path = os.path.dirname(os.path.abspath(__file__))  # Get root_path (.../parfam)
        config_path = os.path.join(root_path, 'config_files', 'wrapper', config_name + '.ini')
        self.config = configparser.ConfigParser()  # Create a ConfigParser object
        self.config.read(config_path)  # Read the configuration file
            
        if separate_test_set is not None:
            self.config['META']['separate_test_set'] = str(separate_test_set)
        if iterate is not None:
            if iterate:
                self.config['META']['model_parameter_search'] = 'Complete'
            else:
                self.config['META']['model_parameter_search'] = 'No'
        
        self.training_parameters = {'model': 'ParFamTorch', 'parallel': False} 
        
        model_parameters = dict(self.config['MODELPARAMETERS'])
        model_parameters = {key: ast.literal_eval(model_parameters[key]) for key in
                             model_parameters.keys()}
        
        model_parameters_fix = dict(self.config['MODELPARAMETERSFIX'])
        model_parameters_fix = {key: ast.literal_eval(model_parameters_fix[key]) for key in model_parameters_fix.keys()}
        
        # Overwrite the config files, if the parameter is specified. Use "is not None" to account for False and 0 values. 
        if maximal_n_active_base_functions is not None:
            model_parameters['maximal_n_functions'] = maximal_n_active_base_functions
        if degree_output_numerator is not None:
            model_parameters['max_deg_output'] = degree_output_numerator
            model_parameters_fix['degree_output_polynomials'] = str(degree_output_numerator)
        if degree_input_numerator is not None:
            model_parameters['max_deg_input'] = degree_input_numerator
            model_parameters_fix['degree_input_polynomials'] = str(degree_input_numerator)
        if degree_input_denominator is not None:
            model_parameters['max_deg_input_denominator'] = degree_input_denominator
            model_parameters_fix['degree_input_denominator'] = str(degree_input_denominator)
        if degree_output_denominator is not None:
            model_parameters['max_deg_output_denominator'] = degree_output_denominator
            model_parameters_fix['degree_output_denominator'] = str(degree_output_denominator)
        if function_names is not None:
            assert len(function_names) == len(functions)
            self.function_names_str = [f'function_{i}' for i in range(len(functions))]  # string names for the function to iterate through them
            self.function_dict = {self.function_names_str[i]: functions[i] for i in range(len(functions))}  # dictionary to map string names of the function to the torch function
            self.function_name_dict = {self.function_names_str[i]: function_names[i] for i in range(len(functions))}  # dictionary to map string names of the function to the sympy function
            extend_function_dict(self.function_dict, self.function_name_dict)
            
            model_parameters['function_names'] = self.function_names_str
            model_parameters_fix['function_names'] = str(self.function_names_str)

            
        if maximal_potence is not None:
            model_parameters['maximal_potence'] = maximal_potence     
            model_parameters_fix['maximal_potence'] = str(maximal_potence)
        if enforce_function:
            model_parameters_fix['enforce_function'] = str(enforce_function)

        # These have to be adapted to fit the number of functions, in case specific functions are picked
        if functions is not None:
            if not degree_output_denominator_specific:
                degree_output_denominator_specific = [1 for _ in range(len(functions))]
                
            if not degree_output_numerator_specific:
                degree_output_numerator_specific = [1 for _ in range(len(functions))]    
        
        if degree_output_numerator_specific is not None:
            model_parameters['max_deg_output_polynomials_specific'] = degree_output_numerator_specific
            model_parameters_fix['degree_output_polynomials_specific'] = str(degree_output_numerator_specific)
        if degree_output_denominator_specific is not None:
            model_parameters['max_deg_output_polynomials_denominator_specific'] = degree_output_denominator_specific
            model_parameters_fix['degree_output_polynomials_denominator_specific'] = str(degree_output_denominator_specific)
        
        self.device = device
        self.model_parameters = model_parameters
        self.config['MODELPARAMETERSFIX'] = model_parameters_fix
            
    def fit(self, x, y, time_limit=None, evaluations_limit=None, max_n_active_parameters=None, seed=None, 
            maxiter1=None, optimizer=None, maxiter_per_dim_local_minimizer=None, lambda_1=None,
            lambda_1_piecewise=None, lambda_1_cut=None, local_minimizer=None, lambda_1_finetuning=None,
            iterative_finetuning=None, normalization=None, enforce_function_iterate=None, custom_loss=None,
            repetitions=None):
        # custom_loss takes y_pred and y as an input. y can be chosen as any necessary variables.
        
        training_parameters = dict(self.config['TRAININGPARAMETERS'])
        training_parameters = {key: ast.literal_eval(training_parameters[key]) for key in training_parameters.keys()}
        
        training_parameters['device'] = self.device
        training_parameters['max_dataset_length'] = len(y)
        training_parameters['classifier'] = None
        
        assert x.shape[0] == len(y)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
            y = torch.tensor(y)
        
        self.n_input = x.shape[1]
        
        if seed is not None:
            training_parameters['seed'] = seed
        if time_limit is not None:
            training_parameters['time_limit'] = time_limit
        if evaluations_limit is not None:
            training_parameters['evaluations_limit'] = evaluations_limit
        if max_n_active_parameters is not None:
            training_parameters['max_n_active_parameters'] = max_n_active_parameters
        if maxiter1 is not None:
            training_parameters['maxiter1'] = maxiter1
        if optimizer is not None:
            training_parameters['optimizer'] = optimizer
        if maxiter_per_dim_local_minimizer is not None:
            training_parameters['maxiter_per_dim_local_minimizer'] = maxiter_per_dim_local_minimizer
        if lambda_1 is not None:
            training_parameters['lambda_1'] = lambda_1
        if lambda_1_piecewise is not None:
            training_parameters['lambda_1_piecewise'] = lambda_1_piecewise
        if lambda_1_cut is not None:
            training_parameters['lambda_1_cut'] = lambda_1_cut
        if local_minimizer is not None:
            training_parameters['local_minimizer'] = local_minimizer
        if lambda_1_finetuning is not None:
            training_parameters['lambda_1_finetuning'] = lambda_1_finetuning
        if iterative_finetuning is not None:
            training_parameters['iterative_finetuning'] = iterative_finetuning
        if normalization is not None:
            training_parameters['normalization'] = normalization
        if enforce_function_iterate is not None:
            training_parameters['enforce_function_iterate'] = enforce_function_iterate
        if repetitions is not None:
            training_parameters['repetitions'] = repetitions
            
        print(f'Training parameters: {training_parameters}')

        self.relative_l2_distance_train, self.relative_l2_distance_val, self.r_squared_val, self.formula, self.training_time, \
        self.n_active_coefficients, self.relative_l2_distance_test, self.r_squared_test, self.n_active_coefficients_reduced, \
        self.relative_l2_distance_test_reduced, self.r_squared_test_reduced, self.formula_reduced, self.r_squared_val_reduced, \
        self.n_evaluations, self.best_model_parameters, self.coefficients, \
        self.coefficients_reduced = model_parameter_search(x, y, None, self.model_parameters, 
                                                            training_parameters=training_parameters, accuracy=0.001, 
                                                            config=self.config, model_parameters_perfect=None,
                                                            logging_to_file=False, custom_loss=custom_loss)
        
        self.model = setup_model(self.best_model_parameters, n_input=x.shape[1], device=self.device, model='ParFamTorch')
        
    def get_formula(self, input_names, decimals=3):
        assert len(input_names) == self.n_input
        self.model.input_names = []
        for input_name in input_names:
            if isinstance(input_name, sympy.Symbol): 
                self.model.input_names.append(input_name)
            elif isinstance(input_name, str):
                self.model.input_names.append(sympy.Symbol(input_name))
            else:
                raise TypeError(f'Input_names should be a list of string values or sympy.Symbol. Not {type(input_names)}.')
        formula_reduced = self.model.get_formula(self.coefficients_reduced, decimals=decimals, verbose=False)
        return formula_reduced 
        
    def predict(self, x, reduced=True):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x, device=self.device)
        if reduced:
            coefficients = self.coefficients_reduced
        else:
            coefficients = self.coefficients
        y = self.model.predict(x=x, coefficients=coefficients)
        return y

    def finetune(self, x, y, cutoff, max_dataset_length=5000):
        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, test_size=0.2)
        new_coeffs = finetune_coeffs(x_train, y_train, self.coefficients_reduced, self.model, cutoff, lambda_1=0.01, max_dataset_length=max_dataset_length)
        if new_coeffs is None:
            return None
        print(f'New formula:')
        print(self.model.get_formula(new_coeffs, decimals=3, verbose=False))
        print(f'Relative l_2-distance train: {relative_l2_distance(self.model, new_coeffs, x_train, y_train)}')
        print(f'Relative l_2-distance validation: {relative_l2_distance(self.model, new_coeffs, x_val, y_val)}')
        print(f'If you want to use the new coefficients call parfam.set_coefficients(coefficients)')
        return new_coeffs
    
    def set_coefficients(self, coefficients, decimals=3):
        self.coefficients_reduced = coefficients
        self.formula_reduced = self.model.get_formula(self.coefficients_reduced, decimals=decimals, verbose=False)
            
if __name__ == '__main__':
    import sympy
    import numpy as np
    import matplotlib.pyplot as plt
    device = 'cpu'
    
    functions = []
    function_names = []

    n_data_points = 500 

    x = np.linspace(0, 5, n_data_points).reshape(n_data_points,1)
    y = ((x**2 + x)**(1/2)).flatten()

    def custom_loss(y_pred, y, theta):
        if theta is None:
            reg = 0
        else: 
            reg = torch.norm(theta, p=1)
        return torch.norm(y_pred - y**2, p=2) / torch.norm(y, p=2) + 0.001 * reg

    parfam = ParFamWrapper(config_name='small', iterate=False, degree_input_denominator=0, degree_output_denominator=0, function_names=function_names, functions=functions, 
                            degree_output_numerator=4, maximal_potence=4)
    parfam.fit(x=x, y=y, custom_loss=custom_loss)

    y_pred = parfam.predict(x)

    plt.plot(x, y**2, '+', label='Samples squared')
    plt.plot(x, y_pred, label='Prediction')
    plt.legend()
    plt.savefig('plot.svg')
    
    raise NotImplementedError
    
    feynman = True
    if not feynman:
        # np.random.seed(12345)
        # torch.manual_seed(12345)
        print(f'Using {device}')
        a = 5 * torch.randn(1)
        x = np.arange(1, 5, 0.2)
        # x = np.sort(np.random.uniform(-1.2, 5, 100))
        x = x.reshape(len(x), 1)
        # x = np.random.uniform(-3, 3, 200).reshape(100, 2)
        print(x.shape)
        x = torch.tensor(x, device=device)
        test_model = False

        def func(a, x, module):
            # return module.sin((a[0] * x + 1) / (0.1 * x + 2))
            # return module.sin((a[0] * x + 1) / (0.1 * x))
            # return module.sin((a[0] * x))
            # return a[0] * x / (1+x)
            # return 0.2 * module.sin(a[0] * x) / x
            # return 0.5 * x / (x + 1)
            return module.log(x + 1.4) + module.log(x ** 2 + 1.3)
            # return module.sin(x ** 2) * module.cos(x) - 1
            # return module.sin(x[0]) + module.sin(x[1]**2)
    else:
        n_datapoints = 500
        dim = 4
        x = np.random.uniform(1, dim, n_datapoints * dim).reshape(n_datapoints, dim)
        print(x.shape)
        x = torch.tensor(x, device=device)
        a = None
        def func(a, x, module):
            # return x[0] * (x[1]*(x[2]**2 + x[3]**2 + x[4]**2))**(1/2)
            return x[0] * ((x[2]**2 + x[3]**2 + x[1]**2))**(1/2)

    if x.shape[1] == 1:
        y = func(a, x, np).squeeze(1)
        x_sym = sympy.symbols('x')
    else:
        y = func(a, x.T, np)
        x_sym = []
        for i in range(x.shape[1]):
            x_sym.append(sympy.symbols(f'x{i}'))
    target_expr = func(a, x_sym, sympy)
    print(f'Target formula: {target_expr}')



    if feynman:
        function_dict = {'sqrt': lambda x: torch.sqrt(torch.abs(x)),
                 'exp': lambda x: torch.minimum(torch.exp(x), np.exp(10) + torch.abs(x)),
                 'log': lambda x: torch.log(torch.abs(x) + 0.000001),
                 'cos': torch.cos, 'sin': torch.sin}
        function_name_dict = {'sqrt': lambda x: sympy.sqrt(sympy.Abs(x)), 'exp': sympy.exp, 'cos': sympy.cos, 'sin': sympy.sin,
                            'log': lambda x: sympy.log(sympy.Abs(x) + 0.000001)}
        
        # functions = [function_dict['sqrt']]
        # function_names = [function_name_dict['sqrt']]
        functions = []
        function_names = []
        parfam = ParFamWrapper(iterate=True, # iterate through multiple different parametric families (costs more time, but is the better choice when one is not sure about the degrees and the functions of the target formula)
                                functions=functions, function_names=function_names, # which functions to use 
                                degree_input_numerator=2, degree_output_numerator=3, degree_input_denominator=0, degree_output_denominator=0,  # the maximal degrees of the polynomials in the parametric family
                                input_names=x_sym,  # the names of the input variables
                                enforce_function=False,  # has only an effect, if iterate=False
                                device='cpu', 
                                separate_test_set=True  # ParFam uses a smaller set for training, to use a part of it as a test set afterwards
                                )
    else:
        functions = [torch.sin]  # [lambda x: torch.log(torch.abs(x) + 0.000001)] * 2
        function_names = [sympy.sin]  # [lambda x: sympy.log(sympy.Abs(x) + 0.000001)] * 2
        parfam = ParFamWrapper(iterate=False, functions=functions, function_names=function_names, degree_input_numerator=1, degree_output_denominator=1, 
                           degree_output_numerator=1, degree_input_denominator=0, enforce_function=False)

        
    parfam.fit(x, y, time_limit=100)
    print(f'Target expr: {target_expr}')
    print(f'Computed formula: {parfam.formula.simplify()}')
    

    y_pred = parfam.predict(x).cpu().detach().numpy()
        
    if x.shape[1] == 1:
        plt.plot(x, y, '+', label='Samples')
        plt.plot(x, y_pred, label='Prediction')
        plt.legend()
        plt.show()

    print(f'Relative l2 distance: {np.linalg.norm(y - y_pred, ord=2) / np.linalg.norm(y, ord=2)}')


    print(f'### Now from utils ###')
    from utils import relative_l2_distance, r_squared
    print(f'Relative l2 distance: {relative_l2_distance(parfam.model, parfam.coefficients_reduced, x, y)}')
    print(f'R squared: {r_squared(parfam.model, parfam.coefficients_reduced, x, y)}')
    
    
    print(parfam.coefficients_reduced)
    parfam.finetune(x, y, cutoff=0.5, max_dataset_length=5000)    
    
    if not feynman:
        print('Formula with variable a')
        print(parfam.get_formula(input_names=[sympy.Symbol('a')], decimals=5))
    else:
        print(f'Computed formula: {parfam.formula_reduced.simplify()}')