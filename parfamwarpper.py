from parfam_torch import ParFamTorch, Evaluator
from utils import get_complete_model_parameter_list, function_dict, function_name_dict
from train import get_max_model_parameter_list, model_parameter_search, extend_function_dict, setup_model
import random
import torch


class ParFamWrapper:
        
    default_model_parameters = {}
    
    def __init__(self, iterate, degree_input_numerator=2, degree_output_numerator=2, degree_input_denominator=2, degree_output_denominator=2,
                 functions=[], function_names=[], input_names=None, degree_input_numerator_specific=None,
                 degree_output_numerator_specific=None, maximal_n_active_base_functions=2, repetitions=1,
                 degree_input_denominator_specific=None, normalize_denom=True, maximal_potence=3,
                 degree_output_denominator_specific=None, enforce_function=False, device='cpu', separate_test_set=True):
        
        self.input_names = input_names
        
        self.config = {'META': {}}
        self.config['META']['separate_test_set'] = str(separate_test_set)
        if iterate:
            self.config['META']['model_parameter_search'] = 'Complete'
        else:
            self.config['META']['model_parameter_search'] = 'No'
        
        self.training_parameters = {'model': 'ParFamTorch', 'repetitions': repetitions, 'parallel': False}    

        
        if not degree_output_denominator_specific:
            degree_output_denominator_specific = [1 for _ in range(len(functions))]
            
        if not degree_output_numerator_specific:
            degree_output_numerator_specific = [1 for _ in range(len(functions))]
            
        assert len(function_names) == len(functions)
        self.function_names_str = [f'function_{i}' for i in range(len(functions))]  # string names for the function to iterate through them
        self.function_dict = {self.function_names_str[i]: functions[i] for i in range(len(functions))}  # dictionary to map string names of the function to the torch function
        self.function_name_dict = {self.function_names_str[i]: function_names[i] for i in range(len(functions))}  # dictionary to map string names of the function to the sympy function
            
        extend_function_dict(self.function_dict, self.function_name_dict)
            
        model_parameters_max = {}
        model_parameters_max['max_deg_output'] = degree_output_numerator
        model_parameters_max['max_deg_input'] = degree_input_numerator
        model_parameters_max['max_deg_input_denominator'] = degree_input_denominator
        model_parameters_max['max_deg_output_denominator'] = degree_output_denominator
        model_parameters_max['function_names'] = self.function_names_str
        model_parameters_max['maximal_n_functions'] = maximal_n_active_base_functions
        model_parameters_max['max_deg_output_polynomials_specific'] = degree_output_numerator_specific
        model_parameters_max['max_deg_output_polynomials_denominator_specific'] = degree_output_denominator_specific
        model_parameters_max['maximal_potence'] = maximal_potence
        self.model_parameters_max = model_parameters_max
            
        self.device = device
        # if iterate:
        #     self.model_parameter_list = get_complete_model_parameter_list(self.model_parameters_max, iterate_polynomials=True)
        # else:
        #     self.model_parameter_list = get_max_model_parameter_list(self.model_parameters_max)
            
    def fit(self, x, y, time_limit=1000, evaluations_limit=1000000, max_n_active_parameters=10, seed=None, 
            maxiter1=100, optimizer='basinhopping', maxiter_per_dim_local_minimizer=100, lambda_1=0.0001,
            lambda_1_piecewise=0.0, lambda_1_cut=0.0, local_minimizer='bfgs', lambda_1_finetuning=0.01,
            iterative_finetuning=True):

        
        assert x.shape[0] == len(y)
        self.n_input = x.shape[1]
        
        if not seed:
            seed = random.randint(1, 100000)
        self.training_parameters['seed'] = seed
        self.training_parameters['device'] = self.device
        self.training_parameters['target_noise'] = 0    # We do not want to add extra noise on our data
        self.training_parameters['feature_noise'] = 0   # We do not want to add extra noise on our data
        self.training_parameters['time_limit'] = time_limit
        self.training_parameters['evaluations_limit'] = evaluations_limit
        self.training_parameters['max_n_active_parameters'] = max_n_active_parameters   # How many parameters should be active in the end (if possible)
        self.training_parameters['maxiter1'] = maxiter1
        self.training_parameters['maxiter2'] = 0
        self.training_parameters['optimizer'] = optimizer
        self.training_parameters['maxiter_per_dim_local_minimizer'] = maxiter_per_dim_local_minimizer
        self.training_parameters['lambda_1'] = lambda_1
        self.training_parameters['lambda_1_piecewise'] = lambda_1_piecewise
        self.training_parameters['lambda_1_cut'] = lambda_1_cut
        self.training_parameters['max_dataset_length'] = len(y)
        self.training_parameters['classifier'] = None
        self.training_parameters['local_minimizer'] = local_minimizer
        self.training_parameters['lambda_1_finetuning'] = lambda_1_finetuning
        self.training_parameters['iterative_finetuning'] = iterative_finetuning
        self.training_parameters['pruning_iterations'] = 1
        
        self.relative_l2_distance_train, self.relative_l2_distance_val, self.r_squared_val, self.formula, self.training_time, \
        self.n_active_coefficients, self.relative_l2_distance_test, self.r_squared_test, self.n_active_coefficients_reduced, \
        self.relative_l2_distance_test_reduced, self.r_squared_test_reduced, self.formula_reduced, self.r_squared_val_reduced, \
        self.n_evaluations, self.best_model_parameters, self.coefficients, \
        self.coefficients_reduced = model_parameter_search(x, y, None, self.model_parameters_max, 
                                                            training_parameters=self.training_parameters, accuracy=0.001, 
                                                            config=self.config, model_parameters_perfect=None,
                                                            logging_to_file=False)
        
        self.model = setup_model(self.best_model_parameters, n_input=x.shape[1], device=self.device, model='ParFamTorch')
        
    def predict(self, x, reduced=True):
        if reduced:
            coefficients = self.coefficients_reduced
        else:
            coefficients = self.coefficients
        y = self.model.predict(x=x, coefficients=coefficients)
        return y
        
        
            
if __name__ == '__main__':
    import sympy
    import numpy as np
    import matplotlib.pyplot as plt
    device = 'cpu'
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
        # Good approximations with both, however, never yields a simple formula
        # return module.sin((a[0] * x + 1) / (0.1 * x + 2))
        # return module.sin((a[0] * x + 1) / (0.1 * x))  # Works for lambda_1=0.1, lambda_denom=0 (not = 1)
        # return module.sin((a[0] * x))  # Works for lambda_1=0.1, lambda_denom=1
        # return a[0] * x / (1+x)
        return 0.2 * module.sin(a[0] * x) / x
        # return 0.5 * x / (x + 1)  # Only works when using normalize_denom=True and lambda_1>=0.3
        # return module.log(x + 1.4) + module.log(x ** 2 + 1.3)
        # return module.sin(x ** 2) * module.cos(x) - 1
        # return module.sin(x[0]) + module.sin(x[1]**2)

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

    functions = [torch.sin] * 2  # [lambda x: torch.log(torch.abs(x) + 0.000001)] * 2
    function_names = [sympy.sin] * 2  # [lambda x: sympy.log(sympy.Abs(x) + 0.000001)] * 2

    parfam = ParFamWrapper(iterate=True, functions=functions, function_names=function_names)
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
        
    
    