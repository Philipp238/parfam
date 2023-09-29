import numpy as np
import sympy
from sympy import symbols
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
import torch

from itertools import product
import copy
import time
import multiprocessing
# from torchmetrics import R2Score


# if torch.cuda.is_available():#
#     device = 'cuda'
#     # torch.cuda.synchronize()
# else:
#     device = 'cpu'
#
# # device = 'cpu'
#
# print(f'Using {device} (file: {__file__}))')


# Convert torch or numpy arrays/tensors to module arrays/tensors
def convert_to_module(x, module, device):
    if isinstance(x, np.ndarray) and module == torch:
        x = torch.tensor(x, device=device)
    if module == np and torch.is_tensor(x):
        x = x.cpu().detach().numpy()
    return x


def _symplify(queue):
    formula_dict = queue.get()
    formula_dict['Formula'] = sympy.simplify(formula_dict['Formula'])
    queue.put(formula_dict)
    return formula_dict


class ParFamTorch:
    def __init__(self, n_input, degree_input_numerator, degree_output_numerator, width=1, functions=[],
                 function_names=[], input_names=None, degree_input_numerator_specific=None,
                 degree_output_numerator_specific=None, degree_input_denominator=0, degree_output_denominator=0,
                 degree_input_denominator_specific=None, normalize_denom=True, maximal_potence=3,
                 degree_output_denominator_specific=None, enforce_function=False, device='cpu'):
        '''
        :param n_input: Number of input variables (the data should be in the form (number data points, n_input))
        :param degree_input_numerator: The degree of the numerator polynomials which are the input to the base functions
        :param degree_output_numerator: The degree of the numerator polynomials of the output rational
        :param width: Determines how often a base function should be used, e.g., if width=2 and the functions sin and
        exp are supposed to be used, then g_1 = sin, g_2 = sin, g_3 = cos and g_4 = cos. Default is 1, meaning that each
        function should only be used 1, which is the recommended setting.
        :param functions: List of the functions being used as torch functionals, e.g., [torch.sin, torch.exp]
        :param function_names: Names of the functions being used as sympy functionals, e.g., [sympy.sin, sympy.exp]
        :param input_names: Names of the input variables. If none, they will be just set to x_0, x_1,...
        :param degree_input_numerator_specific: Highest potence of the specific input variables in the input numerator
        polynomial. Should be set to None usually, then they will all be simply degree_input_numerator.
        :param degree_output_numerator_specific: Highest potence of each base function in the input numerator
        polynomial. If None, they will all be simply degree_output_numerator. It is recommendable, however, to specify
        them as for most base functions it is enough to allow 1 to be the highest potence, since the second potences of
        cos and exp etc can also be constructed using the first cos and exp.
        :param degree_input_denominator: The degree of the denominator polynomials which are the input to the base
        functions
        :param degree_output_denominator: The degree of the numerator polynomials of the output rational
        :param degree_input_denominator_specific: Highest potence of the specific input variables in the input numerator
        polynomial. Should be set to None usually, then they will all be simply degree_input_numerator.
        :param normalize_denom: If the denominator coefficients should be normalized, strongly recommended to be set to
        True, see Appendix B.1
        :param maximal_potence: Maximal potence occuring in any monomial
        :param degree_output_denominator_specific: Highest potence of each base function in the input numerator
        polynomial. If None, they will all be simply degree_output_numerator. It is recommendable, however, to specify
        them as for most base functions it is enough to allow 1 to be the highest potence, since the second potences of
        cos and exp etc can also be constructed using the first cos and exp.
        :param enforce_function: If it should be enforced that only coefficients of the oputput rational that refer to
         terms including one of the base functions is used. Default is False for higher expressivity.
        :param device: torch device
        '''
        # Input numerator/denominators refer to Q_1,...,Q_k
        # Output numerator/denominator refers to Q_{k+1}
        self.maximal_potence = maximal_potence
        self.n_input = n_input
        self.degree_input_numerator = degree_input_numerator
        self.degree_output_numerator = degree_output_numerator
        self.degree_input_denominator = degree_input_denominator
        self.degree_output_denominator = degree_output_denominator
        self.width = width
        self.functions = functions
        self.enforce_function = enforce_function
        self.device = device

        if (not (degree_input_denominator_specific is None)) or (not (degree_input_numerator_specific is None)):
            raise NotImplementedError(f'You specified degree_input_polynomials_denominator_specific or '
                                      f'degree_input_polynomials_specific, which is currently not supported. Please '
                                      f'set them to None.')

        # Numerator
        if degree_input_numerator_specific:
            if len(degree_input_numerator_specific) == len(functions) * width:
                self.degree_input_numerator_specific = self.n_input * [
                    self.maximal_potence] + degree_input_numerator_specific
            else:
                raise ValueError(
                    f'degree_input_polynomials_specific {degree_input_numerator_specific} has to have the same length'
                    f' as functions {functions} times width {width}.')
        else:
            self.degree_input_numerator_specific = [self.maximal_potence for _ in range(self.n_input)]
        if degree_output_numerator_specific:
            self.degree_output_numerator_specific = self.n_input * [
                self.maximal_potence] + degree_output_numerator_specific
        else:
            self.degree_output_numerator_specific = [self.maximal_potence for _ in
                                                     range(self.n_input + self.width * len(functions))]
        assert len(self.degree_output_numerator_specific) == self.n_input + self.width * len(
            self.functions)
        assert len(self.degree_input_numerator_specific) == self.n_input

        self.monomial_list_input = ParFamTorch.list_monomials(self.n_input, self.degree_input_numerator,
                                                              self.degree_input_numerator_specific)
        self.monomial_list_output = ParFamTorch.list_monomials(self.n_input + self.width * len(functions),
                                                               self.degree_output_numerator,
                                                               self.degree_output_numerator_specific)
        if self.enforce_function and len(self.functions) > 0:
            self.monomial_list_output = [monomial for monomial in self.monomial_list_output if
                                         sum(monomial[self.n_input:]) > 0]

        # Denominator
        if degree_input_denominator_specific:
            if degree_input_denominator_specific is None:
                self.degree_input_denominator_specific = self.n_input * [
                    self.maximal_potence]
            else:
                self.degree_input_denominator_specific = degree_input_denominator_specific
        else:
            self.degree_input_denominator_specific = self.n_input * [self.maximal_potence]
        if degree_output_denominator_specific:
            self.degree_output_denominator_specific = self.n_input * [
                self.maximal_potence] + degree_output_denominator_specific
        else:
            self.degree_output_denominator_specific = [self.maximal_potence for _ in
                                                       range(self.n_input + self.width * len(functions))]

        assert len(self.degree_output_denominator_specific) == self.n_input + self.width * len(
            self.functions)
        assert len(self.degree_input_denominator_specific) == self.n_input


        if self.degree_input_denominator > 0:
            self.monomial_list_input_denominator = ParFamTorch.list_monomials(self.n_input,
                                                                              self.degree_input_denominator,
                                                                              self.degree_input_denominator_specific)
        else:
            self.monomial_list_input_denominator = []
        if self.degree_output_denominator > 0:
            self.monomial_list_output_denominator = ParFamTorch.list_monomials(self.n_input,
                                                                               self.degree_output_denominator,
                                                                               self.degree_output_denominator_specific)
        else:
            self.monomial_list_output_denominator = []
        self.normalize_denom = normalize_denom
        self.n_coefficients_first_layer_numerator = len(self.monomial_list_input)
        self.n_coefficients_first_layer_denominator = len(self.monomial_list_input_denominator)
        self.n_coefficients_first_layer = len(self.monomial_list_input) + len(self.monomial_list_input_denominator)
        self.n_coefficients_last_layer_numerator = len(self.monomial_list_output)
        self.n_coefficients_last_layer_denominator = len(self.monomial_list_output_denominator)
        self.n_coefficients_last_layer = len(self.monomial_list_output) + len(self.monomial_list_output_denominator)
        self.input_monomials = None
        self.input_monomials_denominator = None
        self.output_monomials_dict = None
        self.output_monomials_denominator_dict = None
        self.function_names = function_names
        if input_names is None:
            self.input_names = [sympy.symbols(f'x{i}') for i in range(n_input)]
        else:
            self.input_names = input_names

    def get_formula(self, coefficients, decimals=3, verbose=False):
        if type(coefficients) == 'numpy':
            coefficients = torch.tensor(coefficients, device=self.device)
        if len(self.function_names) != len(self.functions):
            raise ValueError(f'Length of function_names {self.function_names} and functions {self.functions} is not the'
                             f' same, please fix this to compute the formula')
        feature_names = copy.copy(self.input_names)

        t_0 = time.time()
        for i, function_name in enumerate(self.function_names):
            current_coefficients = coefficients[
                                   i * self.n_coefficients_first_layer: i * self.n_coefficients_first_layer
                                                                        + self.n_coefficients_first_layer_numerator]
            numerator = ParFamTorch._get_symbolic_polynomial(self.n_input, self.input_names,
                                                             self.degree_input_numerator, current_coefficients,
                                                             self.degree_input_numerator_specific,
                                                             self.monomial_list_input, decimals=decimals)
            if self.degree_input_denominator > 0:
                current_coefficients = coefficients[i * self.n_coefficients_first_layer
                                                    + self.n_coefficients_first_layer_numerator:
                                                    (i + 1) * self.n_coefficients_first_layer]
                if self.normalize_denom:
                    current_coefficients = current_coefficients / torch.norm(current_coefficients)
                denominator = ParFamTorch._get_symbolic_polynomial(self.n_input, self.input_names,
                                                                   self.degree_input_denominator, current_coefficients,
                                                                   self.degree_input_denominator_specific,
                                                                   self.monomial_list_input_denominator,
                                                                   decimals=decimals)
            else:
                denominator = 1.0
            if isinstance(denominator, float) and abs(denominator) < 10 ** (-4):
                denominator = 10 ** (-4)
            feature_name = function_name(numerator / denominator)
            feature_names.append(feature_name)
        t_1 = time.time()
        if verbose:
            print(f'Computing the formula for the input layer took {(t_1 - t_0):.3f} seconds')

        t_0 = time.time()
        if self.degree_output_denominator > 0:
            current_coefficients = coefficients[
                                   -self.n_coefficients_last_layer:-self.n_coefficients_last_layer_denominator]
        else:
            current_coefficients = coefficients[-self.n_coefficients_last_layer:]
        numerator = ParFamTorch._get_symbolic_polynomial(len(feature_names), feature_names,
                                                         self.degree_output_numerator,
                                                         current_coefficients, self.degree_output_numerator_specific,
                                                         multiindices=self.monomial_list_output, decimals=decimals)
        if self.degree_output_denominator > 0:
            current_coefficients = coefficients[-self.n_coefficients_last_layer_denominator:]
            if self.normalize_denom:
                current_coefficients = current_coefficients / torch.norm(current_coefficients)
            denominator = ParFamTorch._get_symbolic_polynomial(len(feature_names), feature_names,
                                                               self.degree_output_denominator,
                                                               current_coefficients,
                                                               self.degree_output_denominator_specific,
                                                               multiindices=self.monomial_list_output_denominator,
                                                               decimals=decimals)
        else:
            denominator = 1
        formula = numerator / denominator
        t_1 = time.time()
        if verbose:
            print(f'Computing the end formula took {(t_1 - t_0):.3f} seconds')

        t_0 = time.time()
        # try:
        #    formula = sympy.simplify(formula)
        # except TimeoutError:
        #     print(f'Simplifying took too long, so we omit it this time.')
        if isinstance(coefficients[0], float):
            if verbose:
                print(f'Estimated expression (before simplification): {formula}')
            timelimit = True
            if timelimit:
                formula_dict = {'Formula': formula}
                queue = multiprocessing.Queue()
                queue.put(formula_dict)
                p = multiprocessing.Process(target=_symplify, args=(queue,))
                p.start()

                # Wait for 10 seconds or until process finishes
                p.join(10)

                # If thread is still active
                if p.is_alive():
                    print("running... let's kill it...")

                    # Terminate - may not work if process is stuck for good
                    p.terminate()
                    # OR Kill - will work for sure, no chance for process to finish nicely however
                    # p.kill()

                    p.join()
                else:
                    p.terminate()
                    p.join()
                    formula = queue.get()['Formula']
            else:
                formula = sympy.simplify(formula)
        t_1 = time.time()
        if verbose:
            print(f'Simplifying the formula took {(t_1 - t_0):.3f} seconds')
        return formula

    @staticmethod
    def _get_symbolic_polynomial(n_input, input_names, degree, coefficients, degrees_specific, multiindices,
                                 decimals):
        formula = 0
        # multiindices = ParFamTorch.list_monomials(n_input, degree, degrees_specific)

        for i, multiindex in enumerate(multiindices):
            coefficient = coefficients[i]
            if isinstance(coefficient, float) and np.abs(coefficient) < 0.000000001:  # 10**(-decimals)
                continue
            if isinstance(coefficient, torch.Tensor) and torch.abs(coefficient) < 0.000000001:  # 10**(-decimals)
                continue
            monomial = 1
            for j, index in enumerate(multiindex):
                monomial *= input_names[j] ** index
            if isinstance(coefficient, float):
                formula += np.round(coefficient, decimals=3) * monomial
            elif isinstance(coefficient, torch.Tensor):
                formula += np.round(coefficient.cpu().detach().numpy(), decimals=3) * monomial
            else:
                formula += coefficient * monomial
        return formula

    def get_number_parameters(self):
        return self.width * len(self.functions) * self.n_coefficients_first_layer + self.n_coefficients_last_layer

    def prepare_input_monomials(self, x):
        # Convert x to torch tensor if x is a numpy array
        # x = convert_to_module(x, torch, device=self.device)
        self.input_monomials = ParFamTorch._evaluate_monomial_features_v1(x, self.monomial_list_input,
                                                                          device=self.device)
        if self.degree_input_denominator > 0:
            self.input_monomials_denominator = \
                ParFamTorch._evaluate_monomial_features_v1(x, self.monomial_list_input_denominator, device=self.device)
        self.output_monomials_dict = self._compute_output_monomial_features_dict(x, self.monomial_list_output)
        if self.degree_output_denominator > 0:
            self.output_monomials_denominator_dict = \
                self._compute_output_monomial_features_dict(x, self.monomial_list_output_denominator)

    def testing_mode(self):
        self.input_monomials = None
        self.input_monomials_denominator = None
        self.output_monomials_dict = None
        self.output_monomials_denominator_dict = None

    def predict(self, coefficients, x, symbolic=False):
        # x = convert_to_module(x, torch, device=self.device)
        # coefficients = convert_to_module(coefficients, torch, device=self.device)
        if self.input_monomials is None:
            input_monomials = ParFamTorch._evaluate_monomial_features_v1(x, self.monomial_list_input,
                                                                         device=self.device)
        else:
            input_monomials = copy.copy(self.input_monomials)
        if self.input_monomials_denominator is None and self.degree_input_denominator > 0:
            input_monomials_denominator = ParFamTorch._evaluate_monomial_features_v1(x,
                                                                                     self.monomial_list_input_denominator,
                                                                                     device=self.device)
        else:
            input_monomials_denominator = copy.copy(self.input_monomials_denominator)
        hidden_layer = torch.zeros((x.shape[0], x.shape[1] + self.width * len(self.functions)), device=self.device)
        hidden_layer[:, :x.shape[1]] = x
        feature_number = x.shape[1]
        for i, function in enumerate(self.width * self.functions):
            current_coefficients = coefficients[
                                   i * self.n_coefficients_first_layer: i * self.n_coefficients_first_layer
                                                                        + self.n_coefficients_first_layer_numerator]
            numerator = ParFamTorch._evaluate_polynomial_v2(inputs=x, coefficients=current_coefficients,
                                                            multiindices=self.monomial_list_input,
                                                            monomial_features=input_monomials, device=self.device)
            if self.degree_input_denominator > 0:
                current_coefficients = coefficients[
                                       i * self.n_coefficients_first_layer + self.n_coefficients_first_layer_numerator:
                                       (i + 1) * self.n_coefficients_first_layer]
                if self.normalize_denom:
                    current_coefficients = current_coefficients / torch.linalg.norm(current_coefficients)
                denominator = ParFamTorch._evaluate_polynomial_v2(inputs=x, coefficients=current_coefficients,
                                                                  multiindices=self.monomial_list_input_denominator,
                                                                  monomial_features=input_monomials_denominator,
                                                                  device=self.device)
                denominator = self.stabilize_denominator(denominator)
            else:
                denominator = 1

            try:
                hidden_layer[:, feature_number + i] = function(numerator / denominator)
            except Exception as e:
                pass
        if self.degree_output_denominator > 0:
            current_coefficients = coefficients[
                                   -self.n_coefficients_last_layer:-self.n_coefficients_last_layer_denominator]
        else:
            current_coefficients = coefficients[-self.n_coefficients_last_layer:]
        numerator = ParFamTorch._evaluate_polynomial_v2(inputs=hidden_layer, coefficients=current_coefficients,
                                                        multiindices=self.monomial_list_output,
                                                        monomial_features=None, n_input=self.n_input,
                                                        monomial_features_dict=self.output_monomials_dict,
                                                        device=self.device)
        if self.degree_output_denominator > 0:
            current_coefficients = coefficients[-self.n_coefficients_last_layer_denominator:]
            if self.normalize_denom:
                current_coefficients = current_coefficients / torch.norm(current_coefficients)
            denominator = ParFamTorch._evaluate_polynomial_v2(inputs=hidden_layer, coefficients=current_coefficients,
                                                              multiindices=self.monomial_list_output_denominator,
                                                              monomial_features=None, n_input=self.n_input,
                                                              monomial_features_dict=self.output_monomials_denominator_dict,
                                                              device=self.device)
            denominator = self.stabilize_denominator(denominator)
        else:
            denominator = 1
        return numerator / denominator

    def _compute_output_monomial_features_dict(self, inputs, multiindices):
        monomial_feature_dict = {}
        for i, multiindex in enumerate(multiindices):
            reduced_multiindex = multiindex[:inputs.shape[1]]
            if reduced_multiindex not in monomial_feature_dict.keys():
                monomial_feature_dict[reduced_multiindex] = torch.ones(inputs.shape[0], dtype=torch.float64,
                                                                       device=self.device)
                for j, index in enumerate(reduced_multiindex):
                    if index == 0:
                        continue
                    elif index == 1:
                        monomial_feature_dict[reduced_multiindex] *= inputs[:, j]
                    else:
                        monomial_feature_dict[reduced_multiindex] *= inputs[:, j] ** index
        return monomial_feature_dict

    def get_normalized_coefficients(self, coefficients):
        # coefficients = convert_to_module(coefficients, torch, device=self.device)
        for i, function in enumerate(self.width * self.functions):
            if self.degree_input_denominator > 0 and self.normalize_denom:
                denominator_indices = slice(
                    i * self.n_coefficients_first_layer + self.n_coefficients_first_layer_numerator, (
                            i + 1) * self.n_coefficients_first_layer)
                coefficients[denominator_indices] /= torch.norm(coefficients[denominator_indices], p=2)

        if self.degree_output_denominator > 0 and self.normalize_denom:
            denominator_indices = slice(-self.n_coefficients_last_layer_denominator, None)
            coefficients[denominator_indices] /= torch.norm(coefficients[denominator_indices], p=2)
        return coefficients

    def stabilize_denominator(self, denominator):
        denominator[torch.abs(denominator) < 10 ** (-5)] = 10 ** (-5)
        return denominator

    def denominator_reg(self, coefficients):
        reg = 0
        if self.degree_input_denominator > 0:
            for i, function in enumerate(self.width * self.functions):
                current_coefficients = coefficients[
                                       i * self.n_coefficients_first_layer + self.n_coefficients_first_layer_numerator:
                                       (i + 1) * self.n_coefficients_first_layer]
                reg += (torch.norm(current_coefficients) - 1) ** 2
        if self.degree_input_denominator > 0:
            current_coefficients = coefficients[-self.n_coefficients_last_layer_denominator:]
            reg += (torch.norm(current_coefficients) - 1) ** 2
        return reg

    def predict_batch(self, coefficients, x, symbolic=False):
        batch_size = coefficients.shape[1]
        if self.input_monomials is None:
            input_monomials = ParFamTorch._evaluate_monomial_features_v1(x, self.monomial_list_input,
                                                                         device=self.device)
        else:
            input_monomials = copy.copy(self.input_monomials)
        hidden_layer = np.zeros((x.shape[0], x.shape[1] + self.width * len(self.functions), batch_size))
        hidden_layer[:, :x.shape[1]] = np.repeat(x.reshape(*x.shape, 1), batch_size, axis=2)
        feature_number = x.shape[1]
        for i, function in enumerate(self.width * self.functions):
            current_coefficients = coefficients[
                                   i * self.n_coefficients_first_layer: (i + 1) * self.n_coefficients_first_layer]
            hidden_layer[:, feature_number + i] = function(
                ParFamTorch._evaluate_polynomial_v2(inputs=x, coefficients=current_coefficients,
                                                    multiindices=self.monomial_list_input,
                                                    monomial_features=input_monomials, device=self.device))

        current_coefficients = coefficients[-self.n_coefficients_last_layer:]
        output = ParFamTorch._evaluate_polynomial_v2(inputs=hidden_layer, coefficients=current_coefficients,
                                                     multiindices=self.monomial_list_output,
                                                     monomial_features=None, batch_size=batch_size, device=self.device)
        return output

    def get_mixed_reg(self, n_features, coefficients):
        penalty = 0
        for feature in range(n_features + self.width * len(self.functions)):
            active_feature_coefficients = []
            if feature < n_features:
                continue
            for i, index in enumerate(self.monomial_list_output):
                active_feature_coefficients.append(index[feature] > 0)
            if sum(np.abs(coefficients[-self.n_coefficients_last_layer:][active_feature_coefficients]) > 0.01) > 1:
                penalty += torch.norm(
                    coefficients[-self.n_coefficients_last_layer:][active_feature_coefficients],
                    p=1)
        return penalty

    def get_random_coefficients_unique(self, n_functions_max=3):
        """
        Get random coefficients, which relate to meaningful functions.
        :param n_functions_max: Number of features to be used, i.e., at most n_functions_max output coefficients are
                                non-zero
        :return: numpy array, which gives the coefficients
        """
        input_coefficients = np.zeros(self.n_coefficients_first_layer * self.width * len(self.functions))
        output_coefficients = np.zeros(self.n_coefficients_last_layer)

        monomial_list_output_lists = [list(multi_index) for multi_index in self.monomial_list_output]

        # Get the active output monomials
        n_functions = 1 + np.random.randint(n_functions_max)
        indices = np.random.choice(len(monomial_list_output_lists), size=n_functions,
                                   replace=False)
        active_output_monomials = []
        for index in indices:
            active_output_monomials.append(monomial_list_output_lists[index])
        # Check if there is any input feature which is used with more than one different degree, e.g.,
        # sin(P_1(x)) and sin(P_1(x))^2. Drop them if yes.
        dict_active_output_monomials = {}
        active_output_monomials_to_remove = []
        for multi_index in active_output_monomials:
            for index, multiplicity in enumerate(multi_index):
                if index < self.n_input or multiplicity == 0:
                    continue
                if index in dict_active_output_monomials.keys():
                    if dict_active_output_monomials[index] != multiplicity:
                        # remove the coefficient if this relates to an already used feature, but with a different degree
                        # this time
                        active_output_monomials_to_remove.append(multi_index)
                        break
                else:
                    dict_active_output_monomials[index] = multiplicity
        for monomial_to_remove in active_output_monomials_to_remove:
            active_output_monomials.remove(monomial_to_remove)

        # Recreate the dict_active_output_monomials to ensure that the removal did not change it
        dict_active_output_monomials = {}
        active_output_monomials_to_remove = []
        for multi_index in active_output_monomials:
            for index, multiplicity in enumerate(multi_index):
                if index < self.n_input or multiplicity == 0:
                    continue
                dict_active_output_monomials[index] = multiplicity

        # Ensure the uniqueness of the formula through the following steps:
        # 1. If only one function is used, cosine is used ==> So if (0,0,1) is the only active coefficient, than it will
        # be substituted by (0,1,0)
        active_output_monomials_to_remove = []
        active_output_monomials_to_add = []
        if not 1 in dict_active_output_monomials.keys() and 2 in dict_active_output_monomials.keys():
            for multi_index in active_output_monomials:
                if multi_index[2] != 0:
                    # active_output_monomials_to_remove.append(multi_index)
                    multi_index[1] = multi_index[2]
                    multi_index[2] = 0
        # 2. If both functions are used, but with different degrees, ensure that cosine has a lower degree than sine
        # ==> So if (0,2,0) and (0,0,1) are active they will both be substituted by (0,0,2) and (0,1,0)
        if len(dict_active_output_monomials) == 2:
            if dict_active_output_monomials[1] > dict_active_output_monomials[2]:
                # swap the multiplicities
                dict_active_output_monomials[1], dict_active_output_monomials[2] = dict_active_output_monomials[2], \
                    dict_active_output_monomials[1]
                for multi_index in active_output_monomials:
                    if multi_index[1] != 0:
                        multi_index[1] = dict_active_output_monomials[1]
                    if multi_index[2] != 0:
                        multi_index[2] = dict_active_output_monomials[2]
                        if sum(multi_index) > self.degree_output_numerator:
                            multi_index[2] = dict_active_output_monomials[1]
        # 3. If both functions are used with the same degree, make sure that
        # a) if there is only 1 active monomial including either cos or sin, we are done.
        # b) if there are 2 active monomials including either cos or sin, make sure that cos is the function used in the
        # monomial with lower overall degree ==> i.e., (2,2,0) and (1,0,2) will be substituted by (1,2,0) and (2,0,2).
        # If this is also the same, e.g., for (0,2,2) and (2,0,2), then the coefficients are chosen such that cosine
        # is active in the one without sine, so they will be substituted by (0,2,2) and (2,2,0).
        # c)
        if len(dict_active_output_monomials) == 2 and dict_active_output_monomials[1] == dict_active_output_monomials[
            2]:
            degree = dict_active_output_monomials[1]
            active_output_monomials_cos_sin = []
            for multi_index in active_output_monomials:
                if multi_index[1] != 0 or multi_index[2] != 0:
                    active_output_monomials_cos_sin.append(multi_index)

            # Case b)
            if len(active_output_monomials_cos_sin) == 2:
                active_output_monomials_cos = []
                active_output_monomials_sin = []
                for multi_index in active_output_monomials_cos_sin:
                    if multi_index[1] != 0:
                        active_output_monomials_cos.append(multi_index)
                    if multi_index[2] != 0:
                        active_output_monomials_sin.append(multi_index)
                if len(active_output_monomials_cos) == 1:
                    # When cos is active all the time it is not a problem
                    if len(active_output_monomials_sin) == 2:
                        # Case b) and sine is more often used than cosine ==> swap
                        for multi_index in active_output_monomials_sin:
                            if multi_index[1] == 0:
                                multi_index[1] = degree
                                multi_index[2] = 0
                    else:
                        # both are only used once
                        if sum(active_output_monomials_cos[0]) > sum(active_output_monomials_sin[0]):
                            # Cosines multi index has a higher overall degree ==> swap
                            active_output_monomials_cos[0][1] = 0
                            active_output_monomials_cos[0][2] = degree
                            active_output_monomials_sin[0][1] = degree
                            active_output_monomials_sin[0][2] = 0

            if len(active_output_monomials_cos_sin) == 3:
                # TODO
                pass

        active_output_monomials_indices = []
        for output_polynomial in active_output_monomials:
            index = self.monomial_list_output.index(tuple(output_polynomial))
            active_output_monomials_indices.append(index)
            output_coefficients[index] = 1

        # Get the partitions in the input layer which is related to the active functions
        active_input_partitions = set()
        for i, multi_index in enumerate(active_output_monomials):
            for index, multiplicity in enumerate(multi_index):
                if index < self.n_input or multiplicity == 0:
                    continue
                active_input_partitions.add(index - self.n_input)

        # Select randomly the active coefficients in the partitions
        for active_input_partition in active_input_partitions:
            n_coeffs = 1 + np.random.randint(self.n_coefficients_first_layer)
            if n_coeffs == 1:
                # make sure that not only the coefficient for the constant term is chosen
                chosen_coeffs = np.random.choice(range(self.n_coefficients_first_layer - 1), size=n_coeffs) + 1
            else:
                chosen_coeffs = np.random.choice(range(self.n_coefficients_first_layer), size=n_coeffs,
                                                 replace=False)
            for chosen_coeff in chosen_coeffs:
                input_coefficients[active_input_partition * self.n_coefficients_first_layer + chosen_coeff] = 1
        return np.concatenate([input_coefficients, output_coefficients])

    def get_random_coefficients(self, n_functions_max=3):
        """
        Get random coefficients, which relate to meaningful functions.
        :param n_functions_max: Number of features to be used, i.e., at most n_functions_max output coefficients are
                                non-zero
        :return: numpy array, which gives the coefficients
        """
        input_coefficients = np.zeros(self.n_coefficients_first_layer * self.width * len(self.functions))
        output_coefficients = np.zeros(self.n_coefficients_last_layer)

        # Get the active output coefficients
        n_functions = 1 + np.random.randint(n_functions_max)
        active_output_monomials_indices = np.random.choice(range(len(self.monomial_list_output)), size=n_functions,
                                                           replace=False)
        for output_index in active_output_monomials_indices:
            output_coefficients[output_index] = 1

        # Get the related input coefficients
        active_output_monomials = [self.monomial_list_output[index] for index in active_output_monomials_indices]
        # Get the partitions in the input layer which is related to the active functions
        active_input_partitions = set()
        dict_active_output_monomials = {}
        for i, multi_index in enumerate(active_output_monomials):
            for index, multiplicity in enumerate(multi_index):
                if index < self.n_input or multiplicity == 0:
                    continue
                # Check if there is any input feature which is used with more than one different degree, e.g.,
                # sin(P_1(x)) and sin(P_1(x))^2
                # By restricting the potence of the functions, this step is not important.
                if index in dict_active_output_monomials.keys():
                    if dict_active_output_monomials[index] == multiplicity:
                        active_input_partitions.add(index - self.n_input)
                    else:
                        # remove the coefficient if this relates to an already used feature, but with a different degree
                        # this time
                        output_coefficients[active_output_monomials_indices[i]] = 0
                else:
                    dict_active_output_monomials[index] = multiplicity
                    active_input_partitions.add(index - self.n_input)

        # Select randomly the active coefficients in the partitions
        for active_input_partition in active_input_partitions:
            n_coeffs = 1 + np.random.randint(self.n_coefficients_first_layer)
            if n_coeffs == 1:
                # make sure that not only the coefficient for the constant term is chosen
                chosen_coeffs = np.random.choice(range(self.n_coefficients_first_layer - 1), size=n_coeffs) + 1
            else:
                chosen_coeffs = np.random.choice(range(self.n_coefficients_first_layer), size=n_coeffs,
                                                 replace=False)
            for chosen_coeff in chosen_coeffs:
                input_coefficients[active_input_partition * self.n_coefficients_first_layer + chosen_coeff] = 1
        return np.concatenate([input_coefficients, output_coefficients])

    @staticmethod
    def _evaluate_polynomial_v2(inputs, coefficients, multiindices, device, monomial_features=None, batch_size=1,
                                monomial_features_dict=None, n_input=None):
        if monomial_features is None:
            monomial_features = ParFamTorch._evaluate_monomial_features_v1(inputs, multiindices, batchsize=batch_size,
                                                                           monomial_features_dict=monomial_features_dict,
                                                                           n_input=n_input, device=device)
        if batch_size == 1:
            if monomial_features.shape[1] != len(coefficients):
                print('Stop')
            return torch.matmul(monomial_features, coefficients)  # No batches, simple matrix-vector product
        else:
            # Batches, so we use the matrix-vector product over the left-most indices
            return torch.einsum('ij...,j...->i...', monomial_features, coefficients)

    @staticmethod
    def list_monomials_uniform_degree(n_input, degree, device):
        multi_indices = []
        indices = torch.arange(degree + 1, device=device)
        repeat_indices = [indices for _ in range(n_input)]
        for i in product(*repeat_indices):
            if sum(i) <= degree:
                multi_indices += [i]
        return multi_indices

    @staticmethod
    def list_monomials(n_input, degree, degrees_specific):
        multi_indices = []
        repeat_indices = [range(degree_specific + 1) for degree_specific in degrees_specific]
        for i in product(*repeat_indices):
            if sum(i) <= degree:
                multi_indices += [i]
        return multi_indices

    @staticmethod
    def _evaluate_monomial_features_v1(inputs, multiindices, device, batchsize=1, monomial_features_dict=None,
                                       n_input=None):
        if batchsize == 1:
            monomial_features = torch.ones((inputs.shape[0], len(multiindices)), dtype=torch.float64, device=device)
        else:
            monomial_features = torch.ones((inputs.shape[0], len(multiindices), batchsize), dtype=torch.float64,
                                           device=device)
        if monomial_features_dict is None:
            for i, multiindex in enumerate(multiindices):
                for j, index in enumerate(multiindex):
                    if index == 0:
                        continue
                    elif index == 1:
                        monomial_features[:, i] *= inputs[:, j]
                    else:
                        monomial_features[:, i] *= inputs[:, j] ** index
        else:
            for i, multiindex in enumerate(multiindices):
                reduced_multiindex_input = multiindex[:n_input]
                reduced_multiindex_func = multiindex[n_input:]
                monomial_features[:, i] = monomial_features_dict[reduced_multiindex_input]
                for j, index in enumerate(reduced_multiindex_func):
                    if index == 0:
                        continue
                    elif index == 1:
                        monomial_features[:, i] *= inputs[:, j + n_input]
                    else:
                        monomial_features[:, i] *= inputs[:, j + n_input] ** index
        return monomial_features

    @staticmethod
    def get_monomial_mask(n_input, degree):
        half_degree = int(torch.ceil(degree / 2))
        inputs = torch.ones((1, n_input)) * 2
        y = inputs.reshape(*inputs.shape, 1) ** (torch.arange(half_degree) + 1).reshape(1, 1, half_degree)
        y_flattened = y.reshape(inputs.shape[0], half_degree * n_input)
        y_flattened_with_1 = torch.concatenate((torch.ones((y.shape[0], 1), device=device), y_flattened), axis=1)
        z = torch.expand_dims(y_flattened_with_1, axis=1) * torch.expand_dims(y_flattened_with_1, axis=2)
        z = torch.squeeze(z, axis=0)
        return (0 < z.triu()) & (z.triu() <= 2 ** degree)


class Evaluator:

    def __init__(self, x, y, model, lambda_0, lambda_1, n_params, lambda_mixed=0, lambda_denom=0,
                 n_best_coefficients=10, mask=None, lambda_05=None, lambda_1_cut=None, lambda_1_piecewise=None):
        """
        :param x: The feature matrix
        :param y: The target variable
        :param model: An instance of ParFamTorch
        :param lambda_0: Regularization parameter for the "0-norm"
        :param lambda_1: Regularization parameter for the 1-norm
        :param n_params: Number of parameters of the ParFamTorch model
        :param lambda_mixed: Regularization parameter for the "mixed norm" (not recommended)
        :param lambda_denom: Regularization parameter for the denominator (not recommended)
        :param n_best_coefficients: How many of the best coefficients should be safed (deprecated feature)
        :param mask: Mask in case some of the coefficients should be set to 0
        :param lambda_05: Regularization parameter for the "0.5-norm" (not recommended)
        :param lambda_1_cut: Regularization parameter for the "cutoff norm" (not recommended)
        :param lambda_1_piecewise: Regularization parameter for the "piecewise norm" (not recommended)
        """
        self.x = x
        self.y = y
        self.model = model
        self.lambda_0 = lambda_0  # regularization factor for p=0 regularization (cutoff value of 0.01)
        self.lambda_1 = lambda_1  # regularization factor for p=1 regularization
        self.lambda_1_piecewise = lambda_1_piecewise  # regularization factor for piecewise p=1 regularization: the
        # smaller coefficient, the higher the regularization parameter
        self.lambda_1_cut = lambda_1_cut  # regularization factor for f(x)=min(|x|, \sqrt(x))
        self.lambda_05 = lambda_05  # regularization factor for p=0.5 regularization
        self.lambda_mixed = lambda_mixed  # regularization factor for functions which are used more than once; default
        self.lambda_denom = lambda_denom  # regularization factor for the coefficients of the denominator, to keep their
        # norm close to 1, as otherwise the values of all parameters can be pressed to 0 due to overparametrization
        self.loss_list = []
        self.l2_dist_list = []
        self.reg_list = []
        self.n_active_parameters_list = []
        self.best_losses = np.full((n_best_coefficients,), float('inf'))
        self.mask = mask
        self.n_params = n_params
        self.device = self.model.device

        self.coefficients_current = None
        self.loss_current = None
        if self.mask == None:
            self.best_coefficients = np.inf * np.ones((n_best_coefficients, n_params))
        else:
            self.n_active_coefficients = sum(mask)
            self.best_coefficients = np.inf * np.ones((n_best_coefficients, self.n_active_coefficients))
        self.evaluations = 0

    # def convert_evaluator_variables_to_module(self, module):
    #    self.x = convert_to_module(self.x, module, device=self.device)
    #    self.y = convert_to_module(self.y, module, device=self.device)
    #    self.best_coefficients = convert_to_module(self.best_coefficients, module, device=self.device)
    #    self.best_losses = convert_to_module(self.best_losses, module, device=self.device)

    def loss_func_torch(self):
        # convert array to tensors?
        # self.convert_evaluator_variables_to_module(torch)
        self.evaluations += 1
        if self.mask is None:
            y_pred = self.model.predict(self.coefficients_current, self.x)
        else:
            coefficients_extended = torch.zeros(self.n_params, device=self.device, dtype=torch.double)
            coefficients_extended[self.mask] = self.coefficients_current
            y_pred = self.model.predict(coefficients_extended, self.x)
        if y_pred.shape != self.y.shape:
            print(f'Careful, there is a shape mismatch in the loss function: '
                  f'y_pred.shape {y_pred.shape} != y.shape {self.y.shape}')
        # Use torch.norm since torch.linalg.norm does not support tensors with requires grad
        rel_l2_dist = torch.norm(y_pred - self.y, p=2) / torch.norm(self.y, p=2)
        n_active_parameters = torch.sum(torch.abs(self.coefficients_current) > 0.01)
        reg_0 = self.lambda_0 * n_active_parameters
        reg_1 = self.lambda_1 * torch.norm(self.coefficients_current, p=1)
        reg_1_piecewise = 0
        if self.lambda_1_piecewise:
            reg_1_piecewise += torch.norm(
                self.coefficients_current[self.coefficients_current < 0.1]) * self.lambda_1_piecewise
            reg_1_piecewise += torch.norm(self.coefficients_current[(0.1 < self.coefficients_current) & (
                    self.coefficients_current < 1)]) * self.lambda_1_piecewise * 0.1 + self.lambda_1_piecewise * 0.1
            reg_1_piecewise += torch.norm(
                self.coefficients_current[self.coefficients_current > 1]) * self.lambda_1_piecewise * 0.01 + \
                               self.lambda_1_piecewise * 0.1 + self.lambda_1_piecewise * 0.9 * 0.1

        if self.lambda_05:
            reg_05 = self.lambda_05 * torch.norm(self.coefficients_current, p=1)
        else:
            reg_05 = 0
        if self.lambda_1_cut:
            reg_1_cut = self.lambda_1_cut * torch.sum(
                torch.minimum(torch.abs(self.coefficients_current), torch.sqrt(torch.abs(self.coefficients_current))))
        else:
            reg_1_cut = 0

        # Penalizing it when one term is used more than once seemed like a good idea, but apparently it rather harms the
        # problem at hand
        if self.lambda_mixed > 0:
            reg_2 = self.model.get_mixed_reg(n_features=self.x.shape[1], coefficients=self.coefficients_current)
        else:
            reg_2 = 0  # Default
        if self.lambda_denom:
            reg_3 = self.lambda_denom * self.model.denominator_reg(self.coefficients_current)
        else:
            reg_3 = 0
        self.loss_current = rel_l2_dist + reg_0 + reg_1 + reg_2 + reg_3 + reg_05 + reg_1_cut + reg_1_piecewise
        if False:
            self.l2_dist_list.append(rel_l2_dist)
            self.reg_list.append(reg_0 + reg_1)
            self.loss_list.append(self.loss_current)
            self.n_active_parameters_list.append(n_active_parameters)

        # if self.loss_current < self.best_losses[-1]:
        #    self.best_losses[-1] = self.loss_current
        #    self.best_coefficients[-1] = self.coefficients_current
        #    order = torch.argsort(self.best_losses)
        #    self.best_losses = self.best_losses[order]
        #    self.best_coefficients = self.best_coefficients[order]
        return 0

    def gradient(self, _):
        # The if-query should be unnecessary. Test later for specific solver....
        # if np.linalg.norm(coefficients - self.coefficients_current.detach().numpy()) > 1e-3:
        #     coefficients = convert_to_module(coefficients, torch)
        #     coefficients.requires_grad_()
        #     self.loss_current = self.loss_func_torch(coefficients)
        self.loss_current.backward()
        return convert_to_module(self.coefficients_current.grad, np, device=self.device)

    def loss_func(self, coefficients):
        # self.coefficients_current = convert_to_module(coefficients, torch, device=self.device)
        self.coefficients_current = torch.from_numpy(coefficients).to(self.device)
        self.coefficients_current.requires_grad_()
        self.loss_func_torch()
        return (self.loss_current).cpu().detach().numpy()

    # def loss_func(self, coefficients):
    #     self.coefficients_current = convert_to_module(coefficients, torch)
    #     self.coefficients_current.requires_grad_()
    #     self.loss_func_torch()
    #     self.loss_current.backward()
    #     gradient = convert_to_module(self.coefficients_current.grad, np)
    #     return convert_to_module(self.loss_current, np), gradient

    def plot_training_statistics(self, width, height):
        fig, axs = plt.subplots(2, 2, figsize=(width, height))
        axs[0, 0].plot(self.loss_list)
        axs[0, 0].set_yscale('log')
        axs[0, 0].set_title(f'Loss')

        axs[1, 0].plot(self.l2_dist_list)
        axs[1, 0].set_yscale('log')
        axs[1, 0].set_title(f'Training l2 distance')

        axs[0, 1].plot(self.reg_list)
        # axs[0, 1].set_yscale('log')
        axs[0, 1].set_title(f'L0 + L1 Regularization')

        axs[1, 1].plot(self.n_active_parameters_list)
        axs[1, 1].set_ylim([0, np.max(self.n_active_parameters_list) + 1])
        # axs[1, 1].set_yscale('log')
        axs[1, 1].set_title(f'Number active terms')

        plt.tight_layout()

    def fit_lbfgs(self, coefficients, verbose):
        t_0 = time.time()
        coefficients.requires_grad = True
        self.coefficients_current = coefficients
        optimizer = torch.optim.LBFGS([self.coefficients_current], line_search_fn='strong_wolfe')
        for i in range(50):
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                self.loss_func_torch()
                loss = self.loss_current
                if loss.requires_grad:
                    loss.backward()
                return loss

            optimizer.step(closure)
        # ret = basinhopping(evaluator.gradient, niter=50, x0=x0, minimizer_kwargs={'jac': True})
        t_1 = time.time()
        if verbose:
            print(f'Training time: {t_1 - t_0}')
        self.model.testing_mode()
        if verbose:
            print(f'Coefficients: {coefficients}')
        if self.mask is None:
            y_pred = self.model.predict(self.coefficients_current, self.x)
        else:
            coefficients_extended = torch.zeros(self.n_params, device=self.device, dtype=torch.double)
            coefficients_extended[self.mask] = self.coefficients_current
            y_pred = self.model.predict(coefficients_extended, self.x)
        relative_l2_distance = torch.norm(self.y - y_pred, p=2) / torch.norm(self.y, p=2)
        if verbose:
            print(f'Relative l2 distance: {relative_l2_distance}')
        return coefficients, relative_l2_distance


def round_expr(expr, num_digits):
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(sympy.Number)})


def get_active_coefficients(coefficients, model, x):
    y = model.predict(coefficients, x)
    a = np.zeros(coefficients.shape)
    for i, coefficient in enumerate(coefficients):
        if coefficient == 0:
            continue
        distorted_coefficient_array = 10 * np.random.randn(10)
        for distorted_coefficient in distorted_coefficient_array:
            test_coefficients = copy.copy(coefficients)
            test_coefficients[i] = distorted_coefficient
            if np.linalg.norm(y - model.predict(test_coefficients, x)) > 10 ** (-8):
                a[i] = 1
                break
    return a


def dev_training(device):
    np.random.seed(12345)
    torch.manual_seed(12345)
    print(f'Using {device}')
    a = 5 * torch.randn(1)
    x = np.arange(1, 10, 0.05)
    x = x.reshape(len(x), 1)
    x = torch.tensor(x, device=device)
    test_model = False

    def func(a, x, module):
        # Good approximations with both, however, never yields a simple formula
        return module.sin((a[0] * x + 1) / (0.1 * x + 2))
        # return module.sin((a[0] * x + 1) / (0.1 * x))  # Works for lambda_1=0.1, lambda_denom=0 (not = 1)
        # return module.sin((a[0] * x))  # Works for lambda_1=0.1, lambda_denom=1
        # return a[0] * x
        # return 0.2 * module.sin(a[0] * x) / x
        # return 0.5 * x / (x + 1)  # Only works when using normalize_denom=True and lambda_1>=0.3

    y = func(a, x, torch).squeeze(-1)
    print(f'Target formula: {func(a, sympy.Symbol("x"), sympy)}')

    functions = [torch.sin]
    function_names = [sympy.sin]
    model = ParFamTorch(n_input=1, degree_input_numerator=2, degree_output_numerator=2, width=1,
                        functions=functions, function_names=function_names, maximal_potence=2,
                        degree_output_numerator_specific=[1], enforce_function=False,
                        degree_input_denominator=2, degree_output_denominator=2, normalize_denom=True,
                        degree_output_denominator_specific=[1], device=device)

    n_params = model.get_number_parameters()
    print(f'Number parameters: {n_params}')
    evaluator = Evaluator(x, y, model=model, lambda_0=0, lambda_1=0.001, lambda_denom=0, n_params=n_params)
    model.prepare_input_monomials(x)

    t_0 = time.time()
    x0 = np.random.randn(n_params)
    ret = basinhopping(evaluator.loss_func, niter=5, x0=x0, minimizer_kwargs={'jac': evaluator.gradient})
    t_1 = time.time()
    print(f'Training time: {t_1 - t_0}')
    if not test_model:
        model.testing_mode()

    print(f'Coefficients: {ret.x}')
    print(
        f'Relative l2 distance: {np.linalg.norm(y - model.predict(torch.tensor(ret.x, device=device), x).cpu().detach().numpy(), ord=2) / np.linalg.norm(y, ord=2)}')
    # metric = R2Score().to(x.device)
    y_pred = model.predict(torch.tensor(ret.x, device=device), x)
    # metric.update(y_pred, y)
    # print(f'R squared: {metric(y_pred, y)}')
    print(f'Formula: {model.get_formula(torch.tensor(ret.x), decimals=10)}')

    print(f'### Now from utils ###')
    from utils import relative_l2_distance, r_squared
    print(f'Relative l2 distance: {relative_l2_distance(model, torch.tensor(ret.x), x, y)}')
    print(f'R squared: {r_squared(model, torch.tensor(ret.x), x, y)}')


def dev_training_jit(device):
    np.random.seed(12345)
    print(f'Using {device}')
    a = 5 * np.random.randn(1)
    x = np.arange(1, 10, 0.05)
    x = x.reshape(len(x), 1)

    test_model = False

    def func(a, x, module):
        # Good approximations with both, however, never yields a simple formula
        return module.sin((a[0] * x + 1) / (0.1 * x + 2))
        # return module.sin((a[0] * x + 1) / (0.1 * x))  # Works for lambda_1=0.1, lambda_denom=0 (not = 1)
        # return module.sin((a[0] * x))  # Works for lambda_1=0.1, lambda_denom=1
        # return a[0] * x
        # return 0.2 * module.sin(a[0] * x) / x
        # return 0.5 * x / (x + 1)  # Only works when using normalize_denom=True and lambda_1>=0.3

    y = func(a, x, np).squeeze(-1)
    print(f'Target formula: {func(a, sympy.Symbol("x"), sympy)}')

    functions = [torch.sin]
    function_names = [sympy.sin]
    model = ParFamTorch(n_input=1, degree_input_numerator=2, degree_output_numerator=2, width=1,
                        functions=functions, function_names=function_names,
                        degree_output_numerator_specific=[1], enforce_function=False,
                        degree_input_denominator=2, degree_output_denominator=2, normalize_denom=True,
                        degree_output_denominator_specific=[1], device=device)

    n_params = model.get_number_parameters()
    print(f'Number parameters: {n_params}')
    model.prepare_input_monomials(x)
    scripted_predict = torch.jit.script(model.predict, example_inputs=[torch.randn(n_params), x, False])

    evaluator = Evaluator(x, y, model=model, lambda_0=0, lambda_1=0.001, lambda_denom=0, n_params=n_params,
                          scripted_predict=scripted_predict)

    lw = [-10] * n_params
    up = [10] * n_params
    t_0 = time.time()
    x0 = np.random.randn(n_params)
    ret = basinhopping(evaluator.loss_func, niter=50, x0=x0, minimizer_kwargs={'jac': evaluator.gradient})
    t_1 = time.time()
    print(f'Training time: {t_1 - t_0}')
    if not test_model:
        model.testing_mode()

    print(f'Coefficients: {ret.x}')
    print(
        f'Relative l2 distance: {np.linalg.norm(y - model.predict(torch.tensor(ret.x, device=device), x).cpu().detach().numpy(), ord=2) / np.linalg.norm(y, ord=2)}')
    print(f'Formula: {model.get_formula(torch.tensor(ret.x), decimals=10)}')


def dev_training_pruning(device):
    print(f'Using {device}')
    a = 5 * np.random.randn(1)
    x = np.arange(1, 10, 0.05)
    x = x.reshape(len(x), 1)

    test_model = False

    def func(a, x, module):
        # Good approximations with both, however, never yields a simple formula
        return module.sin((a[0] * x + 1) / (0.5 * x + 2))
        # return module.sin((a[0] * x + 1) / (0.1 * x))  # Works for lambda_1=0.1, lambda_denom=0 (not = 1)
        # return module.sin((a[0] * x))  # Works for lambda_1=0.1, lambda_denom=1
        # return a[0] * x
        # return 0.2 * module.sin(a[0] * x) / x
        # return 0.5 * x / (x + 1)  # Only works when using normalize_denom=True and lambda_1>=0.3

    y = func(a, x, np).squeeze(-1)
    print(f'Target formula: {func(a, sympy.Symbol("x"), sympy)}')

    functions = [torch.sin]
    function_names = [sympy.sin]
    model = ParFamTorch(n_input=1, degree_input_numerator=2, degree_output_numerator=4, width=1,
                        functions=functions, function_names=function_names,
                        degree_output_numerator_specific=[1], enforce_function=False,
                        degree_input_denominator=2, degree_output_denominator=3, normalize_denom=True,
                        degree_output_denominator_specific=[1])

    n_params = model.get_number_parameters()
    print(f'Number parameters: {n_params}')
    evaluator = Evaluator(x, y, model=model, lambda_0=0, lambda_1=0.001, lambda_denom=0, n_params=n_params)
    model.prepare_input_monomials(x)

    lw = [-10] * n_params
    up = [10] * n_params
    t_0 = time.time()
    x0 = np.random.randn(n_params)
    ret = basinhopping(evaluator.loss_func, niter=50, x0=x0, minimizer_kwargs={'jac': evaluator.gradient})
    t_1 = time.time()
    print(f'Training time: {t_1 - t_0}')
    if not test_model:
        model.testing_mode()

    coefficients = model.get_normalized_coefficients(ret.x)
    print(f'Coefficients (normalized): {coefficients}')
    print(
        f'Relative l2 distance: {np.linalg.norm(y - model.predict(torch.tensor(coefficients, device=device), x).cpu().detach().numpy(), ord=2) / np.linalg.norm(y, ord=2)}')
    print(f'Formula: {model.get_formula(coefficients, decimals=10)}')

    mask = np.abs(coefficients) > 0.01
    evaluator = Evaluator(x, y, model=model, lambda_0=0, lambda_1=0.001, lambda_denom=0, n_params=n_params, mask=mask)
    active_params = sum(mask)
    x0 = np.random.randn(active_params)
    print(f'Active parameters {active_params}')
    t_0 = time.time()
    ret = basinhopping(evaluator.loss_func, niter=50, x0=x0, minimizer_kwargs={'jac': evaluator.gradient})
    t_1 = time.time()
    print(f'Training time (fine tuning): {t_1 - t_0}')
    if not test_model:
        model.testing_mode()

    coefficients = torch.zeros(n_params, device=device, dtype=torch.double)
    coefficients[mask] = torch.tensor(ret.x, device=device, dtype=torch.double)
    coefficients = model.get_normalized_coefficients(coefficients)
    print(f'Coefficients (after pruning & finetuning) (normalized): {coefficients}')
    print(f'Relative l2 distance (after finetuning): '
          f'{np.linalg.norm(y - model.predict(torch.tensor(coefficients, device=device), x).cpu().detach().numpy(), ord=2) / np.linalg.norm(y, ord=2)}')
    print(f'Formula (after finetuning): {model.get_formula(coefficients, decimals=3)}')


def dev_training_mask(device):
    np.random.seed(12345)
    torch.manual_seed(12345)

    a = 5 * torch.randn(1)
    x = torch.arange(-10, 10, 0.05)
    x = x.reshape((len(x), 1))

    test_model = False

    def func(a, x, module):
        return a * x

    y = func(a, x, torch).squeeze(-1)
    print(f'a: {a}')

    functions = [torch.sin, torch.cos]
    function_names = [sympy.sin, sympy.cos]
    model = ParFamTorch(n_input=1, degree_input_numerator=2, degree_output_numerator=2, width=1,
                        functions=functions, function_names=function_names, maximal_potence=2,
                        degree_output_numerator_specific=[1, 1], enforce_function=False,
                        degree_input_denominator=0, degree_output_denominator=0, normalize_denom=True,
                        degree_output_denominator_specific=[1], device=device)

    n_params = model.get_number_parameters()

    mask = [i % 2 == 0 for i in range(n_params)]
    # mask = None


    evaluator = Evaluator(x, y, model=model, lambda_0=0, lambda_1=0.001, lambda_denom=0, n_params=n_params, mask=mask)
    model.prepare_input_monomials(x)

    if mask is None:
        active_params = model.get_number_parameters()
    else:
        active_params = sum(mask)

    print(f'Number parameters: {active_params}')
    t_0 = time.time()
    x0 = np.random.randn(active_params)
    ret = basinhopping(evaluator.loss_func, niter=5, x0=x0, minimizer_kwargs={'jac': evaluator.gradient})
    t_1 = time.time()
    print(f'Training time: {t_1 - t_0}')
    if not test_model:
        model.testing_mode()

    coefficients = torch.zeros(n_params, device=device, dtype=torch.double)
    coefficients[mask] = torch.tensor(ret.x, device=device, dtype=torch.double)
    if mask is not None:
        symbolic_a = [symbols(f'a{i}') if mask[i] else 0 for i in range(n_params)]
    else:
        symbolic_a = [symbols(f'a{i}') for i in range(n_params)]
    print(f'Coefficients: {coefficients}')
    print(
        f'Training RMSE: {torch.norm(torch.tensor(y, device=device, dtype=torch.double) - model.predict(coefficients, x), p=2)}')
    symbolic_expr = model.get_formula(symbolic_a, verbose=False)
    print(f'Symbolic expression: {symbolic_expr}')
    print(f'Formula: {model.get_formula(coefficients)}')

    model.predict(coefficients, x)


def dev_data_generation(ensure_uniqueness):
    torch.seed(12345)
    for _ in range(500):
        if _ == 15:
            pass
        x = torch.arange(-10, 10, 0.1)
        x = x.reshape(len(x), 1)
        functions = [torch.sin, torch.cos]
        function_names = [sympy.sin, sympy.cos]
        target_model = ParFamTorch(n_input=1, degree_input_numerator=1, degree_output_numerator=2, width=1,
                                   functions=functions, function_names=function_names)

        n_params = target_model.get_number_parameters()
        a = 5 * torch.random.randn(n_params)
        if ensure_uniqueness:
            switch = target_model.get_random_coefficients_unique(n_functions_max=3)
        else:
            switch = target_model.get_random_coefficients(n_functions_max=3)
        a = switch * a

        y = target_model.predict(a, x)
        # print(f'a: {a}')

        x_sym = symbols('x')

        symbolic_a = [symbols(f'a{i}') for i in range(n_params)]

        generated_expr = target_model.get_formula(a, verbose=False)
        symbolic_expr = target_model.get_formula(symbolic_a, verbose=False)
        generated_expr = round_expr(generated_expr, 3)
        print(f'Generated expression: {generated_expr}')
        # print(f'Parametric formula: {symbolic_expr}')

        get_active_coefficients(a, target_model, x)


def dev_data_generation_comparison():
    seeds = torch.arange(50)
    for _ in range(50):
        torch.random.seed(seeds[_])

        x = torch.arange(-10, 10, 0.1)
        x = x.reshape(len(x), 1)
        functions = [torch.sin, torch.cos]
        function_names = [sympy.sin, sympy.cos]
        target_model = ParFamTorch(n_input=1, degree_input_numerator=1, degree_output_numerator=2, width=1,
                                   functions=functions, function_names=function_names)

        n_params = target_model.get_number_parameters()
        a = 5 * torch.random.randn(n_params)
        switch = target_model.get_random_coefficients(n_functions_max=3)
        a = switch * a

        y = target_model.predict(a, x)
        # print(f'a: {a}')

        x_sym = symbols('x')

        symbolic_a = [symbols(f'a{i}') for i in range(n_params)]

        generated_expr = target_model.get_formula(a, verbose=False)
        symbolic_expr = target_model.get_formula(symbolic_a, verbose=False)
        generated_expr = round_expr(generated_expr, 3)
        print(f'Generated expression (normal): {generated_expr}')
        # print(f'Parametric formula: {symbolic_expr}')

        ################

        torch.random.seed(seeds[_])

        x = torch.arange(-10, 10, 0.1)
        x = x.reshape(len(x), 1)
        functions = [torch.sin, torch.cos]
        function_names = [sympy.sin, sympy.cos]
        target_model = ParFamTorch(n_input=1, degree_input_numerator=1, degree_output_numerator=2, width=1,
                                   functions=functions, function_names=function_names)

        n_params = target_model.get_number_parameters()
        a = 5 * torch.random.randn(n_params)
        switch = target_model.get_random_coefficients_unique(n_functions_max=3)
        a = switch * a

        y = target_model.predict(a, x)
        # print(f'a: {a}')

        x_sym = symbols('x')

        symbolic_a = [symbols(f'a{i}') for i in range(n_params)]

        generated_expr = target_model.get_formula(a, verbose=False)
        symbolic_expr = target_model.get_formula(symbolic_a, verbose=False)
        generated_expr = round_expr(generated_expr, 3)
        print(f'Generated expression (unique): {generated_expr}')
        # print(f'Parametric formula: {symbolic_expr}')


def dev_batch_prediction():
    x = torch.random.randn(100, 2)

    functions = [torch.sin, torch.cos]
    function_names = [sympy.sin, sympy.cos]
    model = ParFamTorch(n_input=1, degree_input_numerator=1, degree_output_numerator=2, width=1,
                        functions=functions, function_names=function_names)

    n_params = model.get_number_parameters()
    model.prepare_input_monomials(x)
    coefficients = torch.random.randn(n_params, 3)

    y_batch = model.predict_batch(coefficients, x)
    y = model.predict(coefficients[:, 1], x)

    print(f'Difference in prediction: {torch.linalg.norm(y - y_batch[:, 1], ord=2)}')

    plt.plot(y, label='Single prediction')
    plt.plot(y_batch[:, 1], label='Batch prediction')
    plt.legend()
    plt.show()


def dev_training_lbfgs():
    # torch.manual_seed(12345)
    a = 5 * torch.randn(1)
    x = torch.arange(1, 10, 0.1, dtype=torch.double)
    x = x.reshape(len(x), 1)

    test_model = False

    def func(a, x, module):
        # Good approximations with both, however, never yields a simple formula
        return module.sin((a[0] * x + 1) / (0.1 * x + 2))
        # return module.sin((a[0] * x + 1) / (0.1 * x))  # Works for lambda_1=0.1, lambda_denom=0 (not = 1)
        # return module.sin((a[0] * x))  # Works for lambda_1=0.1, lambda_denom=1
        # return a[0] * x
        # return 0.2 * module.sin(a[0] * x) / x
        # return 0.5 * x / (x + 1)  # Only works when using normalize_denom=True and lambda_1>=0.3

    y = func(a, x, np).squeeze(-1)
    print(f'Target formula: {func(a, sympy.Symbol("x"), sympy)}')

    functions = [torch.sin]
    function_names = [sympy.sin]
    model = ParFamTorch(n_input=1, degree_input_numerator=1, degree_output_numerator=2, width=1,
                        functions=functions, function_names=function_names,
                        degree_output_numerator_specific=[1],
                        degree_input_denominator=1, degree_output_denominator=0, normalize_denom=True)

    n_params = model.get_number_parameters()
    evaluator = Evaluator(x, y, model=model, lambda_0=0, lambda_1=0.00000001, lambda_denom=0, n_params=n_params)
    model.prepare_input_monomials(x)

    def gradient_part(const, linear):
        return -2 / len(y) * (y - (x.squeeze() * linear + const))

    def gradient_const(const, linear):
        return sum(gradient_part(const, linear))

    def gradient_linear(const, linear):
        return gradient_part(const, linear) @ x

    def gradient(const, linear):
        return gradient_const(const, linear), gradient_linear(const, linear)

    # lw = [-10] * n_params
    # up = [10] * n_params
    best_coefficients = None
    best_relative_l2_distance = torch.inf
    t_0 = time.time()
    for i in range(100):
        coefficients = torch.randn(n_params, dtype=torch.double) * 5
        coefficients, relative_l2_distance = evaluator.fit_lbfgs(coefficients, verbose=True)
        if relative_l2_distance < best_relative_l2_distance:
            best_relative_l2_distance = relative_l2_distance
            best_coefficients = coefficients
            if relative_l2_distance < 10 ** (-5):
                break

    print(f'#### END OF TRAINING ####')
    t_1 = time.time()
    print(f'Training time: {t_1 - t_0}; Necessary iterations: {i + 1}')
    if not test_model:
        model.testing_mode()
    best_coefficients = best_coefficients
    print(f'Best coefficients: {best_coefficients}')
    print(f'Best relative l2 distance: {best_relative_l2_distance}')
    print(f'Best formula: {model.get_formula(best_coefficients, decimals=3)}')
    print(f'Target formula: {func(a, sympy.Symbol("x"), sympy)}')

    model.predict(best_coefficients, x)


if __name__ == '__main__':
    # np.random.seed(12345)
    dev_training_lbfgs()
    # dev_training('cpu')
    # dev_training_jit()
    # dev_training_pruning()
    # dev_data_generation(ensure_uniqueness=True)
    # dev_data_generation_comparison()
    # dev_batch_prediction()
    #dev_training_mask('cpu')
