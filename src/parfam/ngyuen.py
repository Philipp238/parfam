# This file is provides a method to produce the Nguyen data sets
import torch
import numpy as np
import sympy


def nguyen(function_name, bigger_range):
    if function_name == '1':
        x = np.random.uniform(-1, 1, 20).reshape(-1, 1)
        func = lambda x, module: x ** 3 + x ** 2 + x

    if function_name == '2':
        x = np.random.uniform(-1, 1, 20).reshape(-1, 1)
        func = lambda x, module: x ** 4 + x ** 3 + x ** 2 + x

    if function_name == '3':
        x = np.random.uniform(-1, 1, 20).reshape(-1, 1)
        func = lambda x, module: x ** 5 + x ** 4 + x ** 3 + x ** 2 + x

    if function_name == '4':
        x = np.random.uniform(-1, 1, 20).reshape(-1, 1)
        func = lambda x, module: x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x

    if function_name == '5':
        if bigger_range:
            x = np.random.uniform(-3, 3, 100).reshape(-1, 1)
        else:
            x = np.random.uniform(-1, 1, 20).reshape(-1, 1)
        func = lambda x, module: module.sin(x ** 2) * module.cos(x) - 1

    if function_name == '6':
        if bigger_range:
            x = np.random.uniform(-5, 5, 100).reshape(-1, 1)
        else:
            x = np.random.uniform(-1, 1, 20).reshape(-1, 1)
        func = lambda x, module: module.sin(x) + module.sin(x + x ** 2)

    if function_name == '7':
        x = np.random.uniform(-0.9, 2, 100).reshape(-1, 1)  # previously we had (-0.7, 10, 100)
        func = lambda x, module: module.log(x + 1) + module.log(1 + x ** 2)

    if function_name == '8':
        x = np.random.uniform(0, 4, 20).reshape(-1, 1)
        func = lambda x, module: module.sqrt(x)

    if function_name == '9':
        if bigger_range:
            x = np.random.uniform(-3, 3, 200).reshape(100, 2)
        else:
            x = np.random.uniform(-1, 1, 200).reshape(100, 2)
        func = lambda x, module: module.sin(x[0]) + module.sin(x[1] ** 2)

    if function_name == '10':
        if bigger_range:
            x = np.random.uniform(-5, 5, 200).reshape(100, 2)
        else:
            x = np.random.uniform(-1, 1, 200).reshape(100, 2)
        func = lambda x, module: 2 * module.sin(x[0]) * module.cos(x[1])

    if function_name == '11':
        x = np.random.uniform(0, 2, 200).reshape(100, 2)
        func = lambda x, module: x[0] ** x[1]

    if function_name == '12':
        x = np.random.uniform(-1, 1, 200).reshape(100, 2)
        func = lambda x, module: x[0] ** 4 - x[0] ** 3 + 1 / 2 * x[1] ** 2 - x[1]

    # Now with constants
    if function_name == '13':
        x = np.random.uniform(-1, 1, 20).reshape(-1, 1)
        func = lambda x, module: 3.39 * x ** 3 + 2.12 * x ** 2 + 1.78 * x

    if function_name == '14':
        if bigger_range:
            x = np.random.uniform(-3, 3, 100).reshape(-1, 1)
        else:
            x = np.random.uniform(-1, 1, 100).reshape(-1, 1)
        func = lambda x, module: module.sin(x ** 2) * module.cos(x) - 0.75

    if function_name == '15':
        if bigger_range:
            x = np.random.uniform(-1.2, 5, 100).reshape(-1, 1)
        else:
            x = np.random.uniform(-1.2, 1, 100).reshape(-1, 1)
        func = lambda x, module: module.log(x + 1.4) + module.log(1.3 + x ** 2)

    if function_name == '16':
        x = np.random.uniform(0, 4, 20).reshape(-1, 1)
        func = lambda x, module: module.sqrt(1.23 * x)

    if function_name == '17':
        if bigger_range:
            x = np.random.uniform(-3, 3, 200).reshape(100, 2)
        else:
            x = np.random.uniform(-1, 1, 200).reshape(100, 2)
        func = lambda x, module: module.sin(1.5 * x[0]) + module.sin(0.5 * x[1] ** 2)

    if x.shape[1] == 1:
        y = func(x, np).squeeze(1)
        x_sym = sympy.symbols('x')
    else:
        y = func(x.T, np)
        x_sym = []
        for i in range(x.shape[1]):
            x_sym.append(sympy.symbols(f'x{i}'))
    target_expr = func(x_sym, sympy)

    return torch.tensor(x), torch.tensor(y), target_expr
