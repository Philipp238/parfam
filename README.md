## Prerequisites

This repository contains the code accompanying the paper "ParFam - Symbolic Regression using Global Continuous
Optimization", which investigates a new symbolic regression method leveraging the structure of common physical laws.

The code is implemented in Python 3 and requires the packages specified in ``requirements.txt``.

## Structure

The `parfam_torch.py` module contains the main work of this repository: The `ParFamTorch` class which defines the ParFam
algorithm and the `Evaluator` class which provides the framework to evaluate the learned coefficients for a given data 
set and obtain the loss function as specified in Equation (2) in "ParFam - Symbolic Regression using Global Continuous
Optimization":
$$L(\theta)=\frac{1}{N}\sum_{i=1}^N\left(y_i-f_\theta(x_i)\right)^2+ \lambda R(\theta)$$

The formula is learned by calling any optimizer on the loss function defiend by the `Evaluator` class. An example using 
basinhopping:
````
model = ParFamTorch(n_input=1, degree_input_numerator=2, degree_output_numerator=2, width=1,
                        functions=[torch.sin], function_names=[sympy.sin], maximal_potence=2,
                        degree_output_numerator_specific=[1],
                        degree_input_denominator=2, degree_output_denominator=2,
                        degree_output_denominator_specific=[1], device=device)
n_params = model.get_number_parameters()
evaluator = Evaluator(x, y, model=model, lambda_0=0, lambda_1=0.001, lambda_denom=0, n_params=n_params)
model.prepare_input_monomials(x)
x0 = np.random.randn(n_params)
ret = basinhopping(evaluator.loss_func, niter=5, x0=x0, minimizer_kwargs={'jac': evaluator.gradient})
coefficients = torch.tensor(ret.x, device=device))
````

Running `parfam_torch.py` will start the computation of a simple test function, to see if everything has been installed 
correctly. There are many hyperparameters for both `ParFamTorch` and `Evaluator`, which are introduced in their 
respective constructors. To obtain the function given the coefficients run the command,

````
model.get_formula(ret.x, decimals=10)
````
which returns the formula as a `sympy` function. This can be used to predict new inputs, however, it is recommended to 
avoid `sympy` for evaluation and instead keep everything in pytorch by running
````
model.predict(ret.x, x)
````

These base functionalities are all included in the function `dev_training`, which should run by starting the main 
function to test if the installation worked and can be used to gain familiarity with the framework. It tests the process
for arbitrary mathematical functions which can be specified in `dev_training` directly. 

To run the experiments from Section 3.1 on the SRBench ground-truth problems [1] you first have to download the Feynman 
and Strogatz data sets by downloading https://github.com/EpistasisLab/pmlb. After completing this use the command

````
python benchmark.py -c feynman.ini
````

The config file `feynman.ini` has to be stored in the folder `config_files` and contains parameters about 

1. the experiment itself (results path, dataset path, which dataset, etc.),
2. the model parameters and
3. the training parameters.

The most important thing that has to be customized in the setting is the variable `path_pmlb` which must be set to the
path of the pmlb directory on your machine. To process these results see the notebook `Process the results.ipynb` and to
visualize them to obtain the figures as shown in the paper, follow the notebook 
`results/plots/srbench_groundtruth_results.ipynb`.

To run the experiments for the optimizer comparison set the variable `comparison_subset` in the config file to
`True` and choose the optimizer you want to test. 
All our results are saved in the `results` folder. 

<<<<<<< HEAD
The notebook 'expressivity.ipynb' contains our calculation in Section 2.2.
=======
The notebook `expressivity.ipynb` contains our calculation in Section 2.2.
>>>>>>> 04aa8ac7ba73da04bf24b78fa6c166abb557f423


## References

[1] La Cava, William, Patryk Orzechowski, Bogdan Burlacu, Fabrício Olivetti de França, Marco Virgolin, Ying Jin, 
Michael Kommenda, and Jason H. Moore. "Contemporary symbolic regression methods and their relative performance." 
arXiv preprint arXiv:2107.14351 (2021).
