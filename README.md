## Prerequisites

This repository contains the code accompanying the paper "ParFam - (Neural Guided) Symbolic Regression using Global Continuous
Optimization" by Philipp Scholl, Katharina Bieker, Hillary Hauger, and Gitta Kutyniok, accepted at ICLR 2025 [1]. It proposes a new symbolic regression method leveraging the structure of common physical laws.

## Implementation

As dependencies, the repository needs the packages specified in ``requirements.txt``. To install the repository as a package you can either use the wheel provided in ``dist/`` or run the command

````
pip install parfam
````

## Usage

For applications, it is usually enough to use the wrapper `ParFamWrapper` in `parfamwrapper.py` by simply calling (if the package is installed)

````
from parfam import ParFamWrapper
parfam = ParFamWrapper(iterate=True, functions=functions, function_names=function_names)
parfam.fit(x, y, time_limit=100)
````
This will use the default/small configuration specifified in this [config file](config_files/wrapper/small.md). To pick a [bigger one](config_files/wrapper/big.md), set:

````
parfam = ParFamWrapper(config_name='big', iterate=True, functions=functions, function_names=function_names)
parfam.fit(x, y)
````
Note that these are the same settings used in the paper for the Feynman experiments. All extra parameters that are handed over to ParFamWrapper (such as functions/function_names) or to parfam.fit (such as time_limit), overwrite the values specified in the config files. See also [example.ipynb](example.ipynb) for more possible hyperparameters.

The computed formula can then be assessed by running 
````
parfam.formula_reduced
````
and it can be used for predictions using
````
y_pred = parfam.predict(x).
````
A more thorough introduction is shown in `example.ipynb`. There, many of the necessary hyperparameters are described which might be necessary for indepth experiments.

## Structure

The `parfam_torch.py` module contains the main work of this repository: The `ParFamTorch` class, which defines the ParFam
algorithm, and the `Evaluator` class, which provides the framework to evaluate the learned coefficients for a given data 
set and obtain the loss function as specified in Equation (2) in "ParFam - Symbolic Regression using Global Continuous
Optimization":
$$L(\theta)=\frac{1}{N}\sum_{i=1}^N\left(y_i-f_\theta(x_i)\right)^2+ \lambda R(\theta)$$

The formula is learned by calling any optimizer on the loss function defined by the `Evaluator` class. An example using 
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
coefficients = torch.tensor(ret.x, device=device)
````

Running `parfam_torch.py` will start the computation of a simple test function, to see if everything has been installed 
correctly. There are many hyperparameters for both `ParFamTorch` and `Evaluator`, which are introduced in their 
respective constructors. To obtain the function given the coefficients run the command,

````
model.get_formula(ret.x, decimals=10)
````
which returns the formula as a `sympy` function. This can be used to predict new inputs, however, it is recommended to 
avoid `sympy` for evaluation and instead, keep everything in pytorch by running
````
model.predict(ret.x, x)
````

These base functionalities are all included in the function `dev_training`, which should run by starting the main 
function to test if the installation worked and can be used to gain familiarity with the framework. It tests the process
for arbitrary mathematical functions that can be directly specified in `dev_training`. 

## Plots

To analyze the results shown in [2] and redo the plots, run the cells in `results/srbench_groundtruth_results.ipynb`, which builds up on the code from [2].

## Reproduce experiments

To run the experiments from Section 3 on the SRBench ground-truth problems [2] you first have to download the Feynman 
and Strogatz data sets by downloading https://github.com/EpistasisLab/pmlb. After completing this use the command

````
python benchmark.py -c feynman.ini
````

The config file `feynman.ini` has to be stored in the folder `config_files` and contains parameters about 

1. the experiment itself (results path, dataset path, which dataset, etc.),
2. the model parameters and
3. the training parameters.

The most important thing that has to be customized in the setting is the variable `path_pmlb` which must be set to the
path of the pmlb directory on your machine. By specifying 
````
model_parameter_search = Complete
````
the base algorithm ParFam with the extensive model parameter search will be started. To switch to DL-ParFam, change this 
parameter to
````
model_parameter_search = pretrained
````
For the experiments on SRSD, it is necessary to download the data sets from https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_easy
and https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_medium and set 
````
dataset = srsd
````
in the config file and set the variable *path_srsd* to the path to the SRSD dataset one is interested in.

## Retrain DL-ParFam

To train your own pre-trained network for DL-ParFam, you have to run
````
python trainingOnSyntheticData/train.py -c train_model_parameters.ini
````
which will save the trained model in a subdirectory of trainingOnSyntheticData/results/.


The notebook `expressivity.ipynb` contains our calculation in Section 2.2.

## References

[1] Philipp Scholl, Katharina Bieker, Hillary Hauger, and Gitta Kutyniok. "ParFam--Symbolic Regression Based on Continuous Global Optimization." International Conference on Representation Learning (ICLR). 2025. [arXiv](https://arxiv.org/abs/2310.05537).

[2] La Cava, William, Patryk Orzechowski, Bogdan Burlacu, Fabrício Olivetti de França, Marco Virgolin, Ying Jin, Michael Kommenda, and Jason H. Moore. "Contemporary symbolic regression methods and their relative performance." Conference on Neural Information Processing Systems (NeurIPS). 2021. [arxiv](https://arxiv.org/abs/2107.14351).
