import logging
import argparse
import pathlib
import configparser
import sys
import ast
import shutil

from utils import *
from train import model_parameter_search, function_dict, function_name_dict
from pmlb import fetch_data
import datetime

# import tracemalloc

def init():
    sys.path.append(os.path.dirname(os.getcwd()))

    # Initialize parser
    parser = argparse.ArgumentParser(description="Debugging")
    parser.add_argument('-c', '--config', help='Name of the config file:', default='hillary.ini')
    args = parser.parse_args()

    cwd = os.getcwd()  # Get current working directory
    config_name = args.config
    config_path = os.path.join(cwd, 'config_files', config_name)
    # Get configuration for *.ini file
    config = configparser.ConfigParser()  # Create a ConfigParser object
    config.read(config_path)  # Read the configuration file
    experiment_name = config_name[:-4]

    # Set path to pmlb dataset
    print(f'CWD: {cwd}')
    print(f'#### Config path: {config_path} ####')
    path_pmlb = config['META']['path_pmlb']
    path_pmlb_datasets = os.path.join(path_pmlb, 'datasets')

    df_summary = pd.read_csv(os.path.join(path_pmlb, 'pmlb', "all_summary_stats.tsv"), sep='\t')
    regression_dataset_names = df_summary.query('task=="regression"')['dataset'].tolist()

    datetime_ = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if experiment_name:
        underscore = '_'
    else:
        underscore = ''
    directory = os.path.join('results', datetime_ + underscore + experiment_name)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    print(f'Creating directory {directory}')
    # Initialise logging
    logging.basicConfig(filename=os.path.join(directory, 'experiment.log'), level=logging.INFO)
    logging.info('Starting the logger.')
    logging.debug(f'Directory: {directory}')
    logging.debug(f'File: {__file__}')

    shutil.copyfile(config_path, os.path.join(directory, config_name))
    return config, directory, regression_dataset_names, path_pmlb_datasets


def init_results_dict(model_parameters_dict, training_parameters_dict):
    results_dict = {**{key: [] for key in model_parameters_dict.keys()},
                    **{key: [] for key in training_parameters_dict[0].keys()}}
    results_dict['dataset'] = []
    results_dict['target_formula'] = []
    results_dict['estimated_formula'] = []
    results_dict['relative_l2_train'] = []
    results_dict['relative_l2_val'] = []
    results_dict['relative_l2_test'] = []
    results_dict['r_squared_val'] = []
    results_dict['r_squared_test'] = []
    results_dict['success'] = []
    results_dict['training_time'] = []
    results_dict['n_active_coefficients'] = []
    results_dict['n_active_coefficients_reduced'] = []
    results_dict['relative_l2_distance_test_reduced'] = []
    results_dict['r_squared_test_reduced'] = []
    results_dict['r_squared_val_reduced'] = []
    results_dict['best_formula_reduced'] = []
    results_dict['n_evaluations'] = []
    return results_dict


# Performs experiments for all datasets in dataset_names and hyperparameters in hyperparameters_dict:
#### n_experiment: number of how often the experiment is repeated
#### accuracy: determines wether the experiment was a success or not
def perform_experiments(n_experiments, dataset_names, model_parameters_dict, training_parameters_dict,
                        config, directory, path_pmlb_datasets):
    results_dict = init_results_dict(model_parameters_dict, training_parameters_dict)
    if config['META']['random_equations'] == 'True':
        x_random = torch.arange(-10, 10, 0.1)
        x_random = x_random.reshape((len(x_random), 1))
        functions = []
        function_names = []
        for function_name_str in ast.literal_eval(config['MODELPARAMETERSFIX']['function_names']):
            functions.append(function_dict[function_name_str])
            function_names.append(function_name_dict[function_name_str])
        ensure_uniqueness = False
        degree_input_polynomials = int(config['MODELPARAMETERSFIX']['degree_input_polynomials'])
        degree_output_polynomials = int(config['MODELPARAMETERSFIX']['degree_output_polynomials'])
        width = int(config['MODELPARAMETERSFIX']['width'])
        device = 'cpu'
        normalization = False,
        degree_output_polynomials_specific = ast.literal_eval(
            config['MODELPARAMETERSFIX']['degree_output_polynomials_specific'])
        enforce_function = ast.literal_eval(config['MODELPARAMETERSFIX']['enforce_function']),
        n_functions_max = 2
        y_random, a_random, params_random = create_dataset(len(dataset_names), ensure_uniqueness=ensure_uniqueness,
                                                           degree_input_polynomials=degree_input_polynomials,
                                                           degree_output_polynomials=degree_output_polynomials,
                                                           width=width, functions=functions,
                                                           function_names=function_names, device=device,
                                                           normalization=normalization,
                                                           degree_output_polynomials_specific=degree_output_polynomials_specific,
                                                           enforce_function=enforce_function,
                                                           n_functions_max=n_functions_max)
    for _ in range(n_experiments):
        for i, dataset_name in enumerate(dataset_names):
            if config['META']['random_equations'] == 'True':
                x = x_random
                y = y_random[i]
                model = ParFamTorch(n_input=1, degree_input_numerator=degree_input_polynomials,
                                    degree_output_numerator=degree_output_polynomials, width=width,
                                    functions=functions, function_names=function_names,
                                    enforce_function=enforce_function,
                                    degree_output_numerator_specific=degree_output_polynomials_specific)
                target_formula = model.get_formula(params_random[i], verbose=False)
            else:
                x, y = fetch_data(dataset_name, return_X_y=True, local_cache_dir=path_pmlb_datasets)
                target_formula = get_formula(dataset_name)
            if config['META']['model_parameter_search'] == 'Perfect':
                model_parameters_perfect_config = configparser.ConfigParser()  # Create a ConfigParser object
                model_parameters_perfect_config.read(
                    os.path.join(os.getcwd(), 'config_files', 'model_parameters_perfect.ini'))
                model_parameters_perfect = dict(model_parameters_perfect_config.items(dataset_name))
                model_parameters_perfect = {key: ast.literal_eval(model_parameters_perfect[key]) for key in
                                            model_parameters_perfect.keys()}
                model_parameters_perfect['functions'] = model_parameters_perfect['function_names']
            else:
                model_parameters_perfect = None
            # x,y,target_formula = experiment_data(1000,"test")
            logging.info(f'##########{i + 1} out of {len(dataset_names)} datasets: {dataset_name}##########')
            print(f'##########{i + 1} out of {len(dataset_names)} datasets: {dataset_name}##########')
            logging.info(f"Feature shape: {x.shape}, target shape: {y.shape}")
            # if x.shape[1] > 30:
            #     logging.info(f"Number of features {x.shape[1]} is bigger than 30, skip to next dataset")
            #     continue
            for i, training_parameters in enumerate(training_parameters_dict):
                logging.info(f"###{i + 1} out of {len(training_parameters_dict)} Hyperparameter combinations ###")
                print(f'Training parameters: {training_parameters}')
                logging.info(f'Training parameters: {training_parameters}')
                # Train model
                accuracy = training_parameters["accuracy"] + 1.5 * training_parameters["target_noise"]
                relative_l2_distance_train, relative_l2_distance_val, r_squared_val, estimated_formula, training_time, \
                    n_active_coefficients, relative_l2_distance_test, r_squared_test, n_active_coefficients_reduced, \
                    relative_l2_distance_test_reduced, r_squared_test_reduced, best_formula_reduced, \
                    r_squared_val_reduced, n_evaluations = model_parameter_search(x, y, target_formula,
                                                                                  model_parameters_dict,
                                                                                  training_parameters, accuracy,
                                                                                  config=config,
                                                                                  model_parameters_perfect=model_parameters_perfect)

                # Save results
                logging.info(f"Target formula: {target_formula}")
                logging.info(f"Estimated formula: {estimated_formula}, Training time: {training_time}")
                logging.info(
                    f"Relative L2 distance train: {relative_l2_distance_train}, validation: {relative_l2_distance_val}")
                logging.info(f"R squared validation: {r_squared_val}")
                logging.info(f"Number of active coefficients: {n_active_coefficients}")
                logging.info(f"Number of evaluations in total: {n_evaluations}")
                success = relative_l2_distance_val < accuracy
                append_results_dict_model_training_parameters(results_dict, training_parameters, model_parameters_dict,
                                                              target_formula, estimated_formula,
                                                              relative_l2_distance_train, relative_l2_distance_val,
                                                              r_squared_val, success, training_time, dataset_name,
                                                              n_active_coefficients, relative_l2_distance_test,
                                                              r_squared_test, n_active_coefficients_reduced,
                                                              relative_l2_distance_test_reduced, r_squared_test_reduced,
                                                              best_formula_reduced, r_squared_val_reduced,
                                                              n_evaluations)
                results_df = pd.DataFrame(results_dict)
                results_df.to_csv(os.path.join(directory, 'results.csv'))
                # What is saved in summary results?
                lst = list(training_parameters_dict[0].keys())
                # Function to convert lists to tuples in results_df: otherwise hash error -> list
                convert_list_to_tuple = lambda x: tuple(x) if isinstance(x, list) else x
                results_df = results_df.applymap(convert_list_to_tuple)
                # Save summary algorithms:
                column_names = ["relative_l2_train", "relative_l2_val", "success", "training_time"]
                summary_algorithms = results_df.fillna(value='None').groupby(lst, as_index=False)[column_names].mean()
                summary_algorithms.to_csv(os.path.join(directory, 'summary_algorithms.csv'))
                # Save summary data
                summary_data = results_df[
                    ['dataset', "relative_l2_train", "relative_l2_val", 'success', 'training_time']].groupby(
                    by=['dataset'], as_index=False).mean()
                summary_data.to_csv(os.path.join(directory, 'summary_data.csv'))
        # if _ > 5:
        #     break


if __name__ == '__main__':
    config, directory, regression_dataset_names, path_pmlb_datasets = init()
    # Get model and training parameters from config file
    model_parameters_dict = dict(config.items("MODELPARAMETERS"))
    model_parameters_dict = {key: ast.literal_eval(model_parameters_dict[key]) for key in
                             model_parameters_dict.keys()}

    training_parameters_dict = dict(config.items("TRAININGPARAMETERS"))
    training_parameters_dict = {key: ast.literal_eval(training_parameters_dict[key]) for key in
                                training_parameters_dict.keys()}
    training_parameters_dict = get_hyperparameters_combination(training_parameters_dict)

    n_experiments = 1
    random_equations = config['META']['random_equations']
    if random_equations == 'True':
        dataset_names = range(int(config['META']['n_equations']))
    else:
        start_experiment = int(config['META']['start_experiment'])
        end_experiment = int(config['META']['end_experiment'])
        if eval(config['META']['comparison_subset']):
            np.random.seed(123456)
            feynman_indices = [i for i in range(120, 239) if not i in [179, 188, 193, 220]]
            dataset_names = [regression_dataset_names[i] for i in
                             np.sort(np.random.choice(feynman_indices, 15, replace=False))]
            logging.info(f'Comparison subset: {dataset_names}')
        else:
            dataset_names = regression_dataset_names[start_experiment:end_experiment]  # 120 feynman dataset begins

    perform_experiments(n_experiments, dataset_names, model_parameters_dict, training_parameters_dict,
                        config, directory, path_pmlb_datasets)

    logging.info(f'Finished the experiment on {datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
