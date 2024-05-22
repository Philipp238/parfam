from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gc

from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import matplotlib.pyplot as plt

import sympy
from time import time
import os
import sys
import datetime
import pathlib
import logging
import argparse
import configparser
import ast
import shutil
import copy

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import torch.distributed as dist

print(os.getcwd())
sys.path[0] = os.getcwd()
import utils

torch.autograd.set_detect_anomaly(False)

msg = 'Train for one specific method'

# initialize parser
parser = argparse.ArgumentParser(description=msg)
default_config = 'debug_model_parameters.ini'
# default_config = 'debug_mask.ini'
parser.add_argument('-c', '--config', help='Name of the config file:', default=default_config)

args = parser.parse_args()

config_name = args.config
config = configparser.ConfigParser()
config.read(os.path.join('trainingOnSyntheticData', 'config', config_name))
results_path = config['META']['results_path']
experiment_name = config['META']['experiment_name']

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Using {device}.')

function_dict = {'sqrt-abs': lambda x: torch.sqrt(torch.abs(x)),
                 'exp-lin': lambda x: torch.minimum(torch.exp(x), np.exp(10) + torch.abs(x)),
                 'sqrt': torch.sqrt,
                 'exp': torch.exp,
                 'cos': torch.cos, 'sin': torch.sin}
function_name_dict = {'sqrt-abs': lambda x: sympy.sqrt(sympy.Abs(x)), 'exp-lin': sympy.exp, 'cos': sympy.cos, 'sin': sympy.sin, 'exp': sympy.exp,
                      'sqrt': sympy.sqrt}

def ddp_setup(rank: int, world_size: int):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  torch.cuda.set_device(rank)
  dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

class CustomDataset:
    def __init__(self, directory):
        self.directory = os.path.join(directory, 'training_data')

    def __len__(self):
        with open(os.path.join(self.directory, f'info.txt'), "r", newline="") as file:
            length = int(file.read())
        return length

    def __getitem__(self, idx):
        input = torch.load(os.path.join(self.directory, f'input_{idx}.pt'))
        target = torch.load(os.path.join(self.directory, f'target_{idx}.pt'))
        return input, target

class CustomSchedule():
    def __init__(self, max_lr, warmup_steps=10000):
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

    def get_lr_lambda(self, step):
        # return the factor with which the lr given to the optimizier will be multiplied with
        step += 1 # starts with 0 otherwise
        lr_factor = self.warmup_steps**0.5 * min(step**(-0.5), step * self.warmup_steps ** (-1.5))
        return lr_factor

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def get_bce_weight(weight_active, a_target):
    weight = torch.zeros(*a_target.shape, dtype=torch.float, device=device)
    weight[a_target == 1] = weight_active
    weight[a_target == 0] = 1 / weight_active 
    return weight

def compute_bce(pred, target):
    return -(target*torch.log(pred) + (1-target)*torch.log((1-pred))).mean()

def compute_ce(logits, target, data_parameters, criterion):
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    ce = 0
    index = 0
    if not data_parameters['predict_only_function']:
        if data_parameters['degree_input_numerator']>0:
            ce += criterion(logits[:, index:index + data_parameters['degree_input_numerator']+1], target[:, index:index + data_parameters['degree_input_numerator']+1])
            index += 1+data_parameters['degree_input_numerator']
        if data_parameters['degree_input_denominator']>0:
            ce += criterion(logits[:, index:index + data_parameters['degree_input_denominator']+1], target[:, index:index + data_parameters['degree_input_denominator']+1])
            index += 1+data_parameters['degree_input_denominator']
        if data_parameters['degree_output_numerator']>0:
            ce += criterion(logits[:, index:index + data_parameters['degree_output_numerator']+1], target[:, index:index + data_parameters['degree_output_numerator']+1])
            index += 1+data_parameters['degree_output_numerator']
        if data_parameters['degree_output_denominator']>0:
            ce += criterion(logits[:, index:index + data_parameters['degree_output_denominator']+1], target[:, index:index + data_parameters['degree_output_denominator']+1])
            index += 1+data_parameters['degree_output_denominator']
    if len(data_parameters['function_names_str'])>0:
        index = -(1+len(data_parameters['function_names_str']))
        ce += criterion(logits[:, index:], target[:, index:])
    return ce.mean()

def train(net, optimizer, input, criterion, target, i, balanced_loss, weight_active, gradient_clipping, data_parameters, one_hot, objective):
    optimizer.zero_grad(set_to_none=True)
    
    sigmoid = torch.nn.Sigmoid()
    
    logits = net(input, return_logits=True)
    
    logits_norm = logits.detach().cpu().data.norm(2)
    
    
    if objective == 'mask' or not one_hot:
        prediction = sigmoid(logits)
        if isinstance(criterion, torch.nn.BCELoss):
            if ((0 > prediction) | (1 < prediction)).any():
                print(f'Training: Input value outside [0,1] observed.')
            if prediction.isnan().any():
                print(f'Training: Input value nan observed.')
            if balanced_loss == True:
                criterion.weight = get_bce_weight(weight_active, target)  
        loss = criterion(prediction, target)
    else:
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = compute_ce(logits, target, data_parameters, criterion)
        else:
            prediction = utils.classification_layer(logits, data_parameters, sigmoid, one_hot)
            loss = criterion(prediction, target)

    loss.backward(retain_graph=False)
    
    
    gradient_norm = 0
    for p in net.parameters():
        param_norm = p.grad.detach().data.norm(2)
        gradient_norm += param_norm.item() ** 2
    gradient_norm = gradient_norm ** 0.5
    
    # if gradient_norm > 2:
    #     logging.warning(f'Gradient norm: {gradient_norm}')
    #     if isinstance(criterion, torch.nn.CrossEntropyLoss):
    #         logging.warning(f'Logits       : {logits[0]}')
    #     else:
    #         logging.warning(f'Prediction   : {prediction[0]}')
    #     logging.warning(f'Target       : {target[0]}')
    
    # if gradient_norm < 0.00001:
    #     logging.warning(f'Gradient norm: {gradient_norm}')
    #     if isinstance(criterion, torch.nn.CrossEntropyLoss):
    #         logging.warning(f'Logits       : {logits[0]}')
    #     else:
    #         logging.warning(f'Prediction   : {prediction[0]}')
    #     logging.warning(f'Target       : {target[0]}')
            
    torch.nn.utils.clip_grad_norm_(net.parameters(), gradient_clipping)
    
    gradient_norm_test = 0
    for p in net.parameters():
        param_norm = p.grad.detach().data.norm(2)
        gradient_norm_test += param_norm.item() ** 2
    gradient_norm_test = gradient_norm_test ** 0.5
    
    
    assert gradient_norm_test < 1.5 * gradient_clipping
    
    # # For debugging purposes
    # old_params = []
    # for p in net.parameters():    
    #     old_params.append(copy.deepcopy(p).flatten())
    # old_params = torch.cat(old_params, dim=0)
    
    optimizer.step()
    
    # # For debugging purposes
    # new_params = []
    # for p in net.parameters():
    #     new_params.append(copy.deepcopy(p).flatten())
    # new_params = torch.cat(new_params, dim=0)
    
    # distance_in_param_space = torch.norm(new_params - old_params, p=2)
    
    return loss.item(), gradient_norm, logits_norm.item()


def initialize_weights(model, init):
    for name, param in model.named_parameters():
        if 'weight' in name and param.data.dim() == 2:
            if init == 'xavier':
                nn.init.xavier_uniform(param)
            elif init == 'he':
                nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
            else:
                raise NotImplementedError(f'Please choose init as "default", "xavier" or "he". You chose: {init}.')

def mask_example_predictions(a_pred, a_target, num_examples, model_parameters, epoch, file_name, dimension):
    subset = np.random.choice(range(a_pred.shape[0]), num_examples, replace=False)
    a_pred = a_pred[subset]
    a_target = a_target[subset]
    a_pred = a_pred >= 0.2
    
    model = utils.setup_parfam_for_dlparfam(model_parameters, device, function_dict, function_name_dict, dimension)
    
    with open(file_name, 'a') as file:
        file.write(f'Epoch {epoch}:\n')
        for j in range(a_pred.shape[0]):
            symbolic_a_target = [int(a_target[j, i]) * sympy.symbols(f'a{i}') for i in range(a_target.shape[1])]
            target_formula = model.get_formula(symbolic_a_target, verbose=False)
            file.write(f'   Target formula: {target_formula}\n')
            target_formula = model.get_formula(symbolic_a_target, verbose=False)
            symbolic_a_pred = [int(a_pred[j, i]) * sympy.symbols(f'a{i}') for i in range(a_pred.shape[1])]
            pred_formula = model.get_formula(symbolic_a_pred, verbose=False)
            file.write(f'   Predicted formula: {pred_formula}\n')    

def model_parameter_example_predictions(prediction, target, num_examples, epoch, file_name):    
    subset = np.random.choice(range(target.shape[0]), num_examples, replace=False)
    prediction = prediction[subset]
    target = target[subset]
    
    with open(file_name, 'a') as file:
        file.write(f'Epoch {epoch}:\n')
        for j in range(num_examples):
            file.write(f'   Target model encoding: {target[j, :]}\n')
            file.write(f'   Predicted model encoding: {prediction[j, :]}\n')        

def report_example_predictions(prediction, target, num_examples, model_parameters, epoch, file_name, dimension, objective):
    if objective == 'mask':
        mask_example_predictions(prediction, target, num_examples, model_parameters, epoch, file_name, dimension)
    else:
        model_parameter_example_predictions(prediction, target, num_examples, epoch, file_name)


def get_criterion(training_parameters):
    if training_parameters['loss'] == 'MSE':
        criterion = nn.MSELoss()
    elif training_parameters['loss'] == 'MAE':
        criterion = nn.L1Loss()
    elif training_parameters['loss'] == 'BCE':
        criterion = nn.BCELoss()
    elif training_parameters['loss'] == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif training_parameters['loss'] == 'ASL':
        update = False
        if update:
            logging.info('Activated automatic update rule for ASL.')
        else:
            logging.info('Deactivated automatic update rule for ASL.')
        asl_obj = ASL(gamma_negative=training_parameters['gamma_neg'], p_target=training_parameters['p_target'],
                      update=True)
        criterion = asl_obj.asl
    else:
        raise NotImplementedError(
            f'Wrong loss specification: expected one out of "MSE", "BCE" and "ASL", '
            f'but received {training_parameters["loss"]}.')
    return criterion

def setup_nn(training_parameters, data_parameters, input_validation, target_validation, device, objective):
    if training_parameters['resume_training']:
        net = torch.load(training_parameters['resume_training'], map_location=device)
        return net
    
    if training_parameters['model'] == 'mlp':
        assert data_parameters['num_samples_min'] == data_parameters['num_samples_max']
        net = utils.MLP(input_validation.shape[1] * input_validation.shape[2], output_size=target_validation.shape[1], h=training_parameters['hidden_dim'], 
                        objective=objective, data_parameters=data_parameters, one_hot=training_parameters['one_hot']).to(device)
    elif training_parameters['model'] == 'rnn':
        input_size = input_validation.shape[-1]
        net = utils.RNNModel(input_size=input_size, hidden_size=training_parameters['hidden_dim'],
                       n_layers=training_parameters['num_layers'], output_size=target_validation.shape[1],
                       device=device).to(device)
    elif training_parameters['model'] == 'transformer':
        input_size = input_validation.shape[-1]
        net = utils.TransformerModel(in_features=input_size, embedding_size=training_parameters['hidden_dim'],
                               out_features=target_validation.shape[1], nhead=training_parameters['n_head'],
                               dim_feedforward=4 * training_parameters['hidden_dim'],
                               num_layers=training_parameters['num_layers'], dropout=training_parameters['dropout'],
                               activation="relu",
                               classifier_dropout=training_parameters['dropout'],
                               num_classifier_layer=training_parameters['num_layers_classifier']).to(device)
    elif training_parameters['model'] == 'set-transformer':
        input_size = input_validation.shape[-1]            
        net = utils.SetTransformer(dim_input=input_validation.shape[-1], num_outputs=1, dim_output=target_validation.shape[1], 
                                   dim_embedding=training_parameters['dim_embedding'], embedding=training_parameters['embedding'],
                                   dim_hidden_embedding=training_parameters['dim_hidden_embedding'],
                                   num_inds=training_parameters['num_inds'], one_hot=training_parameters['one_hot'],
                                   dim_hidden=training_parameters['hidden_dim'], objective=objective,
                                    num_heads=training_parameters['n_head'], ln=training_parameters['layer_normalization'], 
                                    sab_in_output=training_parameters['sab_in_output'],
                                    num_layers_enc=training_parameters['num_layers_enc'], num_layers_dec=training_parameters['num_layers_classifier'], 
                                    activation_fct='ReLU',data_parameters=data_parameters, dropout=training_parameters['dropout']).to(device)
    return net

def scheduler_step(scheduler, optimizer, epoch):
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    if before_lr != after_lr:
        print("[time: %d] Epoch %d: SGD lr %.8f -> %.8f" % (time(), epoch, before_lr, after_lr))


def trainer(gpu_id, input_training, target_training, target_validation, input_validation, training_parameters, 
            data_parameters, num_samples_min, lr_schedule, objective, directory, d_time, world_size=None):
    
    training_set_size = input_training.shape[0]
    batch_size = training_parameters['batch_size']
    
    if training_parameters['distributed_training']:
        ddp_setup(rank=gpu_id, world_size=world_size)
        print(f'GPU ID: {gpu_id}')
        # if gpu_id==0:
        if gpu_id>-1:
            logging.basicConfig(filename=os.path.join(directory, f'experiment_{gpu_id}.log'), level=logging.INFO)
            logging.info('Starting the logger in the training process.')
            print('Starting the logger in the training process.')
        # input_validation.to(f'cuda:{gpu_id}')
        # target_validation.to(f'cuda:{gpu_id}')
    
    
        # flag tensor for (early) stopping     
        flag_tensor = torch.zeros(1).to(f'cuda:{gpu_id}')
    
    if device == 'cpu':
        assert not training_parameters['data_loader_pin_memory']
    
    drop_last = input_training.shape[0] >= batch_size
    
    if not training_parameters['curriculum_learning']:
        logging.info(f'{utils.using("Before data loader: ")}')
        train_dataset = TensorDataset(input_training, target_training)
        if not training_parameters['distributed_training']:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=training_parameters['data_loader_num_workers'], 
                                        pin_memory=training_parameters['data_loader_pin_memory'], drop_last=drop_last)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=training_parameters['data_loader_num_workers'], 
                                        pin_memory=training_parameters['data_loader_pin_memory'], drop_last=drop_last,
                                        sampler=DistributedSampler(train_dataset))
        logging.info(f'{utils.using("After data loader: ")}')

    if training_parameters['balanced_loss'] == True:
        weight_active = (1 - target_training).sum() / target_training.sum() 
    else:
        weight_active = None

    criterion = get_criterion(training_parameters)
    
    logging.info(f'{utils.using("Before creating the model: ")}')
    net = setup_nn(training_parameters, data_parameters, input_validation, target_validation, device, objective)
    if training_parameters['distributed_training']:
        net = DDP(net, device_ids=[gpu_id])
    
    logging.info(f'{utils.using("After creating the model: ")}')

    if training_parameters['init'] != 'default':
        initialize_weights(net, training_parameters['init'])

    n_parameters = 0
    for parameter in net.parameters():
        n_parameters += parameter.nelement()
    logging.info(f'Number parameters: {n_parameters}')

    logging.info(f'Memory allocated: {torch.cuda.memory_reserved(device=device)}')
    
    # create your optimizer
    if training_parameters['optimizer'] == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=training_parameters['learning_rate'], betas=(0.9, 0.999))
    elif training_parameters['optimizer'] == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=training_parameters['learning_rate'])
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)
    report_every = 1
    early_stopper = EarlyStopper(patience=int(training_parameters['early_stopping'] / report_every), min_delta=0.0001)
    running_loss = 0
    grad_norm = 0
    logits_norm = 0
    
    training_loss_list = []
    validation_loss_list = []
    grad_norm_list = []
    logits_norm_list = []
    f1_list = []
    epochs = []
    
    plt_activations = training_parameters['plt_activations']
    
    assert (not plt_activations) or (training_parameters['model'] == 'set-transformer')
    
    if plt_activations:
        # a dict to store the activations
        freq_activ_rep = 50
        activation = {}
        def getActivation(name):
            # the hook signature
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        num_layers_enc = training_parameters['num_layers_enc']
        num_layers_classifier = training_parameters['num_layers_classifier']
        
        activation_list_batches = {i: []  for i in range(num_layers_enc + num_layers_classifier)}
    
    # train_opt = torch.compile(train, mode="reduce-overhead")
    best_loss = torch.inf
    
    sigmoid = torch.nn.Sigmoid()
    
    if lr_schedule == 'warm-up':
        custom_lr_schedule = CustomSchedule(max_lr=training_parameters['learning_rate'], warmup_steps=training_parameters['warmup_steps'])
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: custom_lr_schedule.get_lr_lambda(step))
    elif lr_schedule == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    logging.info(f'Training starts now.')
    logging.info(f'{utils.using("CPU Memory")}')


    for epoch in range(training_parameters['n_epochs']):
        
        if training_parameters['distributed_training']:
            dist.all_reduce(flag_tensor,op=dist.ReduceOp.SUM)
            if flag_tensor == 1:
                logging.info("Training stopped")
                break
            train_loader.sampler.set_epoch(epoch)
        
        if (not data_parameters['generate_new_data']) and len(data_parameters['dataset']) > 1:
            dataset_index = epoch % len(data_parameters['dataset'])
            
            data_path = config['META']['data_path']
            dataset = data_parameters['dataset'][dataset_index]
            
            input_training = torch.load(os.path.join(data_path, dataset, 'training_input.pt'))
            target_training = torch.load(os.path.join(data_path, dataset, 'training_target.pt'))

            train_dataset = TensorDataset(input_training, target_training)
            if not training_parameters['distributed_training']:
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=training_parameters['data_loader_num_workers'], 
                                            pin_memory=training_parameters['data_loader_pin_memory'], pin_memory_device=device, drop_last=drop_last)
            else:
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=training_parameters['data_loader_num_workers'], 
                                            pin_memory=training_parameters['data_loader_pin_memory'], pin_memory_device=device, drop_last=drop_last,
                                            sampler=DistributedSampler(train_dataset))
            
        net.train()
        
        if training_parameters['curriculum_learning']:
            logging.info(f'{utils.using("Before data loader: ")}')
            column = int(min(input_training.shape[-1]-1, epoch / 5)) 
            mask = (input_training[:,:,column] != 0).all(dim=1)
            drop_last = (sum(mask) >= batch_size).item()
            train_dataset = TensorDataset(input_training[mask], target_training[mask])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=training_parameters['data_loader_num_workers'], 
                                    pin_memory=training_parameters['data_loader_pin_memory'], pin_memory_device=device, drop_last=drop_last)
            logging.info(f'{utils.using("After data loader: ")}')
        
        if plt_activations:
            hooks = []
            if epoch % freq_activ_rep == 0:
                # encoder
                for j, enc in enumerate(net.enc):
                    hooks.append(enc.mab1.fc_o.register_forward_hook(getActivation(j)))
                # decoder
                hooks.append(net.dec[0].mab.fc_o.register_forward_hook(getActivation(num_layers_enc)))
                for j in range(len(net.dec)):
                    if (j > 0) and (j % 3 == 0):
                        hooks.append(net.dec[j].register_forward_hook(getActivation(num_layers_classifier + int(j / 3))))
        
        for x_y_training_batch, a_training_batch in train_loader:
            
            # if training_parameters['distributed_training']:
                # print(f'[GROUP ID: {gpu_id}] Net device: {next(iter(net.module.parameters())).device}')
                # print(f'[GROUP ID: {gpu_id}] Training input device: {x_y_training_batch.device}')
                # print(f'[GROUP ID: {gpu_id}] Training target device: {a_training_batch.device}')
                # print(f'[GROUP ID: {gpu_id}] Validation input device: {input_validation.device}')
                # print(f'[GROUP ID: {gpu_id}] Validation target device: {target_validation.device}')
            
            # Random subset of the samples, to enable the model to work with different sizes of data sets
            x_y_training_batch_subset = choose_subset(x_y_training_batch, num_samples_min)
            
            if x_y_training_batch_subset.device != device:
                x_y_training_batch_subset = x_y_training_batch_subset.to(device)
                a_training_batch = a_training_batch.to(device)
                        
            batch_loss, batch_grad_norm, batch_logits_norm = train(net, optimizer, x_y_training_batch_subset, criterion, a_training_batch, 
                                                epoch, training_parameters['balanced_loss'], weight_active, 
                                                training_parameters['gradient_clipping'], data_parameters, training_parameters['one_hot'], objective)
            running_loss += batch_loss
            grad_norm += batch_grad_norm
            logits_norm += batch_logits_norm
                
            # mean_loss = running_loss / len(train_loader)
            # scheduler.step(mean_loss)schedu

            if lr_schedule == 'warm-up':
                # Custom scheduler with warm-up happens once per batch
                scheduler_step(scheduler, optimizer, epoch)

        if plt_activations and epoch % freq_activ_rep == 0:
            for key in activation.keys():
                activation_list_batches[key].append(activation[key])
                    
        if lr_schedule == 'step' and early_stopper.counter > 10:
            # stepwise scheduler only happens once per epoch and only if the validation has not been going down for at least 10 epochs
            scheduler_step(scheduler, optimizer, epoch)
        
        # The none-main processes do not have to report anything
        if training_parameters['distributed_training'] and gpu_id != 0:
            continue
        
        if plt_activations and epoch % freq_activ_rep == 0:
            for j, hook in enumerate(hooks):
                hook.remove()
                # plt.imshow(torch.cat(activation_list_batches[j], dim=0).norm(dim=0), cmap='hot', aspect='auto')
                plt.imshow(torch.cat(activation_list_batches[j], dim=0).norm(dim=1, p=1) / activation_list_batches[j][0].shape[1],
                        cmap='hot', aspect='auto')
                plt.colorbar()
                if j < num_layers_enc:
                    plt.title(f'Activations layer (encoder) {j + 1}; training epoch {epoch}')
                else:
                    plt.title(f'Activations layer (decoder) {j + 1 - num_layers_enc}; training epoch {epoch}')
                    
                plt.tight_layout()
                if j < num_layers_enc:
                    plt.savefig(os.path.join(directory,
                                f'Datetime_{d_time}_encoder_neuron_activations_layer_{j}_training_epoch_{epoch}.png'))
                else:
                    plt.savefig(os.path.join(directory,
                                f'Datetime_{d_time}_decoder_neuron_activations_layer_{j}_training_epoch_{epoch}.png'))
                plt.close()
                activation_list_batches[j] = []
        
        if epoch % report_every == report_every - 1:
            epochs.append(epoch)
            net.eval()
            
            logits = predict_batchwise(net, input_validation, batch_size, num_samples_min, return_logits=True)

            if objective == 'mask' or not training_parameters['one_hot']:
                prediction = sigmoid(logits)
                if isinstance(criterion, torch.nn.BCELoss):
                    if ((0 > prediction) | (1 < prediction)).any():
                        print(f'Training: Input value outside [0,1] observed.')
                    if prediction.isnan().any():
                        print(f'Training: Input value nan observed.')
                    if training_parameters['balanced_loss'] == True:
                        criterion.weight = get_bce_weight(weight_active, target)  
                validation_loss = criterion(prediction, compute_ce(logits, target_validation, data_parameters, criterion))
            else:
                if isinstance(criterion, torch.nn.CrossEntropyLoss):
                    validation_loss = compute_ce(logits, target_validation, data_parameters, criterion)
                    # do NOT execute before calling compute_ce, since it changes the logits:
                    prediction = utils.classification_layer(logits, data_parameters, sigmoid, training_parameters['one_hot']) 
                else:
                    prediction = utils.classification_layer(logits, data_parameters, sigmoid, training_parameters['one_hot'])
                    validation_loss = criterion(prediction, target_validation)


            # only one GPU has to report everything
            if (not training_parameters['distributed_training']) or gpu_id == 0:
                report_example_predictions(prediction, target_validation, num_examples=training_parameters['num_example_predictions'],
                                model_parameters=data_parameters, epoch=epoch, file_name=os.path.join(directory,
                                f'Datetime_{d_time}_Loss_training_set_size_{training_set_size}_batch_size_'
                                f'{training_parameters["batch_size"]}_hidden_dim_{training_parameters["hidden_dim"]}.txt'),
                                dimension=data_parameters['d_max'], objective=objective)    
            
            validation_loss_list.append(validation_loss.cpu().detach().numpy())
            training_loss_list.append(running_loss / report_every / (len(train_loader)))
            grad_norm_list.append(grad_norm / report_every / (len(train_loader)))
            logits_norm_list.append(logits_norm / report_every / (len(train_loader)))
            running_loss = 0.0
            grad_norm = 0
            logits_norm = 0
            
            # f1_val = np.round(f1_score(target_validation.detach().cpu().numpy(), a_pred.detach().cpu().numpy(), average='micro'))
            # f1_list.append(f1_val)
            if validation_loss < best_loss:
                best_loss = validation_loss
                filename = os.path.join(directory, f'Datetime_{d_time}_Loss_training_set_size_{training_set_size}_batch_size_' +
                                                   f'{training_parameters["batch_size"]}_hidden_dim_{training_parameters["hidden_dim"]}.pt')
                
                if training_parameters['distributed_training']:
                    utils.checkpoint(net.module, filename)
                else:
                    utils.checkpoint(net, filename)

            # Early stopping
            if training_parameters['early_stopping'] and epoch > 50:
                if early_stopper.early_stop(validation_loss):
                    logging.info(f'EP {epoch}: Early stopping')
                    
                    if training_parameters['distributed_training']:
                        flag_tensor += 1
                    else:
                        break
        if epoch > report_every - 2:
            logging.info(f'[{epoch + 1:5d}] Training loss: {training_loss_list[-1]:.8f}, Validation loss: '
                        f'{validation_loss_list[-1]:.8f}, Gradient norm: {grad_norm_list[-1]:.8f}, Logits: {logits_norm_list[-1]:.8f}')
            logging.info(f'{utils.using("CPU Memory")}')

    # only one GPU has to report everything
    if training_parameters['distributed_training'] and gpu_id != 0:
        return net
    
    if training_parameters['loss'] == 'ASL' and asl_obj.update:
        logging.info(f'The last gamma_negative of ASL was {asl_obj.gamma_negative}.')
        
    optimizer.zero_grad(set_to_none=True)
    if training_parameters['distributed_training']:
        utils.resume(net.module, filename)
    else:
        utils.resume(net, filename)
    
    plt.plot(epochs, training_loss_list, label='training loss')
    plt.plot(epochs, validation_loss_list, label='validation loss')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(directory,
                             f'Datetime_{d_time}_Loss_training_set_size_{training_set_size}_batch_size_'
                             f'{training_parameters["batch_size"]}_hidden_dim_{training_parameters["hidden_dim"]}.png'))
    plt.plot(epochs, grad_norm_list, label='gradient norm')
    plt.plot(epochs, logits_norm_list, label='logits norm')
    # plt.plot(epochs, f1_list, label='F1 score (val)')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(directory,
                             f'Datetime_{d_time}_analytics_training_set_size_{training_set_size}_batch_size_'
                             f'{training_parameters["batch_size"]}_hidden_dim_{training_parameters["hidden_dim"]}.png'))
    plt.close()
    
    if training_parameters['distributed_training']:
        net = net.module
    
    torch.save(net, os.path.join(directory,
                                 f'Datetime_{d_time}_Loss_training_set_size_{training_set_size}_batch_size_'
                                 f'{training_parameters["batch_size"]}_hidden_dim_{training_parameters["hidden_dim"]}.'
                                 f'pt'))
    
    if training_parameters['distributed_training']:
        dist.destroy_process_group()
        
    return net


def evaluate_prec_rec_F1(a_pred, a_target, name, results_dict):
    a_pred = (a_pred.cpu().detach().numpy() > 0.5).astype(int)
    # a_target = (a_training.detach().numpy().squeeze(-1) + 1) / 2
    a_target = a_target.cpu().detach().numpy()
    average = 'micro'

    results_dict[f'Precision score ({name} data)'].append(
        np.round(precision_score(a_target, a_pred, average=average), decimals=5))
    results_dict[f'Recall score ({name} data)'].append(
        np.round(recall_score(a_target, a_pred, average=average), decimals=5))
    results_dict[f'F1 score ({name} data)'].append(np.round(f1_score(a_target, a_pred, average=average), decimals=5))

    logging.info(f'Precision score ({name} data): {results_dict[f"Precision score ({name} data)"][-1]}')
    logging.info(f'Recall score ({name} data): {results_dict[f"Recall score ({name} data)"][-1]}')
    logging.info(f'F1 score ({name} data): {results_dict[f"F1 score ({name} data)"][-1]}')


def evaluate_covering(a_pred, a_target, name, results_dict, quantile):
    # Get the score of how many formulas got a complete covering (all important coefficients have been marked as
    # important).
    # Furthermore, calculate the average number of coefficients, in the case that a full covering has been reached.
    a_pred = a_pred.cpu().detach().numpy()
    if quantile:
        cutoff = np.quantile(a_pred, 0.7)
        description = 'quantile cutoff'
    else:
        cutoff = 0.2
        description = '0.2 cutoff'
    a_pred = (a_pred >= cutoff).astype(int)
    a_target = a_target.cpu().detach().numpy()

    complete_coverings = np.sum(a_target > a_pred, axis=1) == 0
    covering_score = np.round(np.mean(complete_coverings), decimals=5)

    results_dict[f'Covering score ({name} data) ({description})'].append(covering_score)
    logging.info(f'Covering score ({name} data) ({description}): {covering_score}')

    coverings = a_pred[complete_coverings]
    avg_complete_cover_size = coverings.mean()

    results_dict[f'Complete cover size ({name} data) ({description})'].append(avg_complete_cover_size)
    logging.info(f'Complete cover size ({name} data) ({description}): {avg_complete_cover_size}')

def choose_subset(x_y, num_samples_min):
    num_samples = x_y.shape[1]
    batch_num_samples = np.random.randint(num_samples_min, num_samples+1)
    subset = np.random.choice(range(num_samples), batch_num_samples, replace=False)
    return x_y[:, subset]    

def predict_batchwise(net, input, batch_size, num_samples_min, return_logits=False):
    dataset = TensorDataset(input)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    prediction = []

    with torch.no_grad():
        for input_batch in loader:
            # Random subset of the samples, to enable the model to work with different sizes of data sets
            input_batch_subset = choose_subset(input_batch[0], num_samples_min)  # input_batch is a list with 1 element
            prediction.append(net(input_batch_subset, return_logits=return_logits))
    
    return torch.cat(prediction, dim=0)

def predict_data_loader(net, dataloader, num_samples_min):
    prediction_list = []
    target_list = []
    with torch.no_grad():
        for input, target in dataloader:
            input = input.to(device)
            target = target.to(device)
            # Random subset of the samples, to enable the model to work with different sizes of data sets
            input_subset = choose_subset(input, num_samples_min)
            prediction_list.append(net(input_subset))
            
            target_list.append(target)
    return torch.cat(prediction_list, dim=0), torch.cat(target_list, dim=0)

def log_and_save_evaluation(value, key, results_dict):
    value = np.round(value.cpu().detach().numpy(), decimals=5)
    logging.info(f'{key}: {value}')
    if not key in results_dict.keys():
        results_dict[key] = []
    results_dict[key].append(value)

def check_success(target, topk_predictions):
    return ((topk_predictions - target) == 0).all(dim=1).any()

def evaluate_top_k_predictions(prediction, target, name, results_dict, function_names_str, data_parameters, k):
    target = utils.one_hot_to_categorical(target, data_parameters)
    
    n_functions = len(data_parameters['function_names_str'])
    # Use for each polynomial 1+max_degree variables, unless the max_degree is 0; Same for the functions (+1 in case there is no function used)
    lengths = [data_parameters['degree_input_numerator'], data_parameters['degree_input_denominator'], data_parameters['degree_output_numerator'], 
               data_parameters['degree_output_denominator'], len(data_parameters['function_names_str'])]
    
    # reduce the target to omit everything for which we have no predictions 
    mask_target = [True if length>0 else False for length in lengths[:4]] # which degrees are bigger than 0
    mask_target = mask_target + [True for i in range(n_functions)] 
    target = target[:, mask_target]
    
    
    lengths = [length + 1 for length in lengths if length > 0]  
    
    
    n_successes = 0
    for i in range(prediction.shape[0]):
        list_probs = []
        index = 0
        for length in lengths:
            list_probs.append(torch.softmax(prediction[i, index:index+length], dim=0))
            index += length
        probabilities = utils.compute_comb_product(list_probs) 
        v, indices = torch.topk(probabilities.flatten(), k=k)
        topk_predictions = torch.tensor(np.unravel_index(indices.cpu().detach().numpy(), probabilities.shape), device=target.device).T
        if n_functions > 1:
            # functions in target are one hot encoded, why they are categorical in topk_predictions
            topk_predictions = torch.cat([topk_predictions[:, :-1], nn.functional.one_hot(topk_predictions[:,-1], num_classes=n_functions+1)[:, 1:]],
                                         dim=1)
        n_successes += check_success(target[i], topk_predictions)
    log_and_save_evaluation(n_successes / prediction.shape[0], f'Accuracy whole mps top {k} predictions ({name} data)', results_dict)
    
def evaluate_categorical(prediction, target, name, results_dict, function_names_str):
    prediction = prediction.round()
    precision = ((prediction - target) == 0).sum() / prediction.numel()
    log_and_save_evaluation(precision, f'Accuracy ({name} data)', results_dict)

    for i, poly in enumerate(['inp_num', 'inp_den', 'out_num', 'out_den'] + function_names_str):
        precision = ((prediction[:, i] - target[:, i]) == 0).sum() / prediction.shape[0]
        log_and_save_evaluation(precision, f'Accuracy {poly} ({name} data)', results_dict)

    acc_whole_mps = (torch.norm(target - prediction, dim=1) == 0).sum() / prediction.shape[0]
    log_and_save_evaluation(acc_whole_mps, f'Accuracy whole mps ({name} data)', results_dict)

def evaluate_categorical_subgroups(prediction, target, name, results_dict, function_names_str):
    # target consists out: encoding_input_numerator, encoding_input_denominator, encoding_output_numerator, encoding_output_denominator, encoding_functions
    
    # Only look at the polynomials
    reduced_target = torch.cat([target[:,:2], target[:,3:]], dim=1)
    polynomial_mask = (reduced_target == torch.zeros(reduced_target.shape[1], device=target.device)).all(dim=1)
    evaluate_categorical(prediction[polynomial_mask], target[polynomial_mask], 'polynomial ' + name, results_dict, function_names_str)
    
    # Only look at rational functions (polynomials excluded)
    reduced_target = torch.cat([target[:,:2], target[:,4:]], dim=1)
    rational_mask = (reduced_target == torch.zeros(reduced_target.shape[1], device=target.device)).all(dim=1)  # all rational functions
    rational_mask = rational_mask ^ polynomial_mask  # all rational functions that are not polynomials
    if rational_mask.any() > 0:
        evaluate_categorical(prediction[rational_mask], target[rational_mask], 'rational function ' + name, results_dict, function_names_str)
    
    for i, function_name_str in enumerate(function_names_str):
        # Only look at functions involving function_name_str
        reduced_target = torch.cat([target[:,4:4+i], target[:,4+i+1:]], dim=1)  # Contains the columns of the other functions
        function_mask = (reduced_target == torch.zeros(reduced_target.shape[1], device=target.device)).all(dim=1)  # all functions not involving the other functions
        function_mask = function_mask ^ (rational_mask | polynomial_mask)
        if function_mask.any() > 0:
            evaluate_categorical(prediction[function_mask], target[function_mask], f'{function_name_str} ' + name, results_dict, function_names_str)

def evaluate(net, input_training, target_training, target_test, input_test, results_dict, batch_size, objective, num_samples_min, function_names_str, one_hot, data_parameters):
    train_dataset = TensorDataset(input_training, target_training)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    prediction_training, target_training = predict_data_loader(net, train_loader, num_samples_min)
    prediction_test = predict_batchwise(net, input_test, batch_size, num_samples_min)

    if objective == 'mask' or one_hot:
        criterion = nn.BCELoss()
        if training_parameters['balanced_loss'] == True:
            weight_active = (1 - target_training).sum() / target_training.sum()
            criterion.weight = get_bce_weight(weight_active, target_test)  
        bce_test = np.round(criterion(prediction_test, target_test).item(), decimals=5)
        results_dict['bce_test'].append(bce_test)
        logging.info(f'BCE (test data): {bce_test}')
    
    if objective == 'mask':
        evaluate_prec_rec_F1(prediction_training, target_training, 'training', results_dict)
        evaluate_prec_rec_F1(prediction_test, target_test, 'test', results_dict)

        evaluate_covering(prediction_training, target_training, 'training', results_dict, True)
        evaluate_covering(prediction_test, target_test, 'test', results_dict, True)
        evaluate_covering(prediction_training, target_training, 'training', results_dict, False)
        evaluate_covering(prediction_test, target_test, 'test', results_dict, False)
    else:
        try:
            evaluate_top_k_predictions(prediction_test, target_test, 'test', results_dict, function_names_str, data_parameters, k=3)
            evaluate_top_k_predictions(prediction_test, target_test, 'test', results_dict, function_names_str, data_parameters, k=5)
            evaluate_top_k_predictions(prediction_test, target_test, 'test', results_dict, function_names_str, data_parameters, k=10)
            evaluate_top_k_predictions(prediction_test, target_test, 'test', results_dict, function_names_str, data_parameters, k=20)
        except RuntimeError:
            logging.info(f'The used degrees where too low to compute all topk accuracies.')
            
        if one_hot:
            target_training = utils.one_hot_to_categorical(target_training, data_parameters)
            target_test = utils.one_hot_to_categorical(target_test, data_parameters)
            prediction_training = utils.one_hot_to_categorical(prediction_training, data_parameters)
            prediction_test = utils.one_hot_to_categorical(prediction_test, data_parameters)            
            
        evaluate_categorical_subgroups(prediction_training, target_training, 'training', results_dict, function_names_str)
        evaluate_categorical(prediction_training, target_training, 'training', results_dict, function_names_str)
        
        evaluate_categorical_subgroups(prediction_test, target_test, 'test', results_dict, function_names_str)
        evaluate_categorical(prediction_test, target_test, 'test', results_dict, function_names_str)
        
    criterion = nn.MSELoss()
    mse_test = np.round(criterion(prediction_test, target_test).item(), decimals=5)
    results_dict['mse_test'].append(mse_test)
    logging.info(f'MSE (test data): {mse_test}')

def construct_result_dict(entry_names, data_parameters_dict, training_parameters_dict):
    results_dict = {**{key: [] for key in data_parameters_dict[0].keys()},
                    **{key: [] for key in training_parameters_dict[0].keys()}}
    for entry_name in entry_names:
        results_dict[entry_name] = []
    return results_dict

def append_results_dict(results_dict, data_parameters, training_parameters, t_training,
                        t_data_creation):
    for key in data_parameters.keys():
        results_dict[key].append(data_parameters[key])
    for key in training_parameters.keys():
        results_dict[key].append(training_parameters[key])
    results_dict['t_training'].append(t_training)
    results_dict['t_data_creation'].append(t_data_creation)
    
    
if __name__ == '__main__':
    d_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
    directory = os.path.join(results_path, d_time + experiment_name)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    shutil.copy(os.path.join('trainingOnSyntheticData', 'config', config_name), directory)
    print(f'Created directory {directory}')

    logging.basicConfig(filename=os.path.join(directory, 'experiment.log'), level=logging.INFO)
    logging.info('Starting the logger.')
    logging.debug(f'Directory: {directory}')
    logging.debug(f'File: {__file__}')

    logging.info(f'Using {device}.')

    logging.info(f'############### Starting experiment with config file {config_name} ###############')

    training_parameters_dict = dict(config.items("TRAININGPARAMETERS"))
    training_parameters_dict = {key: ast.literal_eval(training_parameters_dict[key]) for key in
                                training_parameters_dict.keys()}
    training_parameters_dict = utils.get_hyperparameters_combination(training_parameters_dict)

    data_parameters_dict = dict(config.items("DATAPARAMETERS"))
    data_parameters_dict = {key: ast.literal_eval(data_parameters_dict[key]) for key in
                            data_parameters_dict.keys()}
    data_parameters_dict = utils.get_hyperparameters_combination(data_parameters_dict, except_keys=['function_names_str', 'dataset'])
    
    objective = config['META']['objective']

    entry_names = ['t_training', 't_data_creation', 'mse_test'] 
    if objective == 'mask': 
        entry_names += ['bce_test',
                   'Precision score (training data)', 'Recall score (training data)',
                   'F1 score (training data)', 'Precision score (test data)', 'Recall score (test data)',
                   'F1 score (test data)', 
                   'Covering score (training data) (quantile cutoff)',
                   'Covering score (training data) (0.2 cutoff)',
                   'Covering score (test data) (quantile cutoff)',
                   'Covering score (test data) (0.2 cutoff)',
                   'Complete cover size (training data) (quantile cutoff)',
                   'Complete cover size (training data) (0.2 cutoff)',
                   'Complete cover size (test data) (quantile cutoff)',
                   'Complete cover size (test data) (0.2 cutoff)']
    else:
        if training_parameters_dict[0]['one_hot']:
            entry_names += ['bce_test']
        # for name in ['training', 'test']:
        #     entry_names.append(f'Accuracy ({name} data)')
        #     for poly in (['inp_num', 'inp_den', 'out_num', 'out_den'] + data_parameters_dict[0]['function_names_str']):
        #         entry_names.append(f'Accuracy {poly} ({name} data)')
        
    results_dict = construct_result_dict(entry_names, data_parameters_dict, training_parameters_dict)

    for i, data_parameters in enumerate(data_parameters_dict):
        logging.info(f"###{i + 1} out of {len(data_parameters_dict)} data set parameter combinations ###")
        print(f'Data parameters: {data_parameters}')
        logging.info(f'Data parameters: {data_parameters}')
        training_set_size = data_parameters['training_set_size']
        validation_set_size = data_parameters['validation_set_size']
        test_set_size = data_parameters['test_set_size']
        dataset_size = training_set_size + validation_set_size + test_set_size
        t_0 = time()
        generate_new_data = data_parameters['generate_new_data']
        
        d_max = data_parameters['d_max']
        d_min = data_parameters['d_min']
        if data_parameters['flexible_dim']:
            assert d_max >= d_min
        else:
            assert d_max == d_min

        if generate_new_data:
            if objective == 'mask':
                input, target, params  = utils.create_dataset_flexible_grid(dataset_size, data_parameters['ensure_uniqueness'],
                                                data_parameters['normalization'],
                                                num_samples = data_parameters['num_samples_max'], 
                                                dimension=data_parameters['dimension'],
                                                n_functions_max=data_parameters['n_functions_max'], 
                                                num_visualizations=data_parameters['num_visualizations'],
                                                function_dict=function_dict, function_name_dict=function_name_dict,
                                                device=training_parameters_dict[0]['data_device'], model_parameters=data_parameters,
                                                debugging=config['META']['experiment_name'].startswith('debug'),
                                                singularity_free_domain=data_parameters['singularity_free_domain'],
                                                param_distribution=data_parameters['param_distribution'])
            else:
                input, list_a, list_params, target  = utils.create_dataset_model_parameters(dataset_size+validation_set_size+test_set_size, 
                                        data_parameters['ensure_uniqueness'],
                                        data_parameters['degree_input_numerator'],
                                        data_parameters['degree_output_numerator'], 
                                        data_parameters['width'],data_parameters['function_names_str'],
                                        data_parameters['normalization'],
                                        max_degree_output_denominator=data_parameters['degree_output_denominator'], 
                                        max_degree_input_denominator=data_parameters['degree_input_denominator'],
                                        num_samples = data_parameters['num_samples_max'], d_min=d_min, d_max=d_max,
                                        n_functions_max=data_parameters['n_functions_max'], num_visualizations=data_parameters['num_visualizations'],
                                        maximal_n_active_base_functions=1, function_dict=function_dict, function_name_dict=function_name_dict,
                                        device=training_parameters_dict[0]['data_device'], maximal_potence=data_parameters['maximal_potence'],
                                        debugging=config['META']['experiment_name'].startswith('debug'),
                                        most_complex_function=data_parameters['most_complex_function'],
                                        singularity_free_domain=data_parameters['singularity_free_domain'],
                                        param_distribution=data_parameters['param_distribution'],
                                        use_fisher_yates_shuffle=data_parameters['use_fisher_yates_shuffle'],
                                        target_noise=data_parameters['target_noise'])
                if training_parameters_dict[0]['one_hot']:
                    target = utils.categorical_to_one_hot(target, data_parameters)
    
            
            # in case the data set is bigger than expected
            input = input[:dataset_size]
            target = target[:dataset_size]
            
            # Now implement the prediction of the model parameters
            num_dismissed_datasets = dataset_size - input.shape[0] 
            if not data_parameters['enforce_training_set_size']:
                training_set_size -= num_dismissed_datasets
                logging.info(f'Training set got reduced by {num_dismissed_datasets} data sets because' +
                    f'of nan values and contains now {training_set_size} many functions')
            
            assert training_set_size > 0
            

            input_training = input[:training_set_size]
            target_training = target[:training_set_size]
            input_validation = input[training_set_size:training_set_size + validation_set_size]
            target_validation = target[training_set_size:training_set_size + validation_set_size]
            input_test = input[training_set_size + validation_set_size:]
            target_test = target[training_set_size + validation_set_size:]        
            logging.info(f'{utils.using("CPU Memory after splitting the data")}')
            
        else:
            data_path = config['META']['data_path']
            dataset = data_parameters['dataset'][0]
            
            input_training = torch.load(os.path.join(data_path, dataset, 'training_input.pt'))
            target_training = torch.load(os.path.join(data_path, dataset, 'training_target.pt'))
            input_validation = torch.load(os.path.join(data_path, dataset, 'validation_input.pt'))
            target_validation = torch.load(os.path.join(data_path, dataset, 'validation_target.pt'))
            input_test = torch.load(os.path.join(data_path, dataset, 'test_input.pt'))
            target_test = torch.load(os.path.join(data_path, dataset, 'test_target.pt'))      
            logging.info(f'{utils.using("CPU Memory after loading the data")}')
        t_1 = time()
        t_data_creation = np.round(t_1 - t_0, 3)
        logging.info(f'Creating the dataset took {t_data_creation}s.')

        for i, training_parameters in enumerate(training_parameters_dict):
            logging.info(f"###{i + 1} out of {len(training_parameters_dict)} training parameter combinations ###")
            print(f'Training parameters: {training_parameters}')
            logging.info(f'Training parameters: {training_parameters}')
            
            batch_size = training_parameters['batch_size']
            logging.info(f'Batch size: {batch_size}')
            
                        
            if training_parameters_dict[0]['one_hot']:
                target_training = utils.label_smoothing(target_training, training_parameters['alpha_label_smoothing'], data_parameters)
            logging.info(f'{utils.using("CPU Memory after label smoothing")}')
            
            if generate_new_data:
                pass
            
                # the following is currently moved insed the trainer function (made it easier with curriculum learning)
                
                # train_dataset = TensorDataset(input_training, target_training)
                # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # if training_parameters['balanced_loss'] == True:
                #     weight_active = (1 - target_training).sum() / target_training.sum() 
                # else:
                #     weight_active = None
            else:
                pass
                # logging.ERROR(f'Loading old data is currently deprecated. Please create new one.')
                # raise NotImplementedError('Loading old data is currently deprecated. Please create new one.')
                # train_dataset = CustomDataset(directory=os.path.join(data_path, dataset))
                # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
                
                # assert not training_parameters['balanced_loss']
                
                # weight_active = None
            
            input_validation = input_validation.to(device)
            target_validation = target_validation.to(device)
            input_test = input_test.to(device)
            target_test = target_test.to(device)
            
            logging.info(f'{utils.using("CPU Memory after moving the test and validation data to the GPU")}')
            
            t_0 = time()
            d_time_train = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            if not training_parameters['distributed_training']:
                net = trainer(0, input_training, target_training, target_validation=target_validation, directory=directory,
                            input_validation=input_validation, training_parameters=training_parameters, data_parameters=data_parameters,
                            num_samples_min=data_parameters['num_samples_min'], lr_schedule=training_parameters['lr_schedule'], objective=objective,
                            d_time=d_time_train)
            else:
                world_size = torch.cuda.device_count()
                mp.spawn(trainer, args=(input_training, target_training, target_validation,
                            input_validation, training_parameters, data_parameters,
                            data_parameters['num_samples_min'], training_parameters['lr_schedule'], objective, 
                            directory, d_time_train, world_size), nprocs=world_size)

                training_set_size = input_training.shape[0]
                net = torch.load(os.path.join(directory,
                                f'Datetime_{d_time_train}_Loss_training_set_size_{training_set_size}_batch_size_'
                                f'{training_parameters["batch_size"]}_hidden_dim_{training_parameters["hidden_dim"]}.'
                                f'pt'), map_location=device)
            
            # os.remove(os.path.join(directory, 'done.txt'))
            
            t_1 = time()
            t_training = np.round(t_1 - t_0, 3)
            logging.info(f'Training the model took {t_training}s.')
            t_0 = time()
            torch.cuda.empty_cache()
            t_1 = time()
            logging.info(f'Emptying the cuda cache took {np.round(t_1 - t_0, 3)}s.')
            evaluate(net=net, input_training=input_training, target_training=target_training, target_test=target_test, input_test=input_test,
                     results_dict=results_dict, batch_size=training_parameters['batch_size'], objective=objective, 
                     num_samples_min=data_parameters['num_samples_min'], function_names_str=data_parameters['function_names_str'],
                     one_hot=training_parameters['one_hot'], data_parameters=data_parameters)
            append_results_dict(results_dict, data_parameters, training_parameters, t_training,
                                t_data_creation)
            results_pd = pd.DataFrame(results_dict)
            results_pd.T.to_csv(os.path.join(directory, 'test.csv'))

            del net
            torch.cuda.empty_cache()
            gc.collect()
