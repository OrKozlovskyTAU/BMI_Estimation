# Training
"""This file is the main file of the whole project. The first section defines the hyperparameters and setup of the model and its behavior.
The file contains the training, testing and evaluation process, as well as histograms, loss and accuracy curves. Finally the best model is being saved
"""

from typing import List
from Model import (
    ResNet_Model,
    EfficientNet_Model,
    MobileNet_Model,
    VitTransformer_Model,
    CVT_Transformer_Model,
    DenseNet_Model,
    RegNet_Model,
)
from Device_and_Seed import device_select, random_seed
from Data_process import BMIdataset, get_data, default_class_separators
from Histogram import make_error_list, make_histogram
from utils import *
import torch
from torch import nn
import torch.utils.data
from tqdm.auto import tqdm
from timeit import default_timer as timer
from datetime import datetime
import pytz
from torchinfo import summary
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import csv
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import argparse
import numpy as np
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader

# device agnostic code
device = device_select()

# parse command line arguments
parser = argparse.ArgumentParser(description='Train and test a model')
parser.add_argument(
    '--model',
    type=str,
    default='EfficientNet_Model',
    help='Model to use for training and testing',
)
parser.add_argument(
    '--folder_path',
    type=str,
    default='./Org_Aug_Seg_sorted',
    help='Path to the folder containing the data',
)

parser.add_argument(
    '--classification_activation',
    action="store_true",
    help='Whether to use classification activation or not',
)
parser.add_argument(
    '--regression_after_classification',
    action="store_true",
    help='Whether to do regression after classification',
)
parser.add_argument(
    '--manual_classes',
    action="store_true",
    help='Whether to use manual classes',
)
parser.add_argument(
    '--add_false',
    action="store_true",
    help='Whether to add false images',
)
arg = parser.parse_args()
##############################################################################################################
# IMPORTANT PARAMETERS THAT CAN BE MODIFIED
# Default Hyperparamaters / Values
SEED = 20
ACTIVATION_FUNCTION = nn.GELU()
RESNET_NUM = 'resnet101'  # used for selecting the specific resnet variant to be used
EFF = '2'  # used for selecting the specific efficient variant to be used
LOSS_FN = nn.MSELoss()  # loss function for regression
LEARN_RATE = 0.001
NUM_EPOCHS = 30
DROP_PROB = 0.1
CHECK_POINT = False  # whether to load previous weights or not
IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
NUM_WORKERS = 1  # WARNING! using too many cpu cores may result in DataLoader issues and freeze the program
checkpoint_path = '.pth'
out_features = 1  # default value of output_features (suited for regression but will be changed auto if needed)
# if classification is True than the problem becomes a classification probelm rather than a regression problem
classification_activation = arg.classification_activation

acc_tolerance = 3  # gap between true and predicted labels to be considered accurate
# scheduler define
scheduler_type = 'plateau'  # Choose - None, step or plateau
scheduler_rep = 3  # how many times scheduler will be activated - valid for plateau only
scheduler_patience = 2  # how many epochs in a row until scheduler is activated (without improving (plateau) or fixed num of epochs (step)). lowest value to set is 0 (not 1).
scheduler_factor = 0.5  # by how much the lr will be reduced
# verify that scheduler_type has a valid input
assert scheduler_type in ['None', 'step', 'plateau'], "scheduler_type must be 'None', 'step', or 'plateau'"

folder_path = Path(arg.folder_path)

train_folder_path = Path.joinpath(folder_path, 'train')
test_folder_path = Path.joinpath(folder_path, 'test')
train_csv_path = Path.joinpath(folder_path, 'CSV/train_data.csv')
test_csv_path = Path.joinpath(folder_path, 'CSV/test_data.csv')

save_figure_path = './saved_figures/'

# path to save results (pth file) along with weights and parameters
save_results_path = './saved_pth/'

check_and_create_folder(save_results_path)

# number of input channels
INPUT_CHANNELS = 3

# PREREQUISITE for the next 2 boolean variables is classification_activation = True, located in Data_process.py
# decide whether to do regression in each class after it was classified - # PREREQUISITE is classification_activation = True
regression_after_classification = arg.regression_after_classification

# merging classes together after the initial classification and before the regression - # PREREQUISITE is classification_activation = True
merge_classes = False

# classification before regression based on true labels (without letting the model decide)
manual_classes = arg.manual_classes
add_false = arg.add_false  # decide whether to add false images from different classes for better generalization

# define the min number of epochs to save data - no need to save if there are only a few epochs, since it is probably a test run
epoch_save_threshold = 0

# whether to clip the results or not
clipping_active = False

config = {
    'model': arg.model,
    'folder_path': folder_path.name,
    'activation_function': ACTIVATION_FUNCTION,
    'seed': SEED,
    'resnet_num': RESNET_NUM,
    'efficient_num': EFF,
    'learn_rate': LEARN_RATE,
    'num_epochs': NUM_EPOCHS,
    'drop_prob': DROP_PROB,
    'check_point': CHECK_POINT,
    'img_size': IMG_SIZE,
    'batch_size': BATCH_SIZE,
    'num_workers': NUM_WORKERS,
    'classification_activation': classification_activation,
    'acc_tolerance': acc_tolerance,
    'scheduler_type': scheduler_type,
    'scheduler_rep': scheduler_rep,
    'scheduler_patience': scheduler_patience,
    'scheduler_factor': scheduler_factor,
    'input_channels': INPUT_CHANNELS,
    'merge_classes': merge_classes,
    'manual_classes': manual_classes,
    'add_false': add_false,
    'epoch_save_threshold': epoch_save_threshold,
    'clipping_active': clipping_active,
}
##############################################################################################################

# selecting a random seed to see progress in testing
random_seed(seed=SEED)

# default value for the number of epochs in previous runs
previous_num_of_epochs = 0

train_data_custom, test_data_custom, train_dataloader, test_dataloader, train_transform, test_transform = get_data(
    folder_path=folder_path,
    save_figure_path=save_figure_path,
    classification_activation=classification_activation,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

if classification_activation:
    out_features = train_data_custom.num_of_classes
    LOSS_FN = nn.CrossEntropyLoss()  # loss function for classification

models_cls_dict = {
    'ResNet_Model': ResNet_Model,
    'EfficientNet_Model': EfficientNet_Model,
    'MobileNet_Model': MobileNet_Model,
    'VitTransformer_Model': VitTransformer_Model,
    'CVT_Transformer_Model': CVT_Transformer_Model,
    'DenseNet_Model': DenseNet_Model,
    'RegNet_Model': RegNet_Model,
}


# Instantiating the model and passing to device
model = models_cls_dict[arg.model](
    out_features=out_features,
    activation_fn=ACTIVATION_FUNCTION,
    drop_prob=DROP_PROB,
    classification=classification_activation,
).to(device)

# Optimizer
OPTIMIZER = torch.optim.Adam(params=model.parameters(), lr=LEARN_RATE)

if CHECK_POINT:
    # Load the model
    checkpoint = torch.load(save_results_path + checkpoint_path)
    # Load the model weights
    model.load_state_dict(checkpoint['Model_state_dict'])
    # Get the total number of epochs performed in previous runs
    previous_num_of_epochs = checkpoint['Hyperparameters']['Total number of epochs']

# reg after class can not take place if class is not active
if not classification_activation:
    regression_after_classification = False
    manual_classes = False

# loss function (criterion) and optimizer
loss_fn = LOSS_FN
optimizer = OPTIMIZER

config.update(
    {
        'out_features': out_features,
        'regression_after_classification': regression_after_classification,
        'manual_classes': manual_classes,
        'add_false': add_false,
        'previous_num_of_epochs': previous_num_of_epochs,
        'loss_fn': loss_fn.__class__.__name__,
        'optimizer': optimizer.__class__.__name__,
    }
)


class Scheduler_select:
    def __init__(self, optimizer=OPTIMIZER, scheduler_type=None, **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        # self.kwargs = kwargs
        self.scheduler = None

        if optimizer and scheduler_type:
            if self.scheduler_type == 'step':
                self.scheduler = StepLR(optimizer, **kwargs)
            elif self.scheduler_type == 'plateau':
                self.scheduler = ReduceLROnPlateau(optimizer, **kwargs)
            else:
                raise ValueError("Invalid scheduler type")

    def step(self, eval_metric):
        if self.scheduler_type == 'step':
            self.scheduler.step()
        elif self.scheduler_type == 'plateau':
            self.scheduler.step(eval_metric)


def save_model(
    state_dict: nn.Module,
    model_results: dict,
    time_dict: dict,
    file_save_folder: str = '',
    act_fn: torch.nn = ACTIVATION_FUNCTION,
    architecture: str = model.__class__.__name__,
    loss_fn: nn.Module = LOSS_FN,
    lr: float = LEARN_RATE,
    opt_fn: torch.optim = OPTIMIZER,
    current_num_of_epoch: int = NUM_EPOCHS,
    check_point: bool = CHECK_POINT,
    num_train_images: int = len(train_data_custom),
    num_test_images: int = len(test_data_custom),
    drop_prob: float = DROP_PROB,
    classification_activation: bool = classification_activation,
    class_idx: str = 'N/A',
    final_res_after_reg_class: bool = False,
):
    '''Saving the model results, hyperparameters and state dicts. Also appending the data to a csv file'''

    # Total run time
    elpased_time = str(time_dict['Hours']) + ':' + str(time_dict['Minutes']) + ':' + str(time_dict['Seconds'])

    # Total number of epochs including previous runs
    total_num_of_epochs = current_num_of_epoch + previous_num_of_epochs

    # total number of images (train + test)
    total_num_of_images = num_train_images + num_test_images

    # getting the losses and accuracies from the dictionary
    train_loss = model_results['train_loss']
    test_loss = model_results['test_loss']
    train_acc = model_results['train_acc']
    test_acc = model_results['test_acc']
    mae = model_results['mae']
    mape = model_results['mape']

    if final_res_after_reg_class:
        hyperparameters = {
            'Folder name': '',
            'Activation function': '',
            'Architecture': '',
            'Loss function': '',
            'Learning rate': '',
            'Optimizer': '',
            'Drop-out Probability': '',
            'Scheduler type': '',
            'Scheduler repetitions': '',
            'Scheduler patience': '',
            'Scheduler factor': '',
            'Current number of epochs': '',
            'Total number of epochs': '',
            'Train transforms': '',
            'Test transforms': '',
            'Check point': '',
            'Train images': '',
            'Test images': '',
            'Total images': '',
            'Classification task': '',
            'Class index': '',
        }
        results = {
            'MAE (min)': mae,
            'MAPE (min)': mape * 100,
            'Elapsed time': '',
            'Train loss (min)': train_loss,
            'Test loss (min)': test_loss,
            'Train accuracy (max)': train_acc * 100,
            'Test accuracy (max)': test_acc * 100,
        }

    else:
        # making a dict for the hyperparameters
        hyperparameters = {
            'Folder name': folder_path.name,
            'Activation function': act_fn,
            'Architecture': architecture,
            'Loss function': loss_fn,
            'Learning rate': lr,
            'Optimizer': opt_fn,
            'Drop-out Probability': drop_prob,
            'Scheduler type': scheduler_type,
            'Scheduler repetitions': scheduler_rep,
            'Scheduler patience': scheduler_patience,
            'Scheduler factor': scheduler_factor,
            'Current number of epochs': current_num_of_epoch,
            'Total number of epochs': total_num_of_epochs,
            'Train transforms': train_transform,
            'Test transforms': test_transform,
            'Check point': check_point,
            'Train images': num_train_images,
            'Test images': num_test_images,
            'Total images': total_num_of_images,
            'Classification task': classification_activation,
            'Class index': class_idx,
            'Image size': IMG_SIZE,
            'Seed': SEED,
        }

        if not classification_activation:
            # making a dict for the results
            results = {
                'MAE (min)': min(mae),
                'MAPE (min)': min(mape) * 100,
                'Elapsed time': elpased_time,
                'Train loss (min)': min(train_loss),
                'Test loss (min)': min(test_loss),
                'Train accuracy (max)': max(train_acc) * 100,
                'Test accuracy (max)': max(test_acc) * 100,
            }
        else:
            # making a dict for the results
            results = {
                'MAE (min)': 'N/A',
                'MAPE (min)': 'N/A',
                'Elapsed time': elpased_time,
                'Train loss (min)': min(train_loss),
                'Test loss (min)': min(test_loss),
                'Train accuracy (max)': max(train_acc) * 100,
                'Test accuracy (max)': max(test_acc) * 100,
            }

    # making a dict for all the data that needs to be saved
    saved_data = {
        'Hyperparameters': hyperparameters,
        'Results': results,
        'Model_state_dict': state_dict,
        'Optimizer_state_dict': opt_fn.state_dict(),
    }

    # get time stamp
    current_time_stamp = time_stamp()

    # save name for the pth file
    file_save_name = f'model_save_{folder_path.name}_{current_time_stamp}.pth'

    if file_save_folder:
        # file save path
        file_save_path = f'{save_results_path}{file_save_folder}{file_save_name}'
        check_and_create_folder(f'{save_results_path}{file_save_folder}', is_print=False)
    else:
        file_save_path = f'{save_results_path}{file_save_name}'

    if not final_res_after_reg_class:
        # saving model state dict
        torch.save(obj=saved_data, f=file_save_path)

    # Append the data to the CSV file
    csv_file_path = f'{save_results_path}new_auto_data_from_pth.csv'

    # check if the csv file exist ans prepare a header data if not
    header_exist = True
    if not os.path.exists(csv_file_path):
        # Header for the csv file
        header_data = [
            'File',
            'Folder name',
            'Activation function',
            'Architecture',
            'Loss function',
            'Learning rate',
            'Optimizer',
            'Drop-out Probability',
            'Scheduler type',
            'Scheduler repetitions',
            'Scheduler patience',
            'Scheduler factor',
            'Current number of epochs',
            'Total number of epochs',
            'Train transforms',
            'Test transforms',
            'Check point',
            'Train images',
            'Test images',
            'Total images',
            'Classification task',
            'Class index',
            'MAE (min)',
            'MAPE (min)',
            'Elapsed time [h:m:s]',
            'Train loss (min)',
            'Test loss (min)',
            'Train accuracy % (max)',
            'Test accuracy % (max)',
        ]
        # check if the all the parameters in the dictionaries are also in the header so the file is complete
        if len(header_data) - 1 != len(hyperparameters) + len(results):
            print(
                'STOP ! -> There is an inconsistency between the features in the dictionaries (hyperparameters and results) and the header !'
            )  # 'File' in header_data is an exception and does not count
            print(f"The len of header_data (minus 'File') is: {len(header_data) - 1}")
            print(f"The len of hyperparameters + results is: {len(hyperparameters) + len(results)}")
        header_exist = False

    # saved file name
    saved_file = 'Final' if final_res_after_reg_class else os.path.basename(file_save_path)

    # Specifying the new row data
    new_row_data = [
        saved_file,
        *[hyperparameters[key] for key in saved_data['Hyperparameters']],
        *[results[key] for key in saved_data['Results']],
    ]

    # Writing a new row to the csv file, containing the data from the current run
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # if file didn't exist before, it will create a header
        if not header_exist:
            writer.writerow(header_data)
        writer.writerow(new_row_data)


def are_state_dicts_equal(saved_dict, loaded_dict):
    '''Checking that the saved and loaded state dict are the same, thus making sure that the saving went well'''
    # Check if the keys are the same
    if saved_dict.keys() != loaded_dict.keys():
        return False

    # Check if the values are equal for each key
    for key in saved_dict.keys():
        if not torch.allclose(saved_dict[key], loaded_dict[key]):
            return False

    return True


def time_stamp():
    '''Generate a timestamp string'''
    # Get the local time zone
    local_tz = pytz.timezone('Israel')

    # Get the current time in the local time zone
    local_time = datetime.now(local_tz)

    # Format the local time as a string
    return local_time.strftime("%d.%m.%Y_%H:%M:%S")
    # return datetime.now().strftime("%d.%m.%Y_%H:%M:%S")


def convert_seconds(seconds):
    '''Calculate hours, minutes, and seconds'''
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_dict = {'Hours': int(hours), 'Minutes': int(minutes), 'Seconds': int(seconds)}
    return time_dict


def plot_loss_and_acc_curves(
    model_results: dict[str, list[float]],
    additional_path: str = '',
    class_idx: int = 0,
    architecture: str = model.__class__.__name__,
):
    '''This function creates loss curves of the train and test. Inputs are the model results.'''

    # Getting the values from the dictionary
    train_loss = model_results['train_loss']
    test_loss = model_results['test_loss']
    train_acc = model_results['train_acc']
    test_acc = model_results['test_acc']

    # get the num of epochs
    epochs = range(len(train_loss))

    # paths to save the curves
    loss_curve_folder_save_path = f'{save_figure_path}loss_curves'
    acc_curve_folder_save_path = f'{save_figure_path}acc_curves'

    # checking if the paths exists
    check_and_create_folder(loss_curve_folder_save_path, is_print=False)
    check_and_create_folder(acc_curve_folder_save_path, is_print=False)

    # If the arcitecture chosen is ResNet, then get the type of resnet used
    if architecture == 'ResNet_Model':
        architecture = RESNET_NUM
    else:
        # Slicing the string to omit 'Model' from name
        _index = architecture.index('_')
        architecture = architecture[:_index]

    if additional_path:
        # creating the folders if they don't exists
        loss_curve_save_path = check_and_create_folder(
            f'{save_figure_path}loss_curves/loss_curves_{additional_path}', is_print=False
        )
        acc_curve_save_path = check_and_create_folder(
            f'{save_figure_path}acc_curves/acc_curves_{additional_path}', is_print=False
        )
        # full save path
        loss_curve_save_path = f'{loss_curve_save_path}/class_{class_idx}.jpg'
        acc_curve_save_path = f'{acc_curve_save_path}/class_{class_idx}.jpg'
    else:
        # paths to save the curves
        loss_curve_save_path = f'{save_figure_path}loss_curves/{architecture}_{time_stamp()}.jpg'
        acc_curve_save_path = f'{save_figure_path}acc_curves/{architecture}_{time_stamp()}.jpg'

    # Loss
    # setup a plot
    plt.figure(figsize=(16, 7))
    # ploting
    plt.plot(epochs, train_loss, label='Train loss')
    plt.plot(epochs, test_loss, label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 50)
    plt.title('Loss curve')
    plt.legend()
    plt.savefig(loss_curve_save_path)

    # Accuracy
    # setup a plot
    plt.figure(figsize=(16, 7))
    # multiplying each element in the list by 100
    train_acc = [acc * 100 for acc in train_acc]
    test_acc = [acc * 100 for acc in test_acc]
    # ploting
    plt.plot(epochs, train_acc, label='Train accuracy')
    plt.plot(epochs, test_acc, label='Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy [%]')
    plt.title('Accuracy curve')
    plt.legend()
    plt.savefig(acc_curve_save_path)


# Class to have default values for test results
class Test_Default_Metric:
    def __init__(self):
        self.best_pred = []
        self.best_model = []
        self.max_acc = 0


# Train step function
def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    tolerance: int,
    classification_active: bool,
    optimizer: torch.optim.Optimizer,
    classes_limits: List = [],
):

    # train mode
    model.train()

    # train loss
    train_loss = 0
    train_acc = 0
    images_file_names_classes = []
    if classification_active:
        # list of lists for the file names - used for regression after classification
        images_file_names_classes = [[] for _ in range(train_data_custom.num_of_classes)]
    num_of_samples = 0

    if classes_limits:
        class_lower_limit = classes_limits[0]
        class_upper_limit = classes_limits[1]

    for batch, (X, y, img_name) in enumerate(dataloader):

        # send data to device
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)
        y_pred = y_pred.squeeze(dim=1)

        # clipping (only applicable if active)
        if classes_limits and (max(y_pred) > class_upper_limit or min(y_pred) < class_lower_limit):
            y_pred = clipping(y_pred, class_lower_limit, class_upper_limit)

        # calculate the loss and accuracy
        loss = loss_fn(y_pred, y)
        train_loss += loss

        if classification_active:
            y_pred_soft = torch.softmax(y_pred, dim=1)
            _, predicted = torch.max(y_pred_soft, 1)
            train_acc += (predicted == y).sum().item()
        else:
            is_within_threshold = torch.abs(y_pred - y) <= tolerance
            acc_calc = torch.sum(is_within_threshold)
            train_acc += acc_calc

        # apply only for regression after classification
        if classification_active:
            # saving the file names to a list
            for i in range(len(predicted)):
                images_file_names_classes[predicted[i]].append(img_name[i])

        # optimize the zero grad
        optimizer.zero_grad()

        # loss backwards
        loss.backward()

        # optimizer step
        optimizer.step()

        # calc the number of samples in the dataset
        num_of_samples += len(X)

    # Loss and accuracy normalization according to the length of the dataloader
    train_acc = float(train_acc)
    train_acc /= num_of_samples
    train_loss /= len(dataloader)

    return train_loss, train_acc, images_file_names_classes


# Test step function
def test_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    tolerance: int,
    classification_active: bool,
    classes_limits: List = [],
):

    # test_loss
    test_loss = 0
    test_acc = 0
    pred = []  ###
    targ = []  ###
    images_file_names_classes = []
    if classification_active:
        # list of lists for the file names - used for regression after classification
        images_file_names_classes = [[] for _ in range(train_data_custom.num_of_classes)]
    num_of_samples = 0

    if classes_limits:
        class_lower_limit = classes_limits[0]
        class_upper_limit = classes_limits[1]

    # eval mode
    model.eval()
    with torch.no_grad():
        for batch, (X_test, y_test, img_name_test) in enumerate(dataloader):

            # send data to device
            X_test, y_test = X_test.to(device), y_test.to(device)

            # forward pass
            y_test_pred = model(X_test)
            y_test_pred = y_test_pred.squeeze(dim=1)

            # clipping (only applicable if active)
            if classes_limits and (max(y_test_pred) > class_upper_limit or min(y_test_pred) < class_lower_limit):
                y_test_pred = clipping(y_test_pred, class_lower_limit, class_upper_limit)

            if classification_active:
                pred.append(torch.argmax(y_test_pred).item())  ###
            else:
                pred.append(y_test_pred.item())  ###

            targ.append(y_test.item())  ###

            # calculate the loss
            loss = loss_fn(y_test_pred, y_test)
            test_loss += loss

            if classification_active:
                y_test_pred_soft = torch.softmax(y_test_pred, dim=1)
                _, predicted = torch.max(y_test_pred_soft, 1)
                test_acc += (predicted == y_test).sum().item()
            else:
                is_within_threshold = torch.abs(y_test_pred - y_test) <= tolerance
                acc_calc = torch.sum(is_within_threshold)
                test_acc += acc_calc

            # apply only for regression after classification
            if classification_active:
                # saving the file names to a list
                images_file_names_classes[predicted].append(img_name_test[0])

            # calc the number of samples in the dataset
            num_of_samples += len(X_test)

        # Loss and accuracy normalization according to the length of the dataloader
        test_acc = float(test_acc)
        test_acc /= num_of_samples
        test_loss /= len(dataloader)

    # return test_loss, test_acc, results.MIN_MAE, results.MIN_MAPE * 100, results.min_pred, targ, results.best_results
    return test_loss, test_acc, pred, targ, images_file_names_classes


# Actual training and testing function
def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    acc_tolerance: int,
    scheduler_rep: int,
    classification_active: bool,
    reg_class_active: bool = False,
    scheduler_instance: Scheduler_select = None,
    class_idx: int = None,
    default_time_stamp: str = '',
    default_metric: Test_Default_Metric = Test_Default_Metric(),
):

    # empty results dictionary
    results = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
        'mae': [],
        'mape': [],
        'best_model': default_metric.best_model,
    }

    # scheduler counter initialization
    scheduler_counter = 0
    classes_separators = default_class_separators
    classes_limits = []  # default list

    if clipping_active:
        if class_idx == 0:
            class_lower_limit = 0
            class_upper_limit = classes_separators[class_idx]
        elif class_idx == len(classes_separators):
            class_lower_limit = classes_separators[class_idx - 1]
            class_upper_limit = float('inf')
        else:
            class_lower_limit = classes_separators[class_idx - 1]
            class_upper_limit = classes_separators[class_idx]

        classes_limits = [class_lower_limit, class_upper_limit]

    if classification_active:
        suffix = " | Classifier"
    elif class_idx is not None:
        suffix = f" | Class {class_idx}"
    else:
        suffix = ""

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc, train_file_names_classes = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            tolerance=acc_tolerance,
            classification_active=classification_active,
            classes_limits=classes_limits,
        )

        test_loss, test_acc, pred, targ, test_file_names_classes = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            tolerance=acc_tolerance,
            classification_active=classification_active,
            classes_limits=classes_limits,
        )

        # printing the progress
        print(
            f'\n\nEpoch: {epoch+1} |\t \
              Train loss: {train_loss:.4f} |\t \
              Test loss: {test_loss:.4f} \n\t\t \
              Train acc: {train_acc * 100:.4f} % |\t \
              Test acc: {test_acc * 100:.4f} %'
        )

        # update results dictionary to a list
        results['train_loss'].append(train_loss.item())
        results['test_loss'].append(test_loss.item())
        results['train_acc'].append(train_acc * 100)
        results['test_acc'].append(test_acc * 100)

        # initial update
        if epoch == 0:
            default_metric = Test_Default_Metric()
            default_metric.best_pred = pred
            default_metric.best_model = model.state_dict()

        # updating best results
        if classification_active:
            if test_acc > default_metric.max_acc:
                default_metric.max_acc = test_acc
                default_metric.best_pred = pred
                default_metric.best_model = model.state_dict()
                train_file_names = train_file_names_classes
                test_file_names = test_file_names_classes
                print('File names has been updated since this is the best epoch acc so far')
        else:
            MAE = mean_absolute_error(targ, pred)  ###
            MAPE = mean_absolute_percentage_error(targ, pred)  ###
            print(
                '\t\
                       Test MAE: {:.4f}     |  \
                Test MAPE: {:.4f} %\n'.format(
                    MAE, MAPE * 100
                )
            )  ###

            # Find the minimum values of MAE, MAPE and the best predictions and weights
            if results['mape'] and MAPE < min(results['mape']):
                default_metric.best_pred = pred  # the best prediction for all the test set
                default_metric.best_model = model.state_dict()  # updating the best weights
                if model.__class__.__name__ in [ResNet_Model, EfficientNet_Model, MobileNet_Model]:
                    image_from_test = next(iter(test_dataloader))[0].to(device)
                    layer_outputs(image_from_test, model, save_figure_path, model.__class__.__name__)

            # adding the latest MAE and MAPE to the list
            results['mae'].append(MAE)
            results['mape'].append(MAPE)

        if scheduler_type == 'plateau' and scheduler_counter < scheduler_rep:
            # Remember the initial learning rate
            initial_lr = optimizer.param_groups[0]['lr']
            if classification_active:
                # Step the scheduler
                scheduler_instance.step(test_acc)
            else:
                # Step the scheduler
                scheduler_instance.step(MAPE)
            # Check if the learning rate has changed
            if optimizer.param_groups[0]['lr'] != initial_lr:
                scheduler_counter += 1
        elif scheduler_type == 'step' and scheduler_counter < scheduler_rep:
            # Step the scheduler
            scheduler_instance.step()

        print('-' * 50)

    # getting the number of different classes
    num_classes = len(list(set(targ)))

    # Printing the best results from all the epochs
    print('-' * 50)

    # find the best values and it's index
    min_train_loss, min_train_loss_index = find_min_max_and_index(results['train_loss'])
    min_test_loss, min_test_loss_index = find_min_max_and_index(results['test_loss'])
    max_train_acc, max_train_acc_index = find_min_max_and_index(results['train_acc'], mode=max)
    max_test_acc, max_test_acc_index = find_min_max_and_index(results['test_acc'], mode=max)

    if not classification_active:
        min_mae, min_mae_index = find_min_max_and_index(results['mae'])
        min_mape, min_mape_index = find_min_max_and_index(results['mape'])
        # Best MAE, MAPE
        print(f'Best MAE is: {min_mae:.4f}  -  epoch: {min_mae_index + 1}')
        print(f'Best MAPE is: {min_mape * 100:.4f} %  -  epoch: {min_mape_index + 1}')

    # Best loss
    print(f"Best train loss is: {min_train_loss:.4f}  -  epoch: {min_train_loss_index + 1}")
    print(f"Best test loss is: {min_test_loss:.4f}  -  epoch: {min_test_loss_index + 1}")
    # Best loss
    print(f"Best train accuracy is: {max_train_acc:.4f} %  -  epoch: {max_train_acc_index + 1}")
    print(f"Best test accuracy is: {max_test_acc:.4f} %  -  epoch: {max_test_acc_index + 1}")
    print('-' * 50)

    # updating the dict to the latest values
    results['best_model'] = default_metric.best_model

    file_names = {}
    if reg_class_active:
        # dictionary for file names
        file_names = {'train': train_file_names, 'test': test_file_names}

    # time stamp for all histograms
    current_time = time_stamp()

    if NUM_EPOCHS > epoch_save_threshold:
        if classification_active:
            # making a histogram of the test predictions and test labels (targets) # Underweight, Normal weight, Overweight, Obese, Severely Obese
            make_histogram(
                data_list=default_metric.best_pred,
                hist_title=f'Test prediction{suffix}',
                x_label='BMI Classes',
                y_label='Quantity',
                current_time=current_time,
                classification_activation=classification_activation,
                num_classes=num_classes,
            )
            label_hist_val, _, _ = make_histogram(
                data_list=targ,
                hist_title=f'Test labels{suffix}',
                x_label='BMI Classes',
                y_label='Quantity',
                classification_activation=classification_activation,
                current_time=current_time,
                num_classes=num_classes,
            )

            # making an error list between the pred and the targ. making a histogram for the errors by BMI
            error_list, error_by_class = make_error_list(
                targ_list=targ,
                pred_list=default_metric.best_pred,
                classification_activation=classification_activation,
                num_classes=num_classes,
            )
            error_hist_val, _, _ = make_histogram(
                data_list=error_list,
                hist_title=f'Test error by BMI Classes{suffix}',
                x_label='BMI Classes',
                y_label='Quantity',
                classification_activation=classification_activation,
                current_time=current_time,
                num_classes=num_classes,
            )
            for i in range(num_classes):
                error_hist_val_by_class, _, _ = make_histogram(
                    data_list=error_by_class[i],
                    hist_title=f'Test errors - predicted classes | true class {i}{suffix}',
                    x_label='BMI Classes',
                    y_label='Quantity',
                    classification_activation=classification_activation,
                    current_time=current_time,
                    num_classes=num_classes,
                )

                # Avoid division by zero by checking for zero values in the denominator
                normalized_hist_values_by_class = np.divide(
                    error_hist_val_by_class, label_hist_val[i], where=(label_hist_val[i] != 0)
                )
                normalized_hist_values_by_class *= 100

                # making a normalized histogram
                make_histogram(
                    data_list=normalized_hist_values_by_class,
                    hist_title=f'Test error by BMI Classes - Normalized | true class {i}{suffix}',
                    x_label='BMI Classes',
                    y_label='Percentage Error [%]',
                    normal_hist=False,
                    current_time=current_time,
                    classification_activation=classification_activation,
                    num_classes=num_classes,
                )

            # Avoid division by zero by checking for zero values in the denominator
            normalized_hist_values = np.divide(error_hist_val, label_hist_val, where=(label_hist_val != 0))
            normalized_hist_values *= 100

            # making a normalized histogram
            make_histogram(
                data_list=normalized_hist_values,
                hist_title=f'Test error by BMI Classes - Normalized{suffix}',
                x_label='BMI Classes',
                y_label='Percentage Error [%]',
                normal_hist=False,
                current_time=current_time,
                classification_activation=classification_activation,
                num_classes=num_classes,
            )

        elif regression_after_classification:
            # making a histogram of the test predictions and test labels (targets)
            make_histogram(
                data_list=default_metric.best_pred,
                hist_title=f'Test prediction for class {class_idx}{suffix}',
                x_label='BMI Values',
                y_label='Quantity',
                current_time=current_time,
                default_time_stamp=default_time_stamp,
                additional_path='reg_class',
                class_idx=class_idx,
            )
            label_hist_val, _, _ = make_histogram(
                data_list=targ,
                hist_title=f'Test labels for class {class_idx}{suffix}',
                x_label='BMI Values',
                y_label='Quantity',
                current_time=current_time,
                default_time_stamp=default_time_stamp,
                additional_path='reg_class',
                class_idx=class_idx,
            )

            # making an error list between the pred and the targ. making a histogram for the errors by BMI
            error_list, _ = make_error_list(
                targ_list=targ, pred_list=default_metric.best_pred, tolerance=3
            )  # tolerance reference - https://www.nber.org/system/files/working_papers/h0108/h0108.pdf - page 9
            if len(error_list) > 0:
                error_hist_val, _, _ = make_histogram(
                    data_list=error_list,
                    hist_title=f'Test error by BMI for class {class_idx}{suffix}',
                    x_label='BMI Values',
                    y_label='Quantity',
                    current_time=current_time,
                    default_time_stamp=default_time_stamp,
                    additional_path='reg_class',
                    class_idx=class_idx,
                )

                # Avoid division by zero by checking for zero values in the denominator
                normalized_hist_values = np.divide(error_hist_val, label_hist_val, where=(label_hist_val != 0))
                normalized_hist_values *= 100

                # making a normalized histogram
                make_histogram(
                    data_list=normalized_hist_values,
                    hist_title=f'Test error by BMI for class {class_idx} - Normalized{suffix}',
                    x_label='BMI Values',
                    y_label='Percentage Error [%]',
                    normal_hist=False,
                    current_time=current_time,
                    default_time_stamp=default_time_stamp,
                    additional_path='reg_class',
                    class_idx=class_idx,
                )

            # calculate the error (L1 loss) and making a histogram of the errors themselves (not BMI dependent)
            absolute_error_array = np.abs(np.array(targ) - np.array(default_metric.best_pred))
            if len(absolute_error_array) > 0:
                make_histogram(
                    data_list=absolute_error_array,
                    hist_title=f'Test absolute error for class {class_idx} (not by BMI){suffix}',
                    x_label='Error (absolute)',
                    y_label='Quantity',
                    calc_bin=True,
                    current_time=current_time,
                    default_time_stamp=default_time_stamp,
                    additional_path='reg_class',
                    class_idx=class_idx,
                )

        else:
            # making a histogram of the test predictions and test labels (targets)
            make_histogram(
                data_list=default_metric.best_pred,
                hist_title=f'Test prediction{suffix}',
                x_label='BMI Values',
                y_label='Quantity',
                current_time=current_time,
            )
            label_hist_val, _, _ = make_histogram(
                data_list=targ,
                hist_title=f'Test labels{suffix}',
                x_label='BMI Values',
                y_label='Quantity',
                current_time=current_time,
            )

            # making an error list between the pred and the targ. making a histogram for the errors by BMI
            error_list, _ = make_error_list(
                targ_list=targ, pred_list=default_metric.best_pred, tolerance=3
            )  # tolerance reference - https://www.nber.org/system/files/working_papers/h0108/h0108.pdf - page 9
            error_hist_val, _, _ = make_histogram(
                data_list=error_list,
                hist_title=f'Test error by BMI{suffix}',
                x_label='BMI Values',
                y_label='Quantity',
                current_time=current_time,
            )

            # Avoid division by zero by checking for zero values in the denominator
            normalized_hist_values = np.divide(error_hist_val, label_hist_val, where=(label_hist_val != 0))
            normalized_hist_values *= 100

            # making a normalized histogram
            make_histogram(
                data_list=normalized_hist_values,
                hist_title=f'Test error by BMI - Normalized{suffix}',
                x_label='BMI Values',
                y_label='Percentage Error [%]',
                normal_hist=False,
                current_time=current_time,
            )

            # calculate the error (L1 loss) and making a histogram of the errors themselves (not BMI dependent)
            absolute_error_array = np.abs(np.array(targ) - np.array(default_metric.best_pred))
            make_histogram(
                data_list=absolute_error_array,
                hist_title=f'Test absolute error (not by BMI){suffix}',
                x_label='Error (absolute)',
                y_label='Quantity',
                calc_bin=True,
                current_time=current_time,
            )

    return results, file_names


def main():

    # set timer to measure how long it takes to train
    start_time = timer()

    if not manual_classes:

        # scheduler parameters
        step_scheduler_params = {'step_size': scheduler_patience, 'gamma': scheduler_factor, 'verbose': True}
        plateau_scheduler_params = {
            'mode': 'max' if classification_activation else 'min',
            'factor': scheduler_factor,
            'patience': scheduler_patience,
            'verbose': True,
        }

        # scheduling learning rate reduce
        if scheduler_type == 'step':
            scheduler_instance = Scheduler_select(scheduler_type='step', **step_scheduler_params)
        elif scheduler_type == 'plateau':
            scheduler_instance = Scheduler_select(scheduler_type='plateau', **plateau_scheduler_params)

        # Check on which device we are operating
        print("Model is currently on device:", next(model.parameters()).device)

        # get info on the model
        summary(model=model, input_size=(BATCH_SIZE, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE))

        if classification_activation:
            class_start_time = timer()

        # training the model
        model_results, all_file_names = train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=NUM_EPOCHS,
            acc_tolerance=acc_tolerance,
            scheduler_rep=scheduler_rep,
            scheduler_instance=scheduler_instance,
            reg_class_active=regression_after_classification,
            classification_active=classification_activation,
        )

        if (
            classification_activation and NUM_EPOCHS > epoch_save_threshold
        ):  # no need to save if there are only a few epochs since it is probably a test run

            # end time
            class_end_time = timer()

            # total time
            class_total_time = class_end_time - class_start_time

            # get the time it took to run in the correct form
            class_time_dict = convert_seconds(class_total_time)

            # print the total training time
            print(
                f"Total training time is: {class_time_dict['Hours']}:{class_time_dict['Minutes']}:{class_time_dict['Seconds']} (h:m:s)"
            )

            # plot the loss and acc curves
            plot_loss_and_acc_curves(model_results=model_results)

            # saving results, weights and parameters
            save_model(
                state_dict=model_results['best_model'],
                model_results=model_results,
                time_dict=class_time_dict,
                file_save_folder='classification/',
                num_train_images=len(train_data_custom),
                num_test_images=len(test_data_custom),
            )

    else:
        # getting the train custom images
        train_images, _ = train_data_custom.convert_image_names_to_classes(
            train_data_custom.images_file_names, train_data_custom.BMI_labels, add_false=add_false
        )
        # getting the test custom images - consider if add_false should be True for the test images
        test_images, _ = test_data_custom.convert_image_names_to_classes(
            test_data_custom.images_file_names, test_data_custom.BMI_labels, add_false=add_false
        )
        # dictionary for the custom images
        all_file_names = {'train': train_images, 'test': test_images}

    # layer_test_first = layer_test
    if regression_after_classification:

        delete_indices = []
        # apply if you want to merge classes based on the default BMI classes
        if merge_classes and len(default_class_separators) > 4:
            for key in all_file_names.keys():
                all_file_names[key][1] = all_file_names[key][1] + all_file_names[key][2]
                all_file_names[key][2] = all_file_names[key][3] + all_file_names[key][4]
                all_file_names[key][3] = all_file_names[key][5] + all_file_names[key][6] + all_file_names[key][7]
                all_file_names[key][4] = all_file_names[key][8]
                delete_indices = [8, 7, 6, 5]
                for index in delete_indices:
                    del all_file_names[key][index]

        empty_dict_counter = 0
        class_reg_loss_results = []
        class_reg_data_param_list = []
        default_time_stamp = time_stamp()

        # default values for the final train and test calc
        final_train_loss = 0
        final_test_loss = 0
        final_train_acc = 0
        final_test_acc = 0
        final_test_mae = 0
        final_test_mape = 0
        train_total_num_of_samples = 0
        test_total_num_of_samples = 0

        # creating a dict based on the lenght of classes after merge (if a merge occurred)
        all_file_names_sliced = {
            'train': [[] for _ in range(train_data_custom.num_of_classes - len(delete_indices))],
            'test': [[] for _ in range(train_data_custom.num_of_classes - len(delete_indices))],
        }

        # delete the first 4 characters from each file name
        for i in range(len(all_file_names['train'])):
            for file_name in all_file_names['train'][i]:
                all_file_names_sliced['train'][i].append(file_name[4:])
            for file_name in all_file_names['test'][i]:
                all_file_names_sliced['test'][i].append(file_name[4:])

        for i in range(len(all_file_names_sliced['train'])):

            # make sure there is no images from the train set in the test set and vice-versa
            for file_name in all_file_names_sliced['train'][i]:
                assert (
                    file_name not in all_file_names_sliced['test'][i]
                ), "STOP! - one of the images in the train set in also in the test set (in some variation)"
            for file_name in all_file_names_sliced['test'][i]:
                assert (
                    file_name not in all_file_names_sliced['train'][i]
                ), "STOP! - one of the images in the test set in also in the train set (in some variation)"

            # check if one of the lists is empty and if True, continue
            if not all_file_names['train'][i] or not all_file_names['test'][i]:
                empty_dict_counter += 1
                continue

            # train transformation on the images
            train_transform_class_reg = transforms.Compose(
                [
                    transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                ]
            )

            # test transformation on the images
            test_transform_class_reg = transforms.Compose(
                [
                    transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                ]
            )

            # train data set
            train_data_class_reg = BMIdataset(
                csv_path=train_csv_path,
                images_folder_path=train_folder_path,
                transform=train_transform_class_reg,
                classification=False,
                image_list=all_file_names['train'][i],
            )

            # test data set
            test_data_class_reg = BMIdataset(
                csv_path=test_csv_path,
                images_folder_path=test_folder_path,
                transform=test_transform_class_reg,
                classification=False,
                image_list=all_file_names['test'][i],
            )

            # Checking length of datasets and making sure they was created succesfully
            print(f'The len of train data is {len(train_data_class_reg)}')
            print(f'The len of test data is {len(test_data_class_reg)}')

            # Setup batch size and number of workers
            BATCH_SIZE_CLASS_REG = 32
            NUM_WORKERS_CLASS_REG = 1

            print(f"Creating DataLoader's with batch size {BATCH_SIZE_CLASS_REG} and {NUM_WORKERS_CLASS_REG} workers.")

            # Create DataLoader's
            train_dataloader_class_reg = DataLoader(
                dataset=train_data_class_reg,
                batch_size=BATCH_SIZE_CLASS_REG,
                num_workers=NUM_WORKERS_CLASS_REG,
                shuffle=True,
            )

            test_dataloader_class_reg = DataLoader(
                dataset=test_data_class_reg, batch_size=1, num_workers=NUM_WORKERS_CLASS_REG, shuffle=False
            )

            # making sure the DataLoader was created succesfully
            print(f'Train data loader: {train_dataloader_class_reg}')
            print(f'Test data loader: {test_dataloader_class_reg}')

            # printing the current class that being trained
            print(f'The currect class is: {i}')

            print('=' * 100)

            ACTIVATION_FUNCTION_CLASS_REG = nn.GELU()
            RESNET_NUM_CLASS_REG = 'resnet101'
            EFF_CLASS_REG = '2'
            LOSS_FN_CLASS_REG = nn.MSELoss()
            LEARN_RATE_CLASS_REG = 0.001
            NUM_EPOCHS_CLASS_REG = 30
            DROP_PROB_CLASS_REG = 0.1
            acc_tolerance_class_reg = 3  # gap between true and predicted labels to be considered accurate
            # scheduler define
            scheduler_type_class_reg = 'plateau'  # Choose - None, step or plateau
            scheduler_rep_class_reg = 3  # how many times scheduler will be activated - valid for plateau only
            scheduler_patience_class_reg = 2  # how many epochs in a row until scheduler is activated (without improving (plateau) or fixed num of epochs (step)). lowest value to set is 0 (not 1).
            scheduler_factor_class_reg = 0.5  # by how much the lr will be reduced
            # verify that scheduler_type has a valid input
            assert scheduler_type_class_reg in [
                'None',
                'step',
                'plateau',
            ], "scheduler_type_class_reg must be 'None', 'step', or 'plateau'"

            # Instantiating the model and passing to device
            model_class_reg = EfficientNet_Model(
                out_features=1,
                activation_fn=ACTIVATION_FUNCTION_CLASS_REG,
                drop_prob=DROP_PROB_CLASS_REG,
                type_select=EFF_CLASS_REG,
                classification=False,
            ).to(device)

            # Optimizer
            OPTIMIZER_CLASS_REG = torch.optim.Adam(params=model_class_reg.parameters(), lr=LEARN_RATE_CLASS_REG)

            # scheduler parameters
            step_scheduler_params_class_reg = {
                'step_size': scheduler_patience_class_reg,
                'gamma': scheduler_factor_class_reg,
                'verbose': True,
            }
            plateau_scheduler_params_class_reg = {
                'mode': 'min',
                'factor': scheduler_factor_class_reg,
                'patience': scheduler_patience_class_reg,
                'verbose': True,
            }

            # scheduling learning rate reduce
            if scheduler_type_class_reg == 'step':
                scheduler_instance_class_reg = Scheduler_select(
                    optimizer=OPTIMIZER_CLASS_REG, scheduler_type='step', **step_scheduler_params_class_reg
                )
            elif scheduler_type_class_reg == 'plateau':
                scheduler_instance_class_reg = Scheduler_select(
                    optimizer=OPTIMIZER_CLASS_REG, scheduler_type='plateau', **plateau_scheduler_params_class_reg
                )

            CHECK_POINT_CLASS_REG = False  # whether to load previous weights or not
            checkpoint_path_class_reg = 'model_save_Aug_Org_sorted_09.04.2024_11:44:46.pth'

            if CHECK_POINT_CLASS_REG:
                # Load the model
                checkpoint = torch.load(save_results_path + checkpoint_path_class_reg)
                # Load the model weights
                model_class_reg.load_state_dict(checkpoint)
                # Get the total number of epochs performed in previous runs
                # previous_num_of_epochs_class_reg = checkpoint['Hyperparameters']['Total number of epochs']

            # set timer to measure how long it takes to train
            temp_start_time = timer()

            # training the model
            model_results, _ = train(
                model=model_class_reg,
                train_dataloader=train_dataloader_class_reg,
                test_dataloader=test_dataloader_class_reg,
                loss_fn=LOSS_FN_CLASS_REG,
                optimizer=OPTIMIZER_CLASS_REG,
                epochs=NUM_EPOCHS_CLASS_REG,
                acc_tolerance=acc_tolerance_class_reg,
                scheduler_rep=scheduler_rep_class_reg,
                scheduler_instance=scheduler_instance_class_reg,
                classification_active=False,
                class_idx=i,
                default_time_stamp=default_time_stamp,
            )

            class_reg_loss_results.append(model_results)
            class_reg_data_param = {
                'train_samples': len(train_data_class_reg),
                'test_samples': len(test_data_class_reg),
            }
            class_reg_data_param_list.append(class_reg_data_param)

            final_train_loss += (
                np.mean(class_reg_loss_results[i - empty_dict_counter]['train_loss'])
                * class_reg_data_param_list[i - empty_dict_counter]['train_samples']
            )
            final_test_loss += (
                np.mean(class_reg_loss_results[i - empty_dict_counter]['test_loss'])
                * class_reg_data_param_list[i - empty_dict_counter]['test_samples']
            )
            final_train_acc += (
                np.mean(class_reg_loss_results[i - empty_dict_counter]['train_acc'])
                * class_reg_data_param_list[i - empty_dict_counter]['train_samples']
            )
            final_test_acc += (
                np.mean(class_reg_loss_results[i - empty_dict_counter]['test_acc'])
                * class_reg_data_param_list[i - empty_dict_counter]['test_samples']
            )
            final_test_mae += (
                np.mean(class_reg_loss_results[i - empty_dict_counter]['mae'])
                * class_reg_data_param_list[i - empty_dict_counter]['test_samples']
            )
            final_test_mape += (
                np.mean(class_reg_loss_results[i - empty_dict_counter]['mape'])
                * class_reg_data_param_list[i - empty_dict_counter]['test_samples']
            )
            train_total_num_of_samples += class_reg_data_param_list[i - empty_dict_counter]['train_samples']
            test_total_num_of_samples += class_reg_data_param_list[i - empty_dict_counter]['test_samples']

            # end time
            temp_end_time = timer()

            # total time
            temp_total_time = temp_end_time - temp_start_time

            # get the time it took to run in the correct form
            time_dict = convert_seconds(temp_total_time)

            if NUM_EPOCHS_CLASS_REG > epoch_save_threshold:

                # plot the loss and acc curves
                plot_loss_and_acc_curves(model_results=model_results, additional_path=default_time_stamp, class_idx=i)

                # saving results, weights and parameters
                save_model(
                    state_dict=model_results['best_model'],
                    model_results=model_results,
                    time_dict=time_dict,
                    file_save_folder=f'class_reg_{default_time_stamp}/class_{i}/',
                    act_fn=ACTIVATION_FUNCTION_CLASS_REG,
                    architecture=model.__class__.__name__,
                    loss_fn=LOSS_FN_CLASS_REG,
                    opt_fn=OPTIMIZER_CLASS_REG,
                    current_num_of_epoch=NUM_EPOCHS_CLASS_REG,
                    check_point=CHECK_POINT_CLASS_REG,
                    num_train_images=len(train_data_class_reg),
                    num_test_images=len(test_data_class_reg),
                    drop_prob=DROP_PROB_CLASS_REG,
                    class_idx=str(i),
                    classification_activation=False,
                )

        print(f'The number of empty classes that were not considered is: {empty_dict_counter}')
        assert len(class_reg_loss_results) + empty_dict_counter == out_features - len(
            delete_indices
        ), "ERROR! - Saved results dict doesn't have the same length as the number of classes"

        final_train_loss /= train_total_num_of_samples
        final_test_loss /= test_total_num_of_samples
        final_train_acc /= train_total_num_of_samples
        final_test_acc /= test_total_num_of_samples
        final_test_mae /= test_total_num_of_samples
        final_test_mape /= test_total_num_of_samples

        final_res = {
            'train_loss': final_train_loss,
            'test_loss': final_test_loss,
            'train_acc': final_train_acc,
            'test_acc': final_test_acc,
            'mae': final_test_mae,
            'mape': final_test_mape,
        }

        # saving final results, weights and parameters
        save_model(
            state_dict=model_results['best_model'],
            model_results=final_res,
            time_dict=time_dict,
            final_res_after_reg_class=True,
        )

        print('=' * 100)
        print('=' * 100)
        print(f'The final train loss is: {final_train_loss:.4f}')
        print(f'The final test loss is: {final_test_loss:.4f}')
        print(f'The final train acc is: {final_train_acc:.4f} %')
        print(f'The final test acc is: {final_test_acc:.4f} %')
        print(f'The final test MAE is: {final_test_mae:.4f}')
        print(f'The final test MAPE is: {final_test_mape * 100:.4f} %')

        print('---END---' * 20)

        return

    # end time
    end_time = timer()

    # default value for training time (will be changed according to actual time, otherwise indication of a problem)
    total_time = 0

    # total time
    total_time = end_time - start_time

    # get the time it took to run in the correct form
    time_dict = convert_seconds(total_time)

    # print the total training time
    print(f"Total training time is: {time_dict['Hours']}:{time_dict['Minutes']}:{time_dict['Seconds']} (h:m:s)")

    if NUM_EPOCHS > epoch_save_threshold:

        # plot the loss and acc curves
        plot_loss_and_acc_curves(model_results=model_results)

        # saving results, weights and parameters
        save_model(state_dict=model_results['best_model'], model_results=model_results, time_dict=time_dict)

    print('---END---' * 20)


if __name__ == "__main__":
    main()
