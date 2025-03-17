'''This code is designed to load the best state dict weights and then evaluate on the test set using those weights.'''

from pathlib import Path
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
from Data_process import get_data
import torch
import os
from torch import nn
import torch.utils.data
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from timeit import default_timer as timer
import argparse
import tqdm

# device agnostic code
device = device_select()

# Default Hyperparamaters / Values
SEED = 20

# parse command line arguments
parser = argparse.ArgumentParser(description='Train and test a model')
parser.add_argument(
    '--folder_path',
    type=str,
    default='./Org_Aug_Seg_sorted',
    help='Path to the folder containing the data',
)
parser.add_argument(
    '--model_path',
    type=str,
    help='Path to the model pth file',
)
args = parser.parse_args()

if not os.path.exists(args.folder_path):
    raise ValueError('Folder path does not exist')

if not os.path.exists(args.model_path):
    raise ValueError('Model path does not exist')

save_figure_path = './saved_figures/test/'
if not os.path.exists(save_figure_path):
    os.makedirs(save_figure_path)

try:
    # Load saved data
    loaded_data = torch.load(args.model_path, weights_only=False)
except Exception as e:
    print(f'Error loading model: {e}')
    raise

hyperparameters = loaded_data['Hyperparameters']
model = hyperparameters['Architecture']
ACTIVATION_FUNCTION = hyperparameters['Activation function']
DROP_PROB = hyperparameters['Drop-out Probability']
LOSS_FN = hyperparameters['Loss function']
IMG_SIZE = hyperparameters['Image size']
SEED = hyperparameters['Seed']
BATCH_SIZE = 32
NUM_WORKERS = 1

random_seed(seed=SEED)

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
loaded_model = models_cls_dict['EfficientNet_Model'](
    out_features=1,
    activation_fn=ACTIVATION_FUNCTION,
    drop_prob=DROP_PROB,
    classification=False,
).to(device)

# Loading the state dict
loaded_model.load_state_dict(loaded_data['Model_state_dict'])

_, test_data_custom, _, test_dataloader, _, _ = get_data(
    folder_path=Path(args.folder_path),
    save_figure_path=save_figure_path,
    classification_activation=False,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)


# Test function
def test(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, tolerance: int = 3):

    # test_loss
    test_loss = 0
    test_acc = 0

    y_list, y_pred_list = [], []
    # eval mode
    model.eval()
    with torch.inference_mode():
        for X_test, y_test, _ in tqdm.tqdm(dataloader, desc='Testing'):
            # send data to device
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_list.append(y_test.item())

            # forward pass
            y_test_pred = model(X_test)
            y_test_pred = y_test_pred.squeeze(dim=1)
            y_pred_list.append(y_test_pred.item())

            # calculate the loss
            loss = loss_fn(y_test_pred, y_test)
            test_loss += loss

            is_within_threshold = torch.abs(y_test_pred - y_test) <= tolerance
            acc_calc = torch.sum(is_within_threshold)
            test_acc += acc_calc

        # Loss and accuracy normalization according to the length of the dataloader
        test_acc = float(test_acc)
        test_loss /= len(dataloader)
        test_acc /= len(test_data_custom)

    MAE = mean_absolute_error(y_list, y_pred_list)  ###
    MAPE = mean_absolute_percentage_error(y_list, y_pred_list)  ###
    print('-' * 100)
    print(
        'Test MAE: {:.4f}\t Test MAPE: {:.4f} %\t Test loss: {:.4f}\t Test accuracy: {:.4f} %'.format(
            MAE, MAPE * 100, test_loss, test_acc * 100
        )
    )  ###
    print('-' * 100)


# set timer to measure how long it takes to train
start_time = timer()

# training the model
test(model=loaded_model, dataloader=test_dataloader, loss_fn=LOSS_FN)

# end time
end_time = timer()

# default value for training time (will be changed according to actual time, otherwise indication of a problem)
total_time = 0

# total time
total_time = end_time - start_time

print(
    f'Total test time for {len(test_data_custom)} images is: {total_time:.4f} seconds.\
      \nTherefor a single image took {total_time / len(test_data_custom):.4f} seconds on average.'
)
