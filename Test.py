'''This code is designed to load the best state dict weights and then evaluate on the test set using those weights.'''


from Model import ResNet_Model, EfficientNet_Model, MobileNet_Model, VitTransformer_Model, CVT_Transformer_Model, DenseNet_Model, RegNet_Model
from Device_and_Seed import device_select
from Data_process import *
from Histogram import *
import torch
from torch import nn
import torch.utils.data
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from timeit import default_timer as timer


# device agnostic code
device = device_select()

# Default Hyperparamaters / Values
SEED = 20
ACTIVATION_FUNCTION = nn.GELU()
RESNET_NUM = 'resnet101'
LOSS_FN = nn.MSELoss()
DROP_PROB = 0
tolerance = 3

# path to the saved model pth file
saved_pth_path = 'model_save_Org_sorted'

# path to save results along with weights and parameters
results_path = './saved_pth/'

# file save path
file_load_path = f'{results_path}{saved_pth_path}'

# Load saved data
loaded_data = torch.load(file_load_path)

# Load model weights only
loaded_state_dict = loaded_data['Model_state_dict']

# instantiating the model
loaded_model = EfficientNet_Model(out_features=1,
                                    activation_fn=ACTIVATION_FUNCTION,
                                    drop_prob=DROP_PROB,
                                    classification=classification_activation).to(device)

# Loading the state dict
loaded_model.load_state_dict(loaded_state_dict)


# Test function
def test(model: nn.Module,
         dataloader: torch.utils.data.DataLoader,
         loss_fn: nn.Module,
         tolerance : int = tolerance):
    
    # test_loss
    test_loss = 0
    test_acc = 0

    # eval mode
    model.eval()
    with torch.inference_mode():
        for batch, (X_test, y_test) in enumerate(dataloader):

            # casting BMI from double to float
            y_test = y_test.type(torch.float32)

            # send data to device
            X_test, y_test = X_test.to(device), y_test.to(device)

            # forward pass
            y_test_pred = model(X_test)
            # y_test_pred = torch.softmax(y_test_pred_logits)
            y_test_pred = y_test_pred.squeeze(dim=1)

            #calculate the loss
            loss = loss_fn(y_test_pred, y_test)
            test_loss += loss

            is_within_threshold = torch.abs(y_test_pred - y_test) <= tolerance
            acc_calc = torch.sum(is_within_threshold)
            test_acc += acc_calc

        # Loss and accuracy normalization according to the length of the dataloader
        test_acc = float(test_acc)
        test_loss /= len(dataloader)
        test_acc /= len(test_data_custom)

    MAE = mean_absolute_error(y_test.to('cpu'), y_test_pred.to('cpu')) ###
    MAPE = mean_absolute_percentage_error(y_test.to('cpu'), y_test_pred.to('cpu')) ###
    print ('-' * 80)
    print ('-' * 80)
    print('Test MAE: {:.4f}\t Test MAPE: {:.4f} %\t Test loss: {:.4f}\t Test accuracy: {:.4f} %'.format(MAE, MAPE*100, test_loss, test_acc*100)) ###


# set timer to measure how long it takes to train
start_time = timer()

# training the model
test(model=loaded_model,
     dataloader=test_dataloader,
     loss_fn=LOSS_FN)

# end time
end_time = timer()

# default value for training time (will be changed according to actual time, otherwise indication of a problem)
total_time = 0

# total time
total_time = end_time - start_time

print(f'Total test time for {len(test_data_custom)} images is: {total_time} seconds.\
      \nTherefor a single image took {total_time / len(test_data_custom)} seconds on average.')

