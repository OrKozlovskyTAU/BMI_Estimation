import os
from pathlib import Path
from datetime import datetime
import pytz
import torch
import matplotlib.pyplot as plt
import torchvision.models as models
import math
import pandas as pd
import csv


def check_and_create_folder(folder_path, is_print = True):
    """If the folder doesn't exist, creat it"""
    if type(folder_path) == str:
            folder_path = Path(folder_path)
    if folder_path.is_dir():
        if is_print:
            print(f"{str(folder_path)} directory exists.")
    else:
        if is_print:
            print(f"Did not find {str(folder_path)} directory, creating one...")
        folder_path.mkdir(parents=True, exist_ok=True)
    return str(folder_path)


def time_stamp():
    '''Generate a timestamp string'''
    # Get the local time zone
    local_tz = pytz.timezone('Israel')

    # Get the current time in the local time zone
    local_time = datetime.now(local_tz)

    # Format the local time as a string
    return local_time.strftime("%d.%m.%Y_%H:%M:%S")


def convert_seconds(seconds):
    '''Calculate hours, minutes, and seconds'''
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_dict = {'Hours': int(hours),
                 'Minutes': int(minutes),
                 'Seconds': int(seconds)}
    return time_dict


def convert_to_classes(labels) -> list:
        """Convert BMI labels to categorical classes."""
        classes = []
        for label in labels:
            if label < 18.5:
                classes.append(0)  # Underweight
            elif label < 25:
                classes.append(1)  # Normal weight
            elif label < 30:
                classes.append(2)  # Overweight
            elif label < 40:
                classes.append(3)  # Obese
            else:
                classes.append(4)  # Severely Obese
        return classes


def to_grayscale(image):
    """
    input is (d,w,h)
    converts 3D image tensor to grayscale images corresponding to each channel
    """
    image = torch.sum(image, dim=0)
    image = torch.div(image, image.shape[0])
    return image


def layer_outputs(image, model, save_figure_path, model_name):
        # list of layer
        modulelist = list(model.features.modules())
        # output after the image was passed
        outputs = []
        # names of layers
        names = []
        prefixes = ['Sequential', 'Bottleneck', 'MBConv', 'Conv2dNormActivation', 'Squeeze', 'Adaptiv', 'Sigmoid', 'Stochastic']
        # path for saving results
        final_save_path = save_figure_path + 'layer_visualization/' + model_name
        # making sure the folder exists
        check_and_create_folder(final_save_path, is_print=False)

        for layer in modulelist[1:]:
            if any(str(layer).startswith(prefix) for prefix in prefixes):  # skipping those since they are not layers, rather blocks. Their inner layers are NOT skipped!
                 continue
            
            try:
                image = layer(image)
                outputs.append(image)
                names.append(str(layer))
            except RuntimeError as e:
                # print(f"Skipping layer: {str(layer)}. Reason: {e}") # printing the layer that was skipped
                continue  # Move to the next layer if there's an error

            # # Check if the layer is a Sequential block
            # if isinstance(layer, torch.nn.Sequential):
            #     # Inside Sequential block, handle individual layers
            #     for submodule in layer:
            #         # Check if the submodule is a Bottleneck block
            #         if isinstance(submodule, models.resnet.Bottleneck):
            #             # Inside Bottleneck block, handle individual layers
            #             bottleneck_layers = list(submodule.children())
            #             for bottleneck_layer in bottleneck_layers:
            #                 # Perform forward pass through the bottleneck_layer
            #                 try:
            #                     image = bottleneck_layer(image)
            #                 except RuntimeError as e:
            #                     print(f"Skipping layer: {str(bottleneck_layer)}. Reason: {e}")
            #                     break  # Move to the next layer if there's an error

            #                 # Append output and layer name
            #                 outputs.append(image)
            #                 names.append(str(bottleneck_layer))

            #                 # Check if the output is a tensor (ignore other types of modules)
            #                 if isinstance(image, torch.Tensor):
            #                     # Extract dimensions of the output tensor
            #                     batch_size, num_channels, height, width = image.shape

            #                     # Print layer name and output shape
            #                     print(f"Layer: {names[-1]}, Output Shape: ({num_channels}, {height}, {width})")


            # else:
            #     image = layer(image)
            #     outputs.append(image)
            #     names.append(str(layer))
            
        output_im = []
        for i in outputs:
            i = i.squeeze(0)
            temp = to_grayscale(i)  # take the mean
            output_im.append(temp.data.cpu().numpy())
            
        fig = plt.figure(figsize=(160, 100))

        for i in range(len(output_im)):
            a = fig.add_subplot(16, math.ceil(len(output_im)/16), i+1)
            imgplot = plt.imshow(output_im[i])
            a.set_axis_off()
            a.set_title(names[i].partition('(')[0], fontsize=10)
        plt.tight_layout()
        plt.savefig(final_save_path + f'/layers_output.jpg', bbox_inches='tight')



def find_overall_mean(first_list, second_list, mode=min):
     '''Finds the overall min between connected lists that have corresponding elements. For example: [1, 3] , [5, 2]. The function will return (3, 2) and not (1, 2) since the indexs are connected'''
     min_max_first_list = mode(first_list)
     min_max_first_list_index = first_list.index(min_max_first_list)
     second_by_first_index = second_list[min_max_first_list_index]

     min_max_second_list = mode(second_list)
     min_max_second_list_index = second_list.index(min_max_second_list)
     first_by_second_index = first_list[min_max_second_list_index]

     if min_max_first_list + second_by_first_index > min_max_second_list + first_by_second_index:
        min_max_first_list = first_by_second_index
     else:
        min_max_second_list = second_by_first_index

     return min_max_first_list, min_max_second_list


    # min_train_loss = min(class_reg_loss_results[i]['train_loss'])
    # min_train_loss_index = class_reg_loss_results[i]['train_loss'].index(min_train_loss)
    # test_loss_by_train_index = class_reg_loss_results[i]['test_loss'][min_train_loss_index]

    # min_test_loss = min(class_reg_loss_results[i]['test_loss'])
    # min_test_loss_index = class_reg_loss_results[i]['test_loss'].index(min_test_loss)
    # train_loss_by_test_index = class_reg_loss_results[i]['train_loss'][min_test_loss_index]

    # if min_train_loss + test_loss_by_train_index > min_test_loss + train_loss_by_test_index:
    #     min_train_loss = train_loss_by_test_index
    # else:
    #     min_test_loss = test_loss_by_train_index



def find_min_max_and_index(lst, mode=min):
    min_max_val = mode(lst)
    min_max_index = lst.index(min_max_val)
    return min_max_val, min_max_index


def clipping(lst, lower_lim, upper_lim):
    for i in range(len(lst)):
        if lst[i] < lower_lim:
            lst[i] = lower_lim
        elif lst[i] > upper_lim:
            lst[i] = upper_lim
    return lst


def delete_duplicate():

    # def remove_prefix(filename):
    #     prefix = ('aug_','org_')
    #     for pre in prefix:
    #         if filename.startswith(prefix):
    #             return filename[len(prefix)+2:]
    #     return filename

    # train_dir = './Aug_Org_sorted/train'
    # test_dir = './Aug_Org_sorted/test'

    # train_images = set(os.listdir(train_dir))
    # test_images = set(os.listdir(test_dir))

    # removed_count = 0
    # for filename in test_images:
    #     name = remove_prefix(filename)
    #     for train_im in train_images:
    #         if remove_prefix(train_im) == name:
    #             removed_count += 1
    #             os.remove(os.path.join(train_dir,train_im))
                
    # print(f"Removed {removed_count} images from train set.")

    # Set the path to your training directory and CSV file
    train_dir_path = './Aug_Org_sorted/train'
    csv_file_path = './Aug_Org_sorted/CSV/train_data.csv'
    new_csv_file_path = './Aug_Org_sorted/CSV/train_data_duplicate_delete.csv'

    # List all filenames in the training directory
    existing_filenames = set(os.listdir(train_dir_path))

    # Initialize a list to hold rows that should be kept
    rows_to_keep = []
    keep_count = 0

    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # Check if the filename in the first column exists in the training directory
            if row[0] in existing_filenames:
                rows_to_keep.append(row)
                keep_count += 1

    # Overwrite the original CSV file with rows that passed the check
    with open(new_csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows_to_keep)

    print("CSV file has been updated.")


