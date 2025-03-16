'''This file calculate a histogram of all the given samples'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import check_and_create_folder, time_stamp, convert_to_classes
import wandb

save_path = './saved_figures/histograms'


# rounding up a number to the closest 2 digit whole number  - 10, 20, ...
def round_num(num_to_round: int, up: bool):
    counter = 0
    while num_to_round > 10:
        num_to_round /= 10
        counter += 1
    if up:
        num_to_round = np.ceil(num_to_round)
    else:
        num_to_round = np.floor(num_to_round)
    if counter > 0:
        num_to_round *= counter * 10
    return num_to_round


# Calculate errors list
def make_error_list(
    targ_list: list, pred_list: list, num_classes: int = 5, classification_activation: bool = False, tolerance: int = 1
):

    # Initializing empty lists
    bin_error_list = []
    error_by_class = [
        [] for _ in range(num_classes)
    ]  # list of list of errors based on prediction - for all error of class 0, what are the predicted classes. and so on.

    # calc the error
    absolute_error_array = np.abs(np.array(targ_list) - np.array(pred_list))

    # Identify indices where errors exceed the tolerance
    error_indices = np.where(absolute_error_array >= tolerance)[0]

    # Appending the values of the targ_list (that have exceeded the threshold) to a new list
    for i in range(len(error_indices) - 1):
        bin_error_list.append(targ_list[error_indices[i]])

    if classification_activation:
        for i in range(len(error_indices) - 1):
            error_by_class[targ_list[error_indices[i]]].append(pred_list[error_indices[i]])

    return bin_error_list, error_by_class


# making histogram from the data
def make_histogram(
    save_path: str = save_path,
    additional_path: str = '',
    csv_file_path: str = '',
    data_list: list = [],
    hist_title: str = '',
    x_label: str = '',
    y_label: str = '',
    num_classes: int = 5,
    is_csv: bool = False,
    calc_bin: bool = False,
    normal_hist: bool = True,
    classification_activation: bool = False,
    current_time: str = time_stamp(),
    default_time_stamp: str = '',
    class_idx: int = 0,
):

    # create folder if not already exist
    if additional_path:
        folder_save_path = check_and_create_folder(
            f'{save_path}/hist_{additional_path}_{default_time_stamp}/hist_class_{class_idx}', is_print=False
        )
    else:
        folder_save_path = check_and_create_folder(f'{save_path}/hist_{current_time}', is_print=False)

    # getting the required data
    if is_csv:
        data = pd.read_csv(csv_file_path)
        data_values = data.iloc[:, 1]  #
        print(len(data_values))
        if classification_activation:
            data_values = convert_to_classes(data_values)
    else:
        data_values = data_list

    # make sure data is not empty
    if len(data_values) == 0:
        print('Data list is empty - possible error (might not be an error for some classes)')
        return

    if classification_activation:
        num_bins = num_classes
        min_bin = -0.5
        max_bin = num_classes - 0.5
        bin_width = 1
    elif calc_bin:
        # calculating the max and min values to determine the num of beans
        max_data = max(data_values)
        min_data = min(data_values)
        max_bin = round_num(max_data, up=True)
        min_bin = round_num(min_data, up=False)

        # calculating the number of bins needed
        num_bins = int(np.ceil((max_bin - min_bin) / 2))
    else:
        # default values - meaning each bin is of size 2
        min_bin = 10
        max_bin = 80
        num_bins = 70
        bin_width = (max_bin - min_bin) / num_bins

    # plotting the histogram
    plt.figure(figsize=(16, 10))
    if normal_hist:  # normal hist is for making a histogram out of the values in the data
        hist_values, bin_edges, patches = plt.hist(
            x=data_values, range=(min_bin, max_bin), bins=num_bins, edgecolor='black'
        )
    else:  # for making a bar graph
        bin_edges = np.arange(min_bin, max_bin + bin_width, bin_width)
        plt.bar(x=bin_edges[:-1], height=data_values, width=bin_width, edgecolor='black', align='edge')
    ticks_font_size = 16
    labels_font_size = 20
    title_font_size = 30
    plt.title(hist_title, fontsize=title_font_size)
    plt.xlabel(x_label, fontsize=labels_font_size)
    plt.ylabel(y_label, fontsize=labels_font_size)
    if classification_activation:
        plt.xticks(range(num_bins), range(num_bins))
    plt.xticks(fontsize=ticks_font_size)
    plt.yticks(fontsize=ticks_font_size)
    plt.grid()
    plt.show()
    plt.savefig(f'{folder_save_path}/{hist_title}')
    print(f'{hist_title} saved at {folder_save_path}/{hist_title}')
    plt.close()  # close to save memory

    # wandb.log({f'{hist_title}': wandb.Image(f'{folder_save_path}/{hist_title}.png')})
    if normal_hist:
        return hist_values, bin_edges, patches


##########################################################################################################
# # Uncomment if you want to make a histogram of the given data based on the folder
# csv_file_path, title = './Org_sorted/CSV/all_data.csv', " | Original Data"
csv_file_path, title = './Org_Aug_Seg_sorted/CSV/all_data.csv', ""

# # make histogram of the BMI data - REGRESSION
make_histogram(
    is_csv=True,
    csv_file_path=csv_file_path,
    save_path=save_path,
    hist_title=f'Histogram of BMI values{title}',
    x_label='BMI values',
    y_label='Quantity',
)

# # make histogram of the BMI data - CLASSIFICATION
make_histogram(
    is_csv=True,
    csv_file_path=csv_file_path,
    save_path=save_path,
    hist_title=f'Histogram of BMI classes{title}',
    x_label='BMI classes',
    y_label='Quantity',
    classification_activation=True,
)
##########################################################################################################
