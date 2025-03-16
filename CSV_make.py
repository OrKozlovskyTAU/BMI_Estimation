'''This file is pre-pre-data process. It takes all the data (images) and makes a single csv file.
Also amending all the data to have a prefix. Essentially preparing the data in a manner that can be processed by Pre_data_process.py'''

import os
from torch.utils.data import Dataset
import pandas as pd
import re


class ImageDataFetcher(Dataset):
    def __init__(self, folder_path):
        self.file_names = os.listdir(folder_path)
        self.folder_path = folder_path

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        return self.process_file(file_name)

    def process_file(self, file_name):
        pattern = r"^.+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+"  # pattern in the data to search by it
        match = re.match(pattern, file_name)  # finding pattern in the file name for gathering data
        if not match:  # raise an error if there is a match problem
            raise ValueError(f"Invalid file name: {file_name}")

        sex = match.group(1).upper()  # getting the sex, make letters capital letters
        age = match.group(2)  # getting the age
        height = int(match.group(3)) / 100000  # getting the height
        weight = int(match.group(4)) / 100000  # getting the weight
        BMI = weight / (height**2)  # calculating the BMI

        return {'File_name': file_name, 'BMI': BMI, 'Height': height, 'Weight': weight, 'Age': age, 'Sex': sex}


#############################################
# Change according to relevant folder
#############################################

folder_path = './Seg_data'
prefix_to_add = os.path.basename(folder_path).split('_')[0].lower()
# prefixes = ['org', 'aug', 'seg']

############################################

# count the number of images
num_of_images = len(os.listdir(folder_path))

# # add prefix to each file name in data folder
# for file in os.listdir(folder_path):
#     if file[:3] in prefixes:
#         raise ValueError(f"one of the files contains a prefix: {file}")
#     old_path = os.path.join(folder_path, file)
#     new_filename = prefix_to_add + '_' + file
#     new_path = os.path.join(folder_path, new_filename)
#     os.rename(old_path, new_path)

# Initializing a list for the data
data_list = []

# Getting the dataset
all_images_data_set = ImageDataFetcher(folder_path)

# Getting the parameters of the files and put them in a list
for index in range(len(all_images_data_set)):
    # getting the data from the file after slicing
    file_data = all_images_data_set[index]
    # appending to the list using the values of the returned dictionary
    data_list.append(list(file_data.values()))

# Create the DataFrame once after collecting all the data
all_images_df = pd.DataFrame(data_list, columns=all_images_data_set[0].keys())
# file_names_col = all_images_df.columns[0]
# all_images_df[file_names_col] = prefix_to_add + '_' + all_images_df[file_names_col]

# path to save csv file
save_path_csv = f'{prefix_to_add.capitalize()}_csv'

# Check if the destination folder exist and if not, creates it
if not os.path.exists(save_path_csv):
    os.mkdir(save_path_csv)

# Saving the data frame to a csv file
all_images_df.to_csv(os.path.join(save_path_csv, f'all_images.csv'), index=False)
