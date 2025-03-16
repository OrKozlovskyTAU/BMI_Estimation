'''This file is pre-pre-data process. It takes all the data (images) and makes a single csv file that is filtered.
Also copying all the data to one folder that contains all the images. Essentially preparing the data in a manner that can be processed by Pre_data_process.py'''

import os
from torch.utils.data import Dataset
import pandas as pd
import re
import shutil


class Dataset(Dataset):
    def __init__(self, folder_path):
        self.Pic_Names = os.listdir(folder_path)
        self.folder_path = folder_path

    def __len__(self):
        return len(self.Pic_Names)

    def __getitem__(self, idx):
        img_name = self.Pic_Names[idx]  # getting the image name
        pattern = r"^.+?_\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+"  # pattern in the data to search by it
        ret = re.match(pattern, img_name)  # finding pattern in the file name for gathering data
        BMI = (int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2  # calculating the BMI
        height = int(ret.group(3)) / 100000  # getting the height
        weight = int(ret.group(4)) / 100000  # getting the weight
        age = ret.group(2)  # getting the age
        sex = ret.group(1)  # getting the sex
        sex = sex.upper()  # make letters capital letters
        # Pic_name = os.path.join(self.folder_path, img_name)
        return img_name, BMI, height, weight, age, sex


#############################################
# Change according to relevant folder
#############################################

folder_names = ['./Org_sorted', './Aug_data', './Seg_data']

############################################

# forming the prefix of the saved path
init_list = []
prefix_name_for_folder = ''
for i in range(len(folder_names)):
    init_list.append((folder_names[i])[2:5])
    prefix_name_for_folder += f'{init_list[i]}_'

# master folder name - for example - Org_Aug_sorted
master_folder_path = os.path.join('./', f'{prefix_name_for_folder}sorted')

if not os.path.exists(master_folder_path):
    os.mkdir(master_folder_path)

# paths to save the single csv file and the data (images)
save_path_dest_csv = os.path.join(master_folder_path, 'CSV')
train_path_dest_folder = os.path.join(master_folder_path, 'train')
test_path_dest_folder = os.path.join(master_folder_path, 'test')


# Initializing a list for the data
data_list = []

# Check if the destination folder exist and if not, creates it
if not os.path.exists(train_path_dest_folder):
    os.mkdir(train_path_dest_folder)
    train_copy_folder_flag = True
else:
    train_copy_folder_flag = False
    if len(os.listdir(train_path_dest_folder)) == 0:  # check if the destination folder for images copy is empty
        train_copy_folder_flag = True

# copy all the train images from org
train_org_folder = './Org_sorted/train'
shutil.copytree(train_org_folder, train_path_dest_folder, dirs_exist_ok=True)

# copy all the test images from org to a new test folder
test_org_folder = './Org_sorted/test'
shutil.copytree(test_org_folder, test_path_dest_folder, dirs_exist_ok=True)

# copy csv (train and test) files to a new csv folder
org_csv_folder = './Org_sorted/CSV'
shutil.copytree(org_csv_folder, save_path_dest_csv, dirs_exist_ok=True)

# path to the destination csv file
dest_csv_train_file = os.path.join(save_path_dest_csv, 'train_data.csv')

# copy all the other images (seg or aug) to the new train folder
for folder in folder_names:
    if folder[2:5] == 'Org':
        continue

    shutil.copytree(folder, train_path_dest_folder, dirs_exist_ok=True)

    # # delete images that are also in the test folder
    # for file in os.listdir(test_path_dest_folder):
    #     if os.path.exists(os.path.join(train_path_dest_folder, file)):
    #         os.remove(os.path.join(train_path_dest_folder, file))

    # Path to CSV file of that category
    category_csv_file = f'{folder[2:5]}_csv/all_images.csv'

    # Open the category CSV file in read mode
    with open(category_csv_file, 'r') as cat_file:
        # Read all lines from the category CSV file, excluding the header
        cat_lines = cat_file.readlines()[1:]

    # Open the original CSV file in append mode
    with open(dest_csv_train_file, 'a') as org_file:
        # Write the lines from the category CSV file to the original CSV file
        org_file.writelines(cat_lines)

print("CSV files concatenated successfully.")


# # Going through each path in the folders list
# for folder in folder_names:
#     if train_copy_folder_flag:  # if the folder already exist there is no need to copy anything
#         # Copy the contents of the source folder to the destination folder
#         shutil.copytree(folder, train_path_dest_folder, dirs_exist_ok=True)

#     # Getting the dataset
#     all_images_data_set = Dataset(folder)

#     # Getting the parameters of the files and put them in a list
#     for i in range(len(os.listdir(folder))):
#         Pic_name, BMI, height, weight, age, sex = Dataset.__getitem__(self=all_images_data_set, idx=i)
#         data_list.append([Pic_name, BMI, height, weight, age, sex])

# # Create the DataFrame once after collecting all the data
# all_images_df = pd.DataFrame(data_list, columns=['Pic_name', 'BMI', 'height', 'weight', 'age', 'sex'])
# # print(all_images_df)

# # Check if the destination folder exist and if not, creates it
# if not os.path.exists(os.path.dirname(save_path_csv)):
#     os.mkdir(os.path.dirname(save_path_csv))

# # Saving the data frame to a csv file
# all_images_df.to_csv(save_path_csv, index=False)


#######################################################################

# # TEST
# # Check if the destination folder exist and if not, creates it
# if not os.path.exists(test_path_dest_folder):
#     os.mkdir(test_path_dest_folder)
#     test_copy_folder_flag = True
# else:
#     test_copy_folder_flag = False
#     if len(os.listdir(test_path_dest_folder)) == 0: # check if the destination folder for images copy is empty
#         test_copy_folder_flag = True
