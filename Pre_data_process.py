# Pre process for data that are stored in a single CSV file along with ONE folder of all the images
# The code will creat a folder '<name>_sorted' and sub folders (train, test, CSV) for the data and csv files, and arrange the data so it can be processed by Data_process.py
# Use just one folder as an input.

import os
from pathlib import Path
import pandas as pd
import shutil
import random

#############################################
# Change according to relevant folders
#############################################

folder_names = [
                './Org_data',
                # './Aug_data',
                # './Seg_data'
                ]

org_cvs_path = './Org_csv/all_images_4189.csv'

############################################

random.seed(42) # set a random seed for the random function. -> otherwise without using the OPTIONAL shutil.retree,
# -> the get_random_images function will generate different batches and the sub folders will get out of order

class Organize():

    def __init__(self, csv_path: str, # path to a single csv file (org_csv)
                images_folder_path: str, # path to single folder (org_data)
                file_column_idx: int = 0,
                different_desired_paths: bool = False,
                org_images_only: bool = True) -> None:
        
        self.csv_path = csv_path
        self.file_column_idx = file_column_idx
        self.original_df = pd.read_csv(self.csv_path)
        self.images_file_names = self.original_df.iloc[:, self.file_column_idx].tolist()
        self.images_folder_path = images_folder_path
        self.train_folder_path = None
        self.test_folder_path = None
        self.different_desired_paths = different_desired_paths
        self.org_images_only = org_images_only # decide if in the test file will be only org images

    def data_split(self, test_precentage=0.2):
        """"Spliting the data to train and test while creating the necessary directories and files"""
        # Getting the folder name (just name) and parent folder path (full path)
        self.parent_dir_name = Path(self.images_folder_path).name
        self.parent_dir_path = Path(Path(self.images_folder_path).parent)

        # number of folders of images to be include
        num_of_folders = len(folder_names)

        #########################################################################################################################
        # default path to create folders is by "self.parent_dir_path". If a different path is desired it will be specified here
        if self.different_desired_paths:
            # ENTER desired folder path
            self.parent_dir_path = Path('./Org_sorted')
            print(f'Created a new folder - {self.parent_dir_path.name} - based on more than 1 type of images')

        #########################################################################################################################

        # getting just the first word/s of the parent folder name - Seg or Org or Aug_Org etc'
        if num_of_folders == 1:
            init_name = self.parent_dir_name[:num_of_folders * 3]
        else:
            init_name = self.parent_dir_name[:num_of_folders * 3 + (num_of_folders - 1)]

        # folder path for 3 inner folders: train, test, CSV
        main_folder_name = f'{init_name}_sorted'
        self.main_folder_path = Path.joinpath(self.parent_dir_path, main_folder_name)
        self.check_and_create_folder(self.main_folder_path)
        # casting from str to Path
        self.main_folder_path = Path(self.main_folder_path)

        # Create path to csv sub folder
        self.parent_csv_path = Path.joinpath(self.main_folder_path, 'CSV')
        # Creates paths to train and test sub folders
        self.train_folder_path = Path.joinpath(Path(self.main_folder_path), 'train')
        self.test_folder_path = Path.joinpath(Path(self.main_folder_path), 'test')

        # # # OPTIONAL - Making sure the folders are empty by deleting the content of the given folders ->
        # # -> it gives the ability to recall the class and get a different split of the data with different images
        # if self.train_folder_path.exists() and self.test_folder_path.exists():
        #     shutil.rmtree(self.train_folder_path)
        #     shutil.rmtree(self.test_folder_path)
        #     shutil.rmtree(self.parent_csv_path)

        # Creating the folders train, test and csv
        self.check_and_create_folder(Path(self.train_folder_path))
        self.check_and_create_folder(Path(self.test_folder_path))
        self.check_and_create_folder(Path(self.parent_csv_path))

        # Aranging the data in new files and folders
        train_df = pd.DataFrame(columns=self.original_df.columns)
        test_df = pd.DataFrame(columns=self.original_df.columns)

        # Setting the data_split and getting splited paths accordingly
        divide_num = (((len(folder_names)-1) * test_precentage) + 1) # calculating the division number for the fomula
        self.calculated_total_images = len(self.images_file_names) / divide_num # in order to get a proper split, while not having seg or aug images of the test set in the train set, -->
        # --> the calc is based on the formula: x = test_split_precentage * (all images - (len(folder_names))*x + x) where x is the test set images.
        
        test_split = int(test_precentage * self.calculated_total_images) # 80% of data used for training set, 20% for testing 
        X_test_file_names = self.get_random_images(test_split)

        # getting the file names without the prefix
        X_test_file_names_no_prefix = []
        for file_name in X_test_file_names:
            if len(folder_names) == 1:
                X_test_file_names_no_prefix.append(file_name)
            else:    
                X_test_file_names_no_prefix.append(file_name[4:])

        # going through the file names in the images_file_names (which include ALL the images file names)
        for file_name in self.images_file_names:

            # file paths of images 
            file_path = Path.joinpath(Path(self.images_folder_path), file_name)

            if file_name in X_test_file_names:
                # copy files from parent folder to test folder
                shutil.copy(file_path, self.test_folder_path)
                # get the entire row for the matched name - for test data
                test_row = self.original_df.loc[self.original_df.iloc[:, self.file_column_idx] == file_name]
                test_df = pd.concat([test_df, test_row], ignore_index=True)

            elif file_name[4:] not in X_test_file_names_no_prefix:
                # copy files from parent folder to train folder
                shutil.copy(file_path, self.train_folder_path)
                # get the entire row for the matched name - for train data
                train_row = self.original_df.loc[self.original_df.iloc[:, self.file_column_idx] == file_name]
                train_df = pd.concat([train_df, train_row], ignore_index=True)

        train_df.to_csv(Path.joinpath(self.parent_csv_path, 'train_data.csv'), index=False)
        test_df.to_csv(Path.joinpath(self.parent_csv_path, 'test_data.csv'), index=False)

        return self.train_folder_path, self.test_folder_path


    def check_and_create_folder(self, folder_path_check, is_print = True):
        """If the folder doesn't exist, creat it"""
        if type(folder_path_check) == str:
                folder_path_check = Path(folder_path_check)
        if folder_path_check.is_dir():
            if is_print:
                print(f"{str(folder_path_check)} directory exists.")
        else:
            if is_print:
                print(f"Did not find {str(folder_path_check)} directory, creating one...")
            folder_path_check.mkdir(parents=True, exist_ok=True)
        return str(folder_path_check)


    # getting random images according to the specified number
    def get_random_images(self, num_images):
        # Ensure num_images is not greater than the total number of images
        num_images = min(num_images, len(self.images_file_names))
        # checking whether org images exist and will be used as the test images
        if self.is_org_file_exist(self.images_file_names):
            images_pool_list = self.get_only_org_images()
        else:
            images_pool_list = self.images_file_names
        # getting random files only from org images
        random_images_file_names = random.sample(population=images_pool_list, k=num_images)
        return random_images_file_names


    def is_org_file_exist(self, file_list):
        for file in file_list:
            if str(file).startswith('org'):
                return True
        return False


    # getting only original images to a new list
    def get_only_org_images(self):
        org_images_list = []
        for file_name in self.images_file_names:
            # specifing the str (that the file starts with) that is used for selection
            if str(file_name).startswith('org'):
                org_images_list.append(file_name)
        return org_images_list


    def check_split(self, lower_limit=0.79, upper_limit=0.81):
        """Check that the split was done correctly"""
        lower_limit_count = int(lower_limit * self.calculated_total_images)
        upper_limit_count = int(upper_limit * self.calculated_total_images)
        self.walk_through_dir(self.main_folder_path, prt=True)
        train_count = int(self.walk_through_dir(self.train_folder_path))
        if lower_limit_count <= train_count <= upper_limit_count:
            print('The data split has been done correctly')
        else:
            print('There is a problem in the data split')


    def walk_through_dir(self, walk_path, prt=False):
        """Walks through dir_path returning file counts of its contents."""
        for fold_path, dir, filenames in os.walk(walk_path):
            if prt:
                print(f"There are {len(dir)} directories and {len(filenames)} images in '{fold_path}'.")
        return len(filenames)


# ############################################################################################################################
# # path to the folder that contains all the images before the pre process is being executed
# images_folder_path = './Aug_Seg_Org_data'

# # path to the single csv file that contains all the data and will be used to divide the data into train and test csv files
# csv_path = './Aug_Seg_Org_csv/all_images_12567.csv'
# ###########################################################################################################################

pre_process = Organize(csv_path=org_cvs_path, images_folder_path=folder_names[0], different_desired_paths=False)
pre_process.data_split()
pre_process.check_split()

