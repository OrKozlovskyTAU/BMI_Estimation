'''The code is taking a pth file, extract all the data and hyperparameters, and write it to a csv file.
Since there are multiple files with different headers, the code handles that by setting a predetermined header
(based on the most updated one) and also leaves a blank cell if the header parameter is not present in the data collected from the specific file'''

import torch
import csv
import os

# Define the directory containing the .pth files
pth_directory = './saved_pth/'

# Define the CSV file path where you want to save the data
csv_file = os.path.join(pth_directory, 'manual_data_from_pth.csv')

# define the header to be used for the csv file
headers = ['File', 'Folder name', 'Activation function', 'Architecture', 'Loss function', 'Learning rate',
                     'Optimizer', 'Drop-out Probability', 'Scheduler type', 'Scheduler repetitions',
                     'Scheduler patience', 'Scheduler factor', 'Current number of epochs', 'Total number of epochs',
                     'Train transforms', 'Test transforms', 'Check point', 'Train images', 'Test images',
                     'Total images', 'Divided to classes (classification)', 'Min_MAE', 'Min_MAPE', 'Elapsed time [h:m:s]',
                     'Train loss (min)', 'Test loss (min)', 'Train accuracy % (max)', 'Test accuracy % (max)']

# Open CSV file in write mode
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=headers)

    # Write the header row
    writer.writeheader()

    # Iterate over each .pth file in the directory
    for filename in os.listdir(pth_directory):
        if filename.endswith('.pth'):

            filename_dict = {headers[0]: filename}
            # extracted_params = extract_params_from_file(filename)

            pth_file = os.path.join(pth_directory, filename)

            # Load the data from the .pth file
            file_data = torch.load(pth_file)
        
            # Extract the required information
            hyperparameters = file_data['Hyperparameters']
            results = file_data['Results']

            # concatenating dictionaries
            extracted_params = {**hyperparameters, **results}
            
            # concatenating dictionaries
            all_data = {**filename_dict, **extracted_params}

            # new dictionary for updated keys
            updated_keys_mapping = {
                'Resnet': 'Architecture',
                'Elapsed time': 'Elapsed time [h:m:s]',
                'Train accuracy (max)': 'Train accuracy % (max)',
                'Test accuracy (max)': 'Test accuracy % (max)',
                'Divided to classes': 'Divided to classes (classification)'
            }

            # We iterate over each key-value pair in all_data.items(). In each iteration, key represents a key from the all_data dictionary, and value represents the corresponding value.
            # For each key from all_data, we use it to search for a corresponding updated key in the updated_keys_mapping dictionary.
            # If the key from all_data is found in updated_keys_mapping, get() returns the corresponding updated key (value) from updated_keys_mapping.
            # If the key from all_data is not found in updated_keys_mapping, get() returns the original key itself.
            all_data_updated = {updated_keys_mapping.get(key, key): value for key, value in all_data.items()}

            # write the data if the header exists, otherwise write ''
            row_data = {}
            for header in headers:
                if header in all_data_updated:
                    row_data[header] = all_data_updated[header]
                else:
                    row_data[header] = ''
            
            # Writing the data after all the processing
            writer.writerow(row_data)