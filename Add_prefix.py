
import os

#####################################################################################
# adding a fixed prefix to all the files names in a specified folder
#####################################################################################

dest_folder = './Org_sorted/test'
prefix_to_add = 'org_'

for file in os.listdir(dest_folder):
    old_file_path = os.path.join(dest_folder, file)
    new_file_path = os.path.join(dest_folder, prefix_to_add + os.path.basename(old_file_path))

    os.rename(old_file_path, new_file_path)

#####################################################################################

#####################################################################################
# adding a fixed prefix to the csv file to the start of each line except the header
#####################################################################################

# Path to the CSV file
dest_file = './Org_csv/all_images_4189_test.csv'
# prefix to be added to the beginning of the line
prefix_to_add = 'org_'

# Add '{prefix_to_add}' to all the rows in the file, except the header
prefixes = ['aug', 'org', 'seg']
with open(dest_file, 'r+') as f:
    lines = f.readlines() # reading the lines in the file
    f.seek(0) # making sure we are starting at the beginning of the file
    for i, line in enumerate(lines):
        if line[:3] not in prefixes and i != 0: # check that non of the prefixes already exist and skipping the header as well
            line = prefix_to_add + line
        f.writelines(line) # writing each line after it has been modified

#####################################################################################
        

