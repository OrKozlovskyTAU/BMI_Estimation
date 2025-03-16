# Become one with the data - this code initialize the datasets and dataloaders as well as ->
# -> defining the transforms and have a function to display the images
import os
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Device_and_Seed import random_seed
import random
import math
from torchvision.transforms.functional import InterpolationMode
from typing import List, Tuple
from utils import check_and_create_folder


#############################################################################################################################
# defining the classes thesholds
# default_class_separators = np.array([18.5])
# default_class_separators = np.array([18.5, 25, 30 ,40])
default_class_separators = np.array([18.5, 22, 25, 27.5, 30, 33.5, 37, 40])
#############################################################################################################################


###########################################################################################################################
# OLD PROJECT FUNCTIONS - but they have been changed
class Resize(transforms.Resize):

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__(size, interpolation, max_size, antialias)
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def _get_image_size(self, img):
        if transforms.functional._is_pil_image(img):
            return img.size
        elif isinstance(img, torch.Tensor) and img.dim() > 2:
            return img.shape[-2:]  # getting the last 2 shapes
        else:
            raise TypeError("Unexpected type {}".format(type(img)))

    def __call__(self, img):
        h, w = self._get_image_size(img)
        aspect_ratio = w / h
        new_w = self.size
        new_h = int(new_w / aspect_ratio)
        # print(f"After Resize - Height: {new_h}, Width: {new_w}")
        return transforms.functional.resize(img, (new_w, new_h), self.interpolation)


###########################################################################################################################


# selecting a random seed to see progress in testing
random_seed(20)


class BMIdataset(Dataset):
    """This class handles a single csv file that contain at least the data file names (images path) and the corresponding labels.
    In addition, it assumes that the actual images (all of them) are stored in a different folder. It then gets all the data
    from the folder and the csv file and split them into train and test set along with two new csv files - one for the train
    data and one for the test data. In addition, it can load an image."""

    def __init__(
        self,
        csv_path: str,
        images_folder_path: str,
        transform: transforms.Compose,
        class_separators: np.array = None,
        classification: bool = False,
        image_list: List[str] = None,
    ) -> None:

        # get the data from the csv file
        self.csv_data = pd.read_csv(csv_path)
        # the folder that contains all the images
        self.images_folder_path = images_folder_path
        # transformations on the images
        self.transform = transform
        # if it's a classification problem
        self.classification = classification
        # if class boundaries in passed to the class it will be used, otherwise there is a default
        self.class_separators = class_separators if class_separators is not None else default_class_separators

        default_false_number = 0
        # image list is a specific list of images - use when only a some images are required and not all the data
        if image_list is not None:
            self.image_list = image_list
            self.images_file_names = self.image_list
            # Retrieve corresponding BMI labels from the CSV file based on the image names
            self.BMI_labels = []
            for image_name in self.image_list:
                # check if the first letter is 'f' (which is a flag for an image that its label is outside the class limits)
                if image_name.startswith(
                    'f'
                ):  # f prefix for false images that are not belong to the class (they are there on purpose)
                    self.BMI_labels.append(
                        default_false_number
                    )  # set a number to serve as the label for all the images that are outside the class
                    continue
                # Find the row in the CSV file where the image name matches
                row = self.csv_data[self.csv_data.iloc[:, 0] == image_name]
                if not row.empty:
                    # If the row is found, extract the BMI label from column 1 - the first element is 0 since we only have 1 row
                    bmi_label = row.iloc[0, 1]  # Assuming BMI label is in the second column (index 1)
                    self.BMI_labels.append(bmi_label)
                else:
                    # If the image name is not found in the CSV file, handle the error accordingly
                    raise ValueError(f"Image name {image_name} not found in the CSV file.")
            # # getting the minimum value of each class and changing the false image values to 5 units less than the min value
            # min_val = min(self.BMI_labels)
            # for i, bmi in enumerate(self.BMI_labels):
            #     if bmi == default_false_number:
            #         self.BMI_labels[i] = min_val - 5

        else:
            # If image_list is not provided, use all images from the CSV file
            self.images_file_names = self.csv_data.iloc[:, 0].tolist()
            if classification:
                # get BMI labels and convert BMI to classes
                self.BMI_labels = self.convert_labels_to_classes(
                    self.csv_data.iloc[:, 1].tolist(), self.class_separators
                )
                # # get the indexs of the images that divided to classes
                # self.images_file_names_classes_lists, self.images_file_names_classes_lists_indexes = self.convert_image_names_to_classes(self.images_file_names, self.BMI_labels)
            else:
                # get all BMI labels from csv
                self.BMI_labels = self.csv_data.iloc[:, 1].tolist()

    def convert_labels_to_classes(self, labels, class_separators):
        """Convert BMI labels to categorical classes. labels are in the form of actual numbers and not classes"""

        # casting to numpy array
        labels_np = np.array(labels)

        # Find the classes for each label
        labels = np.searchsorted(class_separators, labels_np)

        # cast to list
        labels = labels.tolist()

        # for label in labels:

        # # regular division of BMI
        # IF YOU UNCOMMENT THIS - FINAL LAYER OF FC NEEDS TO BE ADJUSTED AS WELL
        # if label < 18.5:
        #     classes.append(1)  # Underweight
        # elif label < 25:
        #     classes.append(2)  # Normal weight
        # elif label < 30:
        #     classes.append(3)  # Overweight
        # elif label < 40:
        #     classes.append(4)  # Obese
        # else:
        #     classes.append(5)  # Severely Obese

        # number of different classes
        self.num_of_classes = len(set(labels))

        return labels

    @staticmethod
    def convert_image_names_to_classes(
        images_file_names: list, labels: list, add_false: bool = False
    ) -> Tuple[list, list]:
        """Convert BMI images file names and their indexes to categorical classes. images_file_names MUST be in the same order as the labels.
        labels is a list of sorted labels. for example - [4, 1, 5, 2, 4, 0].
        Returns the lists of the file names and indexes after sorting."""
        print(f"Number of classes: {len(set(labels))}")
        print(f"Number of images: {len(images_file_names)}")
        print(f"Number of labels: {len(labels)}")
        images_file_names_classes = [[] for _ in range(len(set(labels)))]  # list of lists
        images_file_names_classes_indexes = [[] for _ in range(len(set(labels)))]  # list of lists
        for i, file in enumerate(images_file_names):
            # getting the value of a label, for example 2, which defines to which inner list, the NAME of the image should be appended to
            images_file_names_classes[labels[i]].append(file)
            # getting the value of a label, for example 2, which defines to which inner list, the INDEX of the image should be appended to
            images_file_names_classes_indexes[labels[i]].append(i)

        for i in range(len(images_file_names_classes)):
            assert len(images_file_names_classes[i]) != 0, "There is a problem - one of the classes list is empty!"
            assert (
                len(images_file_names_classes_indexes[i]) != 0
            ), "There is a problem - one of the classes list is empty!"

        if (
            add_false
        ):  # the entire 'if', is applicable when doing manual_classes and we want to add labels that don't belong to the class
            temp_images_file_names_classes = [[] for _ in range(len(set(labels)))]  # list of lists
            for i in range(len(images_file_names_classes)):
                # get length of the current class
                temp_len = len(images_file_names_classes[i])
                # all other classes images
                other_lists = images_file_names_classes[:i] + images_file_names_classes[i + 1 :]
                temp_images_list = [name for sublist in other_lists for name in sublist]
                # get random samples
                print(f"Number of images in class {i}: {temp_len}")
                print(f"Number of images in other classes: {len(temp_images_list)}")
                random_samples_images = random.sample(temp_images_list, k=temp_len)
                random_samples_images = ['f' + name for name in random_samples_images]
                temp_images_file_names_classes[i] = random_samples_images

            # concat original list with the false labels list
            for i in range(len(images_file_names_classes)):
                images_file_names_classes[i].extend(temp_images_file_names_classes[i])

        return images_file_names_classes, images_file_names_classes_indexes

    def load_image(self, index: int) -> Tuple[Image.Image, str]:
        "Opens an image via a path and returns it and it's path."

        # check if image name start with 'f' and delete 'f' if present (used when adding labels not belong to a class)
        if self.images_file_names[index].startswith('f'):
            self.images_file_names[index] = self.images_file_names[index][1:]
            # making sure there are no 'f' left in the beginning of the name
            assert not self.images_file_names[index].startswith('f'), "STOP! - there is a problem with the file names"

        # uncomment if using regular data
        self.image_path = Path.joinpath(self.images_folder_path, self.images_file_names[index])

        # uncomment if using data with special prefix
        # image_path = Path.joinpath(Path(self.images_folder_path), 'seg_' + os.path.basename(self.images_file_names[index]))

        return Image.open(self.image_path), self.image_path

    # def image_path(self, index: int) -> str:
    #     "Get image path based on index."
    #     self.image_path = Path.joinpath(self.images_folder_path, self.images_file_names[index])
    #     return self.image_path

    def __len__(self) -> int:
        return len(self.images_file_names)

    def __getitem__(self, idx) -> Tuple[Image.Image, int]:
        img, img_path = self.load_image(index=idx)
        # img_path = self.image_path(index=idx)[1]
        bmi_idx = self.images_file_names.index(os.path.basename(img_path))
        BMI_label = self.BMI_labels[bmi_idx]

        # # Print original image size
        # print(f"Original Image Size: {img.size}")

        # Transforms
        if self.transform:
            img = self.transform(img)

        # # Print transformed image size
        # print(f"Transformed Image Size: {img.size()}")

        if self.classification:
            # casting from float64 to float32
            BMI_label = torch.tensor(BMI_label, dtype=torch.long)
        else:
            # casting from float64 to float32
            BMI_label = torch.tensor(BMI_label, dtype=torch.float32)

        return img, BMI_label, self.images_file_names[idx]


# Function to display images and visualize the data
def display_random_images(dataset: Dataset, save_path: str, num_of_images: int = 6):

    # Adjust display if n too high
    if num_of_images > 9:
        num_of_images = 9
        print(f"For display purposes, n shouldn't be larger than 9, setting to 9.")

    # Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=num_of_images)

    # Calculate the number of rows and columns for subplots
    ncols = min(num_of_images, 3)
    nrows = math.ceil(num_of_images / ncols)

    # Setup plot
    fig1, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12))

    # Loop through samples and display random samples
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # Adjust image tensor shape for plotting: [color_channels, height, width] -> [height, width, color_channels]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Flatten the axes array to simplify indexing
        axes = axes.flatten()

        # Plot adjusted samples
        ax = axes[i] if i < len(axes) else None

        if ax:
            ax.imshow(targ_image_adjust)
            ax.axis("off")
            title = f"BMI label: {targ_label:.3f}\n Shape: {targ_image_adjust.shape}"
            ax.set_title(title)

        # plt.subplots_adjust(hspace=5)
        # plt.imshow(targ_image_adjust)
        # plt.axis("off")
        # title = f"BMI label: {targ_label:.3f}\n Shape: {targ_image_adjust.shape}"
        # plt.title(title)
        # plt.savefig('Example.jpg')

    # Saving figure
    fig1.savefig(save_path + 'Train image examples.jpg')

    # Adjust layout to prevent clipping of titles
    fig1.tight_layout()

    plt.show()


def get_data(
    folder_path: Path,
    save_figure_path: Path,
    classification_activation: bool,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[BMIdataset, BMIdataset, DataLoader, DataLoader]:

    train_folder_path = Path.joinpath(folder_path, 'train')
    test_folder_path = Path.joinpath(folder_path, 'test')
    train_csv_path = Path.joinpath(folder_path, 'CSV/train_data.csv')
    test_csv_path = Path.joinpath(folder_path, 'CSV/test_data.csv')

    # cosider using old paper Resize function instead of transforms.Resize . results may be better.
    train_transform = transforms.Compose(
        [
            # Resize(img_size),
            transforms.Resize(size=(img_size, img_size)),
            # transforms.TrivialAugmentWide(num_magnitude_bins=31),
            # transforms.Pad(img_size),
            # transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            # transforms.Normalize(IMG_MEAN, IMG_STD)
        ]
    )

    test_transform = transforms.Compose(
        [
            # Resize(img_size),
            transforms.Resize(size=(img_size, img_size)),
            # transforms.Pad(img_size),
            # transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            # transforms.Normalize(IMG_MEAN, IMG_STD)
        ]
    )

    # creating train and test datasets
    train_data_custom = BMIdataset(
        csv_path=train_csv_path,
        images_folder_path=train_folder_path,
        transform=train_transform,
        classification=classification_activation,
    )

    test_data_custom = BMIdataset(
        csv_path=test_csv_path,
        images_folder_path=test_folder_path,
        transform=test_transform,
        classification=classification_activation,
    )

    # Checking length of datasets and making sure they was created succesfully
    print(f'The len of train data is {len(train_data_custom)}')
    print(f'The len of test data is {len(test_data_custom)}')

    print(f"Creating DataLoader's with batch size {batch_size} and {num_workers} workers.")

    # Create DataLoader's
    train_dataloader = DataLoader(
        dataset=train_data_custom, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    test_dataloader = DataLoader(
        dataset=test_data_custom,
        batch_size=1,  # Consider which is better - 1 or batch_size
        num_workers=num_workers,
        shuffle=False,
    )

    # making sure the DataLoader was created succesfully
    print(f'Train data loader: {train_dataloader}')
    print(f'Test data loader: {test_dataloader}')

    # path to save fig
    check_and_create_folder(save_figure_path)

    # visualization of the data
    display_random_images(dataset=train_data_custom, save_path=save_figure_path)

    return train_data_custom, test_data_custom, train_dataloader, test_dataloader, train_transform, test_transform
