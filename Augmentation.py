from pathlib import Path
import numpy as np
import cv2
import os
import random
import torchvision.transforms as transforms


# Define probabilities for rotation, horizontal flip and gaussian noise (and amount of noise)
rotation_prob = 0.5
horizontal_flip_prob = 0.5
noise_prob = 0.5
rand_erase_prob = 0.5
noise_std = 15

# Define the angle range for rotation (in degrees)
rotation_angle_range = 15

# Function to add Gaussian noise to an image
def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    gauss = np.random.normal(mean, noise_std, (row, col, ch))
    noisy_image = np.clip(image + gauss, 0, 255)
    return noisy_image.astype(np.uint8)

# Function to add Gaussian noise to an image
def rand_erase(image):
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomErasing(p=1, scale=(0.01, 0.1), ratio=(0.3, 3.3), value=0),
            ])
    tran_img = transform(image)
    image = tran_img.numpy()  # Convert to NumPy
    image = np.transpose(image, (1, 2, 0))  # Convert (C, H, W) → (H, W, C)
    image = (image * 255).astype(np.uint8)
    return image


org_folder = './Org_sorted/train/'
dst_folder = './Aug_data/'


def image_aug(original_folder: str, destination_folder: str, aug_3x: bool = False) -> None:

    # Create the destination folder if it doesn't exist
    if not Path(destination_folder).exists():
        Path(destination_folder).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(original_folder):
        
        if f'aug_{file}' in os.listdir(destination_folder):
            continue

        # Opening the image and apply transforms
        image = cv2.imread(os.path.join(original_folder, file))
        
        if aug_3x:
            angle = random.uniform(-rotation_angle_range, rotation_angle_range)
            # Rotate the image
            rows, cols, _ = image.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            temp_image_1 = cv2.warpAffine(image, M, (cols, rows))
            # Save the modified image to the output directory
            cv2.imwrite(os.path.join(destination_folder, 'au1_' + file[4:]), temp_image_1)

            # Apply horizontal flip
            temp_image_2 = cv2.flip(image, 1)
            # Save the modified image to the output directory
            cv2.imwrite(os.path.join(destination_folder, 'au2_' + file[4:]), temp_image_2)
            
            # Apply Gaussian noise
            temp_image_3 = add_gaussian_noise(image)
            # Save the modified image to the output directory
            cv2.imwrite(os.path.join(destination_folder, 'au3_' + file[4:]), temp_image_3)

        else:
            # flag to insure at least one augmentation is performed on the image
            is_aug = False

            while not is_aug:
                # Apply rotation with a certain probability
                if random.random() <= rotation_prob:
                    angle = random.uniform(-rotation_angle_range, rotation_angle_range)
                    # Rotate the image
                    rows, cols, _ = image.shape
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                    image = cv2.warpAffine(image, M, (cols, rows))
                    is_aug = True

                # Apply horizontal flip with a certain probability
                if random.random() <= horizontal_flip_prob:
                    image = cv2.flip(image, 1)
                    is_aug = True

                # Check if Gaussian noise should be applied
                if random.random() <= noise_prob:
                    # Apply Gaussian noise
                    image = add_gaussian_noise(image)
                    is_aug = True
                    
                # Check if Gaussian noise should be applied
                if random.random() <= rand_erase_prob:
                    # Add random erasing
                    image = rand_erase(image)
                    is_aug = True

            # Save the modified image to the output directory
            cv2.imwrite(os.path.join(destination_folder, 'aug_' + file[4:]), image) # getting the file name without the first 4 characters


if __name__ == "__main__":
    image_aug(original_folder=org_folder, destination_folder=dst_folder, aug_3x=False)

