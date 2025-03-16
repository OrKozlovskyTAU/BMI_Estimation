from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import numpy as np
import torch
import glob
import os

# load Mask2Former fine-tuned on COCO panoptic segmentation
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")

image_paths = './Org_sorted/train/*'
dest_folder = './Seg_data/'
image_paths = glob.glob(image_paths)
os.makedirs(dest_folder, exist_ok=True)


for path in image_paths:
    if not os.path.exists(os.path.join(dest_folder, 'seg_' + os.path.basename(path)[4:])):
        # Load an input image
        image = Image.open(path)
        inputs = processor(images=image, return_tensors="pt")

        # no gradient for faster processing
        with torch.no_grad():
            outputs = model(**inputs)

        # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        # you can pass them to processor for postprocessing
        result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        # we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)
        predicted_panoptic_map = result["segmentation"]
        predicted_panoptic_map_info = result["segments_info"]

        # print(f'Segmentation map: {predicted_panoptic_map}')
        # print(f'Segmentation info: {predicted_panoptic_map_info}')

        my_array = predicted_panoptic_map.numpy().astype(int)

        # # Save NumPy array to CSV file with integer formatting
        # np.savetxt('my_tensor.csv', my_array, delimiter=',', fmt='%d')

        # Flatten the tensor into a 1D tensor
        flattened_tensor = predicted_panoptic_map.view(-1)

        # Find unique values and count them
        unique_values = torch.unique(flattened_tensor)
        num_unique_values = unique_values.size(0)

        # print("Number of unique values in map:", num_unique_values)

        # Create an empty list to store label_id values
        label_ids_list = []

        # Iterate through all label_id values in predicted_panoptic_map_info
        for segment_info in predicted_panoptic_map_info:
            label_id = segment_info['label_id']
            label_ids_list.append(label_id)

        # print(f'Label ids list: {label_ids_list}')

        # Define a color mapping for segmentation classes
        color_map = {
            1: (255, 0, 0),     # Red
            2: (0, 255, 0),     # Green
            3: (0, 0, 255),     # Blue
            4: (255, 255, 0),   # Yellow
            5: (255, 0, 255),   # Magenta
            6: (0, 255, 255),   # Cyan
            7: (128, 0, 0),     # Maroon
            8: (0, 128, 0),     # Green (dark)
            9: (0, 0, 128),     # Blue (dark)
            10: (128, 128, 0),  # Olive
            11: (128, 0, 128),  # Purple
            12: (0, 128, 128),  # Teal
            13: (128, 128, 128),# Gray
            14: (192, 192, 192),# Silver
            15: (255, 165, 0)   # Orange
            # Add more colors as needed
        }


        # Convert the original image to a numpy array
        image_array = np.array(image, dtype=np.uint8)

        # Create a blank image for the segmented output
        segmented_image = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)

        # # Apply color mapping to the segmentation map
        # for class_label, color in color_map.items():
        #     # Create a mask for the current class label
        #     mask = (predicted_panoptic_map == class_label)
        #     # Assign the color to the pixels where the mask is True
        #     segmented_image[mask] = color
        
        idx = min(len(label_ids_list), len(color_map.keys()))

        # creating a tensor of 'False' in the size of predicted_panoptic_map
        temp_mask = torch.full_like(predicted_panoptic_map, False, dtype=torch.bool)
        # print(temp_mask.size())
        # print(predicted_panoptic_map.size())

        # Apply color mapping to the segmentation map and remove specific parts from the original image
        for label, color in color_map.items():
            # Get the ID corresponding to the label
            color_label = list(color_map.keys()).index(label)
            # Create a mask for the current class label
            mask = (predicted_panoptic_map == color_label)
            # Overlay the mask on the blank image with corresponding color
            segmented_image[mask] = color

            # Counting the number of 'True' values in the tensors
            num_of_true_mask = torch.sum(mask).item()
            num_of_true_temp_mask = torch.sum(temp_mask).item()
            # Updating temp_mask
            if color_label <= idx and label_ids_list[color_label - 1] == 0 and num_of_true_mask > num_of_true_temp_mask:
                temp_mask = mask

            # # Remove the corresponding parts from the original image
            # if color_label < idx and label_ids_list[color_label] != 131:
            # # if color_label < idx and label_ids_list[color_label] == 0:
            #     # Set the pixel values of the hair region to white (remove hair)
            #     image_array[mask] = [255, 255, 255]
            
            # if color_label <= idx:
            #     # Convert the segmented image to PIL format
            #     segmented_image_pil = Image.fromarray(segmented_image)

            #     # Blend the original image with the segmented colors using alpha blending
            #     alpha = 0.5  # Adjust the alpha value for blending
            #     overlayed_image = Image.blend(image, segmented_image_pil, alpha)

            #     # Convert the modified numpy array back to a PIL image
            #     modified_image = Image.fromarray(image_array)

            #     # Save the overlayed image with the mask to a file
            #     overlayed_image.save("overlayed_image_with_mask_____coco" + str(color_label) + ".png")

            #     # Save the modified image with removed parts to a file
            #     modified_image.save("modified_image_removed_parts__________coco" + str(color_label) + ".png")

        inverse_mask = ~temp_mask

        # Turning the map to an array of type int
        inverse_mask_array = inverse_mask.numpy().astype(int)

        # # Save NumPy array to CSV file with integer formatting
        # np.savetxt('inverse_mask_tensor.csv', inverse_mask_array, delimiter=',')

        image_array[inverse_mask] = [255, 255, 255]

        modified_image = Image.fromarray(image_array)

        # Save the segmented image to a file in the destination folder
        output_path = os.path.join(dest_folder, 'seg_' + os.path.basename(path)[4:]) # getting the file name without the first 4 characters
        modified_image.save(output_path)


        # print("Images saved successfully!")


# Checking if there might be a problem with the segmented images
white_threshold = 0.7
# white color space
white_pixels = [255, 255, 255]
# list for suspicious images
suspicious_images = []

for file in os.listdir(dest_folder):
    # loading each image
    image = Image.open(os.path.join(dest_folder, file))
    # casting from PIL to numpy array
    image_np = np.array(image)
    # Count the number of white pixels
    white_pixel_count = np.sum(np.all(image_np == white_pixels, axis=2))
    # calculating the total number of pixels in the image
    image_total_pixels = image.size[0]*image.size[1]
     # check if more than {threshold} of the segmented image is covered in white
    if white_pixel_count > image_total_pixels*white_threshold:
        print(f'There might be a problem with the segmentation of this image: {file}')
        suspicious_images.append(file)

print(f'The number of suspected bad segmented images is: {len(suspicious_images)}')





# "categories":
# [{"id": 1, "name": "bed"}, {"id": 2, "name": "windowpane"}, {"id": 3, "name": "cabinet"}, {"id": 4, "name": "person"}, {"id": 5, "name": "door"}, {"id": 6, "name": "table"},
# {"id": 7, "name": "curtain"}, {"id": 8, "name": "chair"}, {"id": 9, "name": "car"}, {"id": 10, "name": "painting"}, {"id": 11, "name": "sofa"}, {"id": 12, "name": "shelf"}, {"id": 13, "name": "mirror"},
# {"id": 14, "name": "armchair"}, {"id": 15, "name": "seat"}, {"id": 16, "name": "fence"}, {"id": 17, "name": "desk"}, {"id": 18, "name": "wardrobe"}, {"id": 19, "name": "lamp"}, {"id": 20, "name": "bathtub"},
# {"id": 21, "name": "railing"}, {"id": 22, "name": "cushion"}, {"id": 23, "name": "box"}, {"id": 24, "name": "column"}, {"id": 25, "name": "signboard"}, {"id": 26, "name": "chest of drawers"}, 
# {"id": 27, "name": "counter"}, {"id": 28, "name": "sink"}, {"id": 29, "name": "fireplace"}, {"id": 30, "name": "refrigerator"}, {"id": 31, "name": "stairs"}, {"id": 32, "name": "case"},
# {"id": 33, "name": "pool table"}, {"id": 34, "name": "pillow"}, {"id": 35, "name": "screen door"}, {"id": 36, "name": "bookcase"}, {"id": 37, "name": "coffee table"}, {"id": 38, "name": "toilet"}, 
# {"id": 39, "name": "flower"}, {"id": 40, "name": "book"}, {"id": 41, "name": "bench"}, {"id": 42, "name": "countertop"}, {"id": 43, "name": "stove"}, {"id": 44, "name": "palm"}, {"id": 45, "name": "kitchen island"},
# {"id": 46, "name": "computer"}, {"id": 47, "name": "swivel chair"}, {"id": 48, "name": "boat"}, {"id": 49, "name": "arcade machine"}, {"id": 50, "name": "bus"}, {"id": 51, "name": "towel"}, {"id": 52, "name": "light"},
# {"id": 53, "name": "truck"}, {"id": 54, "name": "chandelier"}, {"id": 55, "name": "awning"}, {"id": 56, "name": "streetlight"}, {"id": 57, "name": "booth"}, {"id": 58, "name": "television receiver"},
# {"id": 59, "name": "airplane"}, {"id": 60, "name": "apparel"}, {"id": 61, "name": "pole"}, {"id": 62, "name": "bannister"}, {"id": 63, "name": "ottoman"}, {"id": 64, "name": "bottle"}, {"id": 65, "name": "van"}, 
# {"id": 66, "name": "ship"}, {"id": 67, "name": "fountain"}, {"id": 68, "name": "washer"}, {"id": 69, "name": "plaything"}, {"id": 70, "name": "stool"}, {"id": 71, "name": "barrel"}, {"id": 72, "name": "basket"}, 
# {"id": 73, "name": "bag"}, {"id": 74, "name": "minibike"}, {"id": 75, "name": "oven"}, {"id": 76, "name": "ball"}, {"id": 77, "name": "food"}, {"id": 78, "name": "step"}, {"id": 79, "name": "trade name"}, 
# {"id": 80, "name": "microwave"}, {"id": 81, "name": "pot"}, {"id": 82, "name": "animal"}, {"id": 83, "name": "bicycle"}, {"id": 84, "name": "dishwasher"}, {"id": 85, "name": "screen"}, {"id": 86, "name": "sculpture"},
# {"id": 87, "name": "hood"}, {"id": 88, "name": "sconce"}, {"id": 89, "name": "vase"}, {"id": 90, "name": "traffic light"}, {"id": 91, "name": "tray"}, {"id": 92, "name": "ashcan"}, {"id": 93, "name": "fan"},
# {"id": 94, "name": "plate"}, {"id": 95, "name": "monitor"}, {"id": 96, "name": "bulletin board"}, {"id": 97, "name": "radiator"}, {"id": 98, "name": "glass"}, {"id": 99, "name": "clock"}, {"id": 100, "name": "flag"}]}
