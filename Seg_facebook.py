from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from PIL import Image
import numpy as np
import torch
import glob
import os

# Load MaskFormer fine-tuned on COCO panoptic segmentation
processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco")

image_paths = './Org_data/*'
dest_folder = './Seg_data_temp/'
image_paths = glob.glob(image_paths)
os.makedirs(dest_folder, exist_ok=True)

for path in image_paths:
    if not os.path.exists(os.path.join(dest_folder, 'seg_' + os.path.basename(path)[4:])):
        # Load an input image
        image = Image.open(path)
        inputs = processor(images=image, return_tensors="pt")

        outputs = model(**inputs)
        # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        # print(f'Class queries logits: {class_queries_logits}')
        # print(f'Class queries logits: {class_queries_logits.shape}')

        # you can pass them to processor for postprocessing
        result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
        predicted_panoptic_map = result["segmentation"]
        predicted_panoptic_map_info = result["segments_info"]

        # print(f'Segmentation map: {predicted_panoptic_map}')
        # print(f'Segmentation info: {predicted_panoptic_map_info}')

        # # Turning the map to an array of type int
        # csv_array = predicted_panoptic_map.numpy().astype(int)

        # # Save NumPy array to CSV file with integer formatting
        # np.savetxt('img_tensor.csv', csv_array, delimiter=',', fmt='%d')

        # # Flatten the tensor into a 1D tensor
        # flattened_tensor = predicted_panoptic_map.view(-1)

        # # Find unique values and count them
        # unique_values = torch.unique(flattened_tensor)
        # num_unique_values = unique_values.size(0)

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
        
        idx = min(len(label_ids_list), len(color_map.keys()))

        # creating a tensor of 'False' in the size of predicted_panoptic_map
        temp_mask = torch.full_like(predicted_panoptic_map, False, dtype=torch.bool)
        # print(temp_mask.size())
        # print(predicted_panoptic_map.size())

        # Apply color mapping to the segmentation map and remove specific parts from the original image
        for label, color in color_map.items():
            # Get the ID corresponding to the label
            color_label = list(color_map.keys()).index(label) + 1
            # Create a mask for the current class label
            mask = (predicted_panoptic_map == color_label)
            # Overlay the mask on the blank image with corresponding color
            segmented_image[mask] = color

            # Counting the number of 'True' values in the tensors
            num_of_true_mask = torch.sum(mask).item()
            num_of_true_temp_mask = torch.sum(temp_mask).item()
            # Stopping early after all image segments are done
            if color_label > idx:
                break
            # Updating temp_mask
            if label_ids_list[color_label - 1] == 0 and num_of_true_mask > num_of_true_temp_mask:
                temp_mask = mask
            
            ### CHECKS ### - DO NOT DELETE
            # # Remove the corresponding parts from the original image
            # if color_label < idx and label_ids_list[color_label - 1] != 0:
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

        # # Turning the map to an array of type int
        # inverse_mask_array = inverse_mask.numpy().astype(int)

        # # Save NumPy array to CSV file with integer formatting
        # np.savetxt('inverse_mask_tensor.csv', inverse_mask_array, delimiter=',')

        image_array[inverse_mask] = [255, 255, 255]

        modified_image = Image.fromarray(image_array)

        # Save the segmented image to a file in the destination folder
        output_path = os.path.join(dest_folder, 'seg_' + os.path.basename(path)[4:]) # getting the file name without the first 4 characters
        modified_image.save(output_path)

        # print("Images saved successfully!")


###
# not that good segmentation - seg_1_F_26_177800_14514957.jpg



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




# # Define a color mapping for segmentation classes
# category_colors_coco = {
#     "person": [220, 20, 60],            # Reddish Pink
#     "bicycle": [119, 11, 32],           # Dark Red
#     "car": [0, 0, 142],                 # Deep Blue
#     "motorcycle": [0, 0, 230],          # Blue
#     "airplane": [106, 0, 228],          # Purple
#     "bus": [0, 60, 100],                # Navy Blue
#     "train": [0, 80, 100],              # Dark Blue
#     "truck": [0, 0, 70],                # Black
#     "boat": [0, 0, 192],                # Blue
#     "traffic light": [250, 170, 30],    # Orange
#     "fire hydrant": [100, 170, 30],     # Yellow-Green
#     "stop sign": [220, 220, 0],         # Yellow
#     "parking meter": [175, 116, 175],   # Purple
#     "bench": [250, 0, 30],              # Red
#     "bird": [165, 42, 42],              # Brown
#     "cat": [255, 77, 255],              # Pink
#     "dog": [0, 226, 252],               # Sky Blue
#     "horse": [182, 182, 255],           # Light Blue
#     "sheep": [0, 82, 0],                # Dark Green
#     "cow": [120, 166, 157],             # Green-Grey
#     "elephant": [110, 76, 0],           # Brown
#     "bear": [174, 57, 255],             # Light Purple
#     "zebra": [199, 100, 0],             # Orange-Brown
#     "giraffe": [72, 0, 118],            # Dark Purple
#     "backpack": [255, 179, 240],        # Light Pink
#     "umbrella": [0, 125, 92],           # Dark Green
#     "handbag": [209, 0, 151],           # Purple-Pink
#     "tie": [188, 208, 182],             # Light Green-Grey
#     "suitcase": [0, 220, 176],          # Turquoise
#     "frisbee": [255, 99, 164],          # Pink
#     "skis": [92, 0, 73],                # Dark Red-Purple
#     "snowboard": [133, 129, 255],       # Light Purple-Blue
#     "sports ball": [78, 180, 255],      # Light Blue
#     "kite": [0, 228, 0],                # Green
#     "baseball bat": [174, 255, 243],    # Light Blue-Green
#     "baseball glove": [45, 89, 255],    # Light Blue
#     "skateboard": [134, 134, 103],      # Grey
#     "surfboard": [145, 148, 174],       # Grey-Blue
#     "tennis racket": [255, 208, 186],   # Light Orange
#     "bottle": [197, 226, 255],          # Light Blue
#     "wine glass": [171, 134, 1],        # Dark Yellow
#     "cup": [109, 63, 54],               # Dark Red-Brown
#     "fork": [207, 138, 255],            # Light Purple-Pink
#     "knife": [151, 0, 95],              # Dark Pink
#     "spoon": [9, 80, 61],               # Dark Green
#     "bowl": [84, 105, 51],              # Dark Green
#     "banana": [74, 65, 105],            # Dark Purple-Blue
#     "apple": [166, 196, 102],           # Light Green
#     "sandwich": [208, 195, 210],        # Light Grey-Purple
#     "orange": [255, 109, 65],           # Orange
#     "broccoli": [0, 143, 149],          # Greenish Blue
#     "carrot": [179, 0, 194],            # Purple-Pink
#     "hot dog": [209, 99, 106],          # Reddish Pink
#     "pizza": [5, 121, 0],               # Dark Green
#     "donut": [227, 255, 205],           # Light Green
#     "cake": [147, 186, 208],            # Light Blue
#     "chair": [153, 69, 1],              # Dark Orange
#     "couch": [3, 95, 161],              # Dark Blue
#     "potted plant": [163, 255, 0],      # Bright Green
#     "bed": [119, 0, 170],               # Purple
#     "dining table": [0, 182, 199],      # Blue-Green
#     "toilet": [0, 165, 120],            # Dark Cyan-Green
#     "tv": [183, 130, 88],               # Brown
#     "laptop": [95, 32, 0],              # Dark Brown
#     "mouse": [130, 114, 135],           # Light Purple-Grey
#     "remote": [110, 129, 133],          # Grey-Blue
#     "keyboard": [166, 74, 118],         # Pink-Purple
#     "cell phone": [219, 142, 185],      # Light Purple-Pink
#     "microwave": [79, 210, 114],        # Light Green
#     "oven": [178, 90, 62],              # Brown-Orange
#     "toaster": [65, 70, 15],            # Dark Green-Brown
#     "sink": [127, 167, 115],            # Green
#     "refrigerator": [59, 105, 106],     # Grey-Blue
#     "book": [142, 108, 45],             # Brown-Yellow
#     "clock": [196, 172, 0],             # Yellow
#     "vase": [95, 54, 80],               # Dark Purple-Red
#     "scissors": [128, 76, 255],         # Light Purple-Blue
#     "teddy bear": [201, 57, 1],         # Reddish Brown
#     "hair drier": [246, 0, 122],        # Pink-Red
#     "toothbrush": [191, 162, 208],      # Light Purple
# }


