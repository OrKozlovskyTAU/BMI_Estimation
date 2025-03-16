from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import torch.nn as nn
import numpy as np

# Initialize the model and processor
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# URL of the image to segment
url = "./Org_data/org_002217_M_32_177800_11249092.jpg"

# Load the image from the URL
image = Image.open(url)
inputs = processor(images=image, return_tensors="pt")

# Perform semantic segmentation
outputs = model(**inputs)
logits = outputs.logits.cpu()

# Upsample the logits to the original image size
upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

# Get the predicted segmentation map
pred_seg = upsampled_logits.argmax(dim=1)[0]

# Define the class colors
class_colors = {
    "Background": [0, 0, 0],         # Black
    "Hat": [255, 0, 0],               # Red
    "Hair": [0, 255, 0],              # Green
    "Sunglasses": [0, 0, 255],        # Blue
    "Upper_clothes": [255, 255, 0],   # Yellow
    "Skirt": [255, 0, 255],           # Magenta
    "Pants": [0, 255, 255],           # Cyan
    "Dress": [128, 128, 128],         # Gray
    "Belt": [0, 128, 128],            # Teal
    "Left_shoe": [128, 0, 128],       # Purple
    "Right_shoe": [128, 128, 0],      # Olive
    "Face": [255, 255, 255],          # White
    "Left_leg": [0, 0, 128],          # Navy
    "Right_leg": [0, 128, 0],         # Green (dark)
    "Left_arm": [128, 0, 0],          # Maroon
    "Right_arm": [0, 128, 0],         # Green (light)
    "Bag": [255, 128, 0],             # Orange
    "Scarf": [128, 255, 0]            # Lime
}

# Convert the original image to a numpy array
image_array = np.array(image)

# Create a blank image for the segmented output
seg_result_colored = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)

# Apply color mapping to the segmentation map and remove specific parts from the original image
for label, color in class_colors.items():
    # Get the ID corresponding to the label
    label_id = list(class_colors.keys()).index(label)
    # Create a mask for the current class label
    mask = (pred_seg == label_id)
    # Overlay the mask on the blank image with corresponding color
    seg_result_colored[mask] = color
    # Remove the corresponding parts from the original image
    if label != 'Upper_clothes' and label != 'Pants':
        # Set the pixel values of the hair region to white (remove hair)
        image_array[mask] = [255, 255, 255]

# Convert the numpy array of the segmented image to PIL format
seg_image = Image.fromarray(seg_result_colored)

# Blend the original image with the segmented colors using alpha blending
alpha = 0.5  # Adjust the alpha value for blending
overlayed_image = Image.blend(image, seg_image, alpha)

# Convert the modified numpy array back to a PIL image
modified_image = Image.fromarray(image_array)

# Save the overlayed image with the mask to a file
overlayed_image.save("overlayed_image_with_mask.png")

# Save the modified image with removed parts to a file
modified_image.save("modified_image_removed_parts.png")

print("Images saved successfully!")