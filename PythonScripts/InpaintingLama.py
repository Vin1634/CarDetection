#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import cv2
import re
from simple_lama import SimpleLama
from PIL import Image
import torch
from PIL import Image, ExifTags

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Print the CUDA version
print("CUDA version:", torch.version.cuda)

# Print the device name
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))


# Folder paths
# csv folder contains the CSVs with the bounding boxes of moving cars
csv_folder = r'CarDetection\Testzone\Zone1_intersection\results\movingcars_csv'
# image folder contains the images
image_folder = r'CarDetection\Testzone\Zone1_intersection\raw_images'
# mask folder contains the masks
mask_folder = r'\Testzone\Zone1_intersection\results\masks'
# output folder contains the inpainted images
output_folder = r'CarDetection\Testzone\Zone1_intersection\results\inpainted'
# Folder with all the images for the orthomosaic
final_image_folder = r"\MT_Data\results\Zone1_intersection\all_inpainted"
padding = 20

# ### Create Masks
# Ensure the mask folder exists
os.makedirs(mask_folder, exist_ok=True)

# Iterate through all images
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith((".jpg", ".png")):
        image_path = os.path.join(image_folder, image_file)
        image_name = os.path.splitext(image_file)[0]  # Without .jpg/.png

        # Find the corresponding CSV file
        csv_file = os.path.join(csv_folder, f"movingcar_BB_{image_name}.JPG.csv")
        if not os.path.exists(csv_file):
            print(f"No CSV for {image_file} found, skipping...")
            continue

        # Load the image to get the correct mask size
        img = cv2.imread(image_path)
        height, width, _ = img.shape

        # Create a black mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Load CSV data
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            x_center, y_center, w, h = int(row["x"]), int(row["y"]), int(row["width"]), int(row["height"])
            
            # Add padding to the bounding box
            padding = 20
            w += 2 * padding
            h += 2 * padding
            
            # Convert center coordinates to top-left coordinates
            x = x_center - w // 2
            y = y_center - h // 2
            
            # Draw the bounding box as a white rectangle on the mask
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255), thickness=-1)

        # Save the mask as PNG
        mask_path = os.path.join(mask_folder, f"{image_name}_mask_p{padding}.png")
        cv2.imwrite(mask_path, mask)
        print(f"Mask saved: {mask_path}")


# ### Simple Lama Inpaint 
# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# SimpleLama 
lama = SimpleLama()

# Iterate through all images
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith((".jpg", ".png")):
        image_path = os.path.join(image_folder, image_file)
        image_name = os.path.splitext(image_file)[0]  # Without .jpg/.png

        # Check if the image has already been inpainted
        output_path = os.path.join(output_folder, f"{image_name}_inpainted_p{padding}.JPEG")
        if os.path.exists(output_path):
            print(f"Image {image_file} has already been inpainted, skipping...")
            continue

        # Find the corresponding mask file
        mask_file = f"{image_name}_mask_p{padding}.png"
        mask_path = os.path.join(mask_folder, mask_file)
        if not os.path.exists(mask_path):
            print(f"No mask found for {image_file}, skipping...")
            continue

        # Load the image and the mask
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale

        # Make the mask binary (0 or 255)
        mask = mask.point(lambda x: 255 if x > 128 else 0)  # Everything above 128 becomes 255, otherwise 0
        mask = mask.convert('1')  # Convert to binary format

        # Debug: Check if the image & mask have the same size
        assert image.size == mask.size, f"Error: Image size {image.size} != Mask size {mask.size}"

        # Perform inpainting
        result = lama(image, mask)

        # Load the original image with EXIF data
        original_image = Image.open(image_path)
        exif_data = original_image.info.get("exif")

        # Save the inpainted image with the EXIF data of the original
        output_path = os.path.join(output_folder, f"{image_name}_inpainted_p{padding}.JPEG")
        result.save(output_path, "JPEG", quality=90, optimize=True, exif=exif_data)

        # Debug: Check if the file was saved
        assert os.path.exists(output_path), f"Error: File was not saved {output_path}"

        print(f"Inpainting successful! Result saved at: {output_path}")

# Folder containing images

def find_missing_images(folder):
    # Regex pattern to match filenames like DJI_XXXX, ignoring extra suffixes
    pattern = re.compile(r"DJI_(\d{4})", re.IGNORECASE)
    
    # Extract numbers from filenames
    numbers = []
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            numbers.append(int(match.group(1)))  # Extract XXXX as an integer

    if not numbers:
        print("No matching images found.")
        return
    
    # Find the first and last number
    numbers.sort()
    first, last = numbers[0], numbers[-1]
    
    # Find missing numbers in sequence
    missing_numbers = [num for num in range(first, last + 1) if num not in numbers]

    if missing_numbers:
        print(f"Missing image numbers: {missing_numbers}")
        missing_filenames = [f"DJI_{num:04d}.JPG" for num in missing_numbers]
        print(f"Expected missing files: {missing_filenames}")
    else:
        print("No missing images found. All numbers are consecutive.")

# Run the check
find_missing_images(final_image_folder)
