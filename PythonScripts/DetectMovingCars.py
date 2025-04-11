#!/usr/bin/env python
# coding: utf-8

# Check for right environment
import sys
print(sys.executable)

### Import libraries
import os
from collections import Counter
import math
import re
import subprocess
import cv2
import numpy as np
from skimage import io
from skimage.registration import phase_cross_correlation
from skimage.transform import rotate
import csv
import pandas as pd
from shapely.geometry import box
import matplotlib.pyplot as plt

# Define folder paths
# Select the test zone 
image_folder_path = r"CarDetection\Testzone\Zone10_Market\raw_images"
detect_folder_path = r"CarDetection\Testzone\Zone10_Market\detected"
csv_results = r"CarDetection\Testzone\Zone10_Market\results\movingcars"

def get_image_files(path):
    """Returns a set of image filenames (without extensions)."""
    return {os.path.splitext(f)[0] for f in os.listdir(path) if f.lower().endswith('.jpg')}

def get_detection_files(path):
    """Returns a set of detection filenames (base name only, no extension)."""
    files = [f.replace("_detections.csv", "") for f in os.listdir(path) if f.lower().endswith('_detections.csv')]
    return set(files), Counter(files)  # Return set + count dictionary

# Get files
image_files = get_image_files(image_folder_path)
detect_files, detect_count = get_detection_files(detect_folder_path)

# Print counts
print(f" Number of images: {len(image_files)}")
print(f" Number of detection files: {len(detect_files)}")

### Read Exifdata from UAV-Images

# Path to the ExifTool executable
exifToolPath = r"CarDetection\exiftool.exe"

# Initialize the dictionary to store metadata
image_metadata = {}

# Filter for image files in the folder (common extensions)
valid_extensions = {'.jpg', '.jpeg', '.png'}
image_files = [f for f in os.listdir(image_folder_path) if os.path.splitext(f)[-1].lower() in valid_extensions]

# Process each image in the folder
for filename in image_files:
    image_path = os.path.join(image_folder_path, filename)
    
    # Initialize subprocess to call ExifTool
    process = subprocess.Popen([exifToolPath, image_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    # Read and parse metadata
    infoDict = {}
    for tag in process.stdout:
        line = tag.strip().split(':')  # Split each line at the ':'
        if len(line) > 1:  # Ensure valid key-value pair
            infoDict[line[0].strip()] = line[-1].strip()

    # Extract required metadata
    try:
        relAlt = float(infoDict.get("Relative Altitude", 0))  # Default to 0 if not found
        absAlt = float(infoDict.get("Absolute Altitude", 0))
        gpslat = infoDict.get("GPS Latitude", "")
        gpslon = infoDict.get("GPS Longitude", "")
        iw = float(infoDict.get("Image Width", 0))
        fov = infoDict.get("Field Of View", "0")
        gimbal_roll_deg = float(infoDict.get("Gimbal Roll Degree", 0))
        gimbal_yaw_deg = float(infoDict.get("Gimbal Yaw Degree", 0))
        gimbal_pitch_deg = float(infoDict.get("Gimbal Pitch Degree", 0))
        flight_yaw_deg = float(infoDict.get("Flight Yaw Degree", 0))

        
        # Parse latitude
        if gpslat:
            lat_ori = gpslat[-1]
            matches = re.findall(r'(\d+(\.\d+)?)', gpslat)
            gpslat = [float(match[0]) for match in matches]
            lat_deg, lat_min, lat_sec = gpslat
            lat_dec = abs(lat_deg) + lat_min / 60 + lat_sec / 3600
            if lat_ori == 'S':
                lat_dec = -lat_dec
        else:
            lat_dec = None

        # Parse longitude
        if gpslon:
            lon_ori = gpslon[-1]
            matches = re.findall(r'(\d+(\.\d+)?)', gpslon)
            gpslon = [float(match[0]) for match in matches]
            lon_deg, lon_min, lon_sec = gpslon
            lon_dec = abs(lon_deg) + lon_min / 60 + lon_sec / 3600
            if lon_ori == 'W':
                lon_dec = -lon_dec
        else:
            lon_dec = None

        # Calculate GSD
        if iw > 0 and relAlt < 88:
            fov_value = float(re.findall(r'\d+\.\d+', fov)[0]) if fov else 0
            fovrad = math.radians(fov_value / 2)
            gsd = (math.tan(fovrad) * relAlt) / (iw / 2)
        else:
            gsd = 0.022 # Default value

        # Store the data in a dictionary
        image_metadata[filename] = {
            'Latitude': lat_dec,
            'Longitude': lon_dec,
            'Relative Altitude': relAlt,
            'Absolute Altitude': absAlt,
            'GSD': gsd,
            "Gimbal Roll Degree": gimbal_roll_deg,
            "Gimbal Yaw Degree": gimbal_yaw_deg,
            "Gimbal Pitch Degree": gimbal_pitch_deg,
            "Flight Yaw Degree": flight_yaw_deg
        }
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

# Output the collected metadata
for image_name, metadata in image_metadata.items():
    print(f"Image: {image_name}")
    #for key, value in metadata.items():
    #    print(f"  {key}: {value}")
    #print()


### Shift calculation in between two consecutive images

def preprocess_image(image):
    """Preprocess the image by converting it to grayscale and normalizing."""
    if image.ndim == 3:
        image = image.mean(axis=2)  # Convert to grayscale by averaging color channels
    return image.astype(np.float32) / 255.0  # Normalize the image

def rotate_shift(shift, yaw):
    """Apply yaw rotation to the shift vector."""
    yaw_rad = math.radians(yaw)
    rotation_matrix = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad)],
        [math.sin(yaw_rad), math.cos(yaw_rad)]
    ])
    return rotation_matrix.dot(shift)

def estimate_shift_orb(image1, image2):
    """Estimate shift between two images using ORB feature matching."""
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=5000)  # Increase features for more robust matching

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Use FLANN based matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Compute the median shift
    shift_x = np.median(points2[:, 0] - points1[:, 0])
    shift_y = np.median(points2[:, 1] - points1[:, 1])

    print(f"Estimated Shift: x={shift_x:.2f}, y={shift_y:.2f}")

    return shift_x, shift_y

def estimate_shift_phase_correlation(image1, image2, yaw1, yaw2):
    """Estimate shift using phase cross-correlation if rotation is detected."""
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    # Compute the initial pixel shift using cross-correlation
    shift, _, _ = phase_cross_correlation(image1, image2, upsample_factor=10)

    # Detect if there’s a direction change
    yaw_diff = yaw2 - yaw1
    rotation_detected = abs(yaw_diff) > 1  # Small threshold to account for noise
    corrected_shift = shift   # Initialize corrected shift

    rotation_angle = 0.0
    if rotation_detected:
        # Estimate the rotation angle based on yaw difference
        rotation_angle = yaw_diff
        
        # Rotate the second image for alignment
        image2_rotated = rotate(image2, angle=-rotation_angle, resize=False)
        
        # Recompute the shift after rotation correction
        corrected_shift, _, _ = phase_cross_correlation(image1, image2_rotated, upsample_factor=10)

    # Apply yaw correction to the shift vector
    corrected_shift = rotate_shift(corrected_shift, yaw_diff)

    return corrected_shift[0], corrected_shift[1]  # Return y, x shift

def process_image_folder_with_orb_and_phase_correlation(folder_path, image_metadata):
    """Process all images in the folder, estimating shifts using ORB and phase cross-correlation."""
    images = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.JPG', '.png', '.jpeg'))])
    results = {}

    for i in range(len(images) - 1):
        img1_path = os.path.join(folder_path, images[i])
        img2_path = os.path.join(folder_path, images[i + 1])

        image1 = io.imread(img1_path)
        image2 = io.imread(img2_path)
        # Get yaw angles from metadata
        yaw1 = image_metadata.get(images[i], {}).get('Gimbal Yaw Degree', 0)
        yaw2 = image_metadata.get(images[i + 1], {}).get('Gimbal Yaw Degree', 0)

        yaw_diff = yaw2 - yaw1
        #print(f"Yaw Difference between {images[i]} and {images[i+1]}: {yaw_diff:.2f}°")

        if abs(yaw_diff) > 1:  # Use phase cross-correlation if rotation is detected
            shift_y, shift_x = estimate_shift_phase_correlation(image1, image2, yaw1, yaw2)
        else:  # Use ORB for shift estimation if no significant rotation
            shift_x, shift_y = estimate_shift_orb(image1, image2)

        results[f"{images[i]}_{images[i+1]}"] = {
            "shift": {"y": shift_y, "x": shift_x},
            "rotation_angle": yaw_diff
        }

        print(f"Between {images[i]} and {images[i+1]}: Shift (y, x) = {shift_y:.2f}, {shift_x:.2f}")

    return results

shift_results = process_image_folder_with_orb_and_phase_correlation(image_folder_path, image_metadata)


### Bounding Box Matching


def apply_shift_and_rotation(bbox, shift, rotation_angle, image_shape):
    """Applies shift and 180-degree rotation to a bounding box."""
    x, y, width, height = bbox
    image_width, image_height = image_shape  # Ensure correctly passed as (w, h)
    
    center_x, center_y = image_width / 2, image_height / 2  # Compute image center

    x_shifted = x - shift['x']
    y_shifted = y - shift['y']

    # Step 2: Apply 180-degree rotation if needed
    if 179 <= abs(rotation_angle) <= 181:  # Allow small variations like 180.7
        x_rotated = 2 * center_x - x_shifted
        y_rotated = 2 * center_y - y_shifted
    else:
        x_rotated, y_rotated = x_shifted, y_shifted  # No rotation applied

    return x_rotated, y_rotated, width, height

def compute_rgb_similarity(cropped1, cropped2):
    """Compares two image crops based on average RGB values."""
    avg_color1 = np.mean(cropped1, axis=(0, 1))  # Average color for RGB channels of image 1
    avg_color2 = np.mean(cropped2, axis=(0, 1))  # Average color for RGB channels of image 2
    
    # Euclidean distance between the color values
    color_distance = np.linalg.norm(avg_color1 - avg_color2)
    
    # Normalize (max distance = 255 * sqrt(3) if black-white difference)
    max_distance = np.linalg.norm([255, 255, 255])
    similarity = 1 - (color_distance / max_distance)  # Similarity between 0 and 1

     # Visualisation (optional, takes a lot of resources since every match will be visualised)
    """""
    print("compute_rgb_similarity",similarity)
    plt.figure(figsize=(10, 5)) 
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cropped1, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Image 1")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cropped2, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Image 2")
    plt.axis("off")
    plt.show()
    """""
    return similarity

def convert_bbox_format(bbox):
    x_min = bbox[0] - bbox[2] / 2
    y_min = bbox[1] - bbox[3] / 2
    return [x_min, y_min, bbox[2], bbox[3]]

def compute_iou_and_rgb(bbox1, bbox2, org_bbox2, image1, image2):
    """Computes IoU and RGB similarity between two bounding boxes."""
    bbox1 = convert_bbox_format(bbox1)
    bbox2 = convert_bbox_format(bbox2)
    org_bbox2 = convert_bbox_format(org_bbox2)
    
    box1 = box(bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3])
    box2 = box(bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3])
    
    intersection = box1.intersection(box2).area
    union = box1.union(box2).area
    iou = round(intersection / union, 2) if union > 0 else 0

    if iou > 0.5:
        x1_1, y1_1, x2_1, y2_1 = int(bbox1[0]), int(bbox1[1]), int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3])
        x1_2, y1_2, x2_2, y2_2 = int(org_bbox2[0]), int(org_bbox2[1]), int(org_bbox2[0] + org_bbox2[2]), int(org_bbox2[1] + org_bbox2[3])
        
        image1 = cv2.imread(image1)
        image2 = cv2.imread(image2)
        
        cropped1 = image1[y1_1:y2_1, x1_1:x2_1]
        cropped2 = image2[y1_2:y2_2, x1_2:x2_2]
        
        if cropped1.size == 0 or cropped2.size == 0:
            print(f"Error: Empty crop for {image1} or {image2}")
            return iou, 0

        rgb_similarity = compute_rgb_similarity(cropped1, cropped2)
    else:
        rgb_similarity = 0

    return iou, rgb_similarity

def process_bounding_box_matching(detect_folder_path, image_folder_path, shift_results, iou_threshold):
    """Processes bounding box matching between previous, current, and next images using shifts."""
    shift_results = {k.lower(): v for k, v in shift_results.items()}  # Convert keys to lowercase for case-insensitive matching
    csv_files = sorted([f for f in os.listdir(detect_folder_path) if f.endswith('_detections.csv')])
    
    processed_bboxes = set()
    
    for i in range(len(csv_files)):
        current_file = csv_files[i]
        current_detections = pd.read_csv(os.path.join(detect_folder_path, current_file))
       

        image_shape = (5472, 3648)  # Replace with actual image dimensions if available

        if i == 0:
            next_file = csv_files[i + 1]
            next_detections = pd.read_csv(os.path.join(detect_folder_path, next_file))
            pair_key = f"{current_file.replace('_detections.csv', '').lower()}_{next_file.replace('_detections.csv', '').lower()}"
            
            if pair_key not in shift_results:
                print(f"Case A No shift/rotation data for {pair_key}. Skipping.")
                continue

            shift = shift_results[pair_key]['shift']
            rotation_angle = shift_results[pair_key]['rotation_angle']

            if 175 < rotation_angle < 185:
                shift = {'y': -shift['y'], 'x': -shift['x']}

            print(f"\nProcessing Case A {current_file} -> {next_file} with shift: {shift} and rotation: {rotation_angle}°")
            image1_path = os.path.join(image_folder_path, current_file.replace('_detections.csv', ''))
            image2_path = os.path.join(image_folder_path, next_file.replace('_detections.csv', ''))
            bbm_file = os.path.join(detect_folder_path, f"BBM_{current_file.replace('_detections.csv', '')}.csv")
            
            with open(bbm_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["x", "y", "width", "height", "IoU"])

                for _, bbox1 in current_detections.iterrows():
                    bbox1_coords = list(bbox1[['x', 'y', 'width', 'height']].values)
                    
                    for _, bbox2 in next_detections.iterrows():
                        bbox2_coords = list(bbox2[['x', 'y', 'width', 'height']].values)
                        transformed_bbox2_coords = list(apply_shift_and_rotation(bbox2_coords, shift, rotation_angle, image_shape))

                        iou, rgb = compute_iou_and_rgb(bbox1_coords, transformed_bbox2_coords, bbox2_coords, image1_path, image2_path)
                        if iou > 0 and iou >= iou_threshold and rgb > 0.95:
                            writer.writerow([bbox1_coords[0], bbox1_coords[1], bbox1_coords[2], bbox1_coords[3], iou])
                            print("Case A", bbox1_coords, iou, round(rgb,2))

        elif i == len(csv_files) - 1:
            prev_file = csv_files[i - 1]
            prev_detections = pd.read_csv(os.path.join(detect_folder_path, prev_file))
            pair_key = f"{prev_file.replace('_detections.csv', '').lower()}_{current_file.replace('_detections.csv', '').lower()}"

            if pair_key not in shift_results:
                print(f"Case B No shift/rotation data for {pair_key}. Skipping.")
                continue
            
            shift = shift_results[pair_key]['shift']
            inverse_shift = {'y': -shift['y'], 'x': -shift['x']}
            rotation_angle = shift_results[pair_key]['rotation_angle']

            print(f"\nProcessing Case B {current_file} -> {prev_file} with inverted shift: {inverse_shift} and rotation: {rotation_angle}°")
            image1_path = os.path.join(image_folder_path, current_file.replace('_detections.csv', ''))
            image2_path = os.path.join(image_folder_path, prev_file.replace('_detections.csv', ''))
            bbm_file = os.path.join(detect_folder_path, f"BBM_{current_file.replace('_detections.csv', '')}.csv")
            
            with open(bbm_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["x", "y", "width", "height", "IoU"])

                for _, bbox1 in current_detections.iterrows():
                    bbox1_coords = list(bbox1[['x', 'y', 'width', 'height']].values)

                    for _, bbox2 in prev_detections.iterrows():
                        bbox2_coords = list(bbox2[['x', 'y', 'width', 'height']].values)
                        transformed_bbox2_coords = list(apply_shift_and_rotation(bbox2_coords, inverse_shift, rotation_angle, image_shape))

                        iou, rgb = compute_iou_and_rgb(bbox1_coords, transformed_bbox2_coords, bbox2_coords, image1_path, image2_path)
                        if iou > 0 and iou >= iou_threshold and rgb > 0.95:
                            writer.writerow([bbox1_coords[0], bbox1_coords[1], bbox1_coords[2], bbox1_coords[3], iou])
                            print("Case B", bbox1_coords, iou, round(rgb,2))

        else:
            prev_file = csv_files[i - 1]
            next_file = csv_files[i + 1]

            prev_detections = pd.read_csv(os.path.join(detect_folder_path, prev_file))
            next_detections = pd.read_csv(os.path.join(detect_folder_path, next_file))

            pair_key_prev = f"{prev_file.replace('_detections.csv', '').lower()}_{current_file.replace('_detections.csv', '').lower()}"
            if pair_key_prev in shift_results:
                shift_prev = shift_results[pair_key_prev]['shift']
                inverse_shift_prev = {'y': -shift_prev['y'], 'x': -shift_prev['x']}
                rotation_angle_prev = shift_results[pair_key_prev]['rotation_angle']

                print(f"\nProcessing Case C {current_file} -> {prev_file} with inverted shift: {inverse_shift_prev} and rotation: {rotation_angle_prev}°")

                image1_path = os.path.join(image_folder_path, current_file.replace('_detections.csv', ''))
                image2_path = os.path.join(image_folder_path, prev_file.replace('_detections.csv', ''))
                bbm_file = os.path.join(detect_folder_path, f"BBM_{current_file.replace('_detections.csv', '')}.csv")

                with open(bbm_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if os.stat(bbm_file).st_size == 0:  # Write header only if file is empty
                        writer.writerow(["x", "y", "width", "height", "IoU"])

                    for _, bbox1 in current_detections.iterrows():
                        bbox1_coords = tuple(bbox1[['x', 'y', 'width', 'height']].values)

                        for _, bbox2 in prev_detections.iterrows():
                            bbox2_coords = list(bbox2[['x', 'y', 'width', 'height']].values)
                            transformed_bbox2_coords = list(apply_shift_and_rotation(bbox2_coords, inverse_shift_prev, rotation_angle_prev, image_shape))

                            iou, rgb = compute_iou_and_rgb(bbox1_coords, transformed_bbox2_coords, bbox2_coords, image1_path, image2_path)

                            if iou > 0 and iou >= iou_threshold and rgb > 0.95:
                                if bbox1_coords not in processed_bboxes:  # Prevent duplicate entries
                                    writer.writerow([*bbox1_coords, iou])
                                    processed_bboxes.add(bbox1_coords)  # Store the bounding box
                                    print("Case C", bbox1_coords, iou, round(rgb,2))

            pair_key_next = f"{current_file.replace('_detections.csv', '').lower()}_{next_file.replace('_detections.csv', '').lower()}"
            if pair_key_next in shift_results:
                shift_next = shift_results[pair_key_next]['shift']
                rotation_angle_next = shift_results[pair_key_next]['rotation_angle']

                if 175 < rotation_angle_next < 185:
                    shift_next = {'y': -shift_next['y'], 'x': -shift_next['x']}

                print(f"\nProcessing Case D {current_file} -> {next_file} with shift: {shift_next} and rotation: {rotation_angle_next}°")

                image1_path = os.path.join(image_folder_path, current_file.replace('_detections.csv', ''))
                image2_path = os.path.join(image_folder_path, next_file.replace('_detections.csv', ''))

                with open(bbm_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)

                    for _, bbox1 in current_detections.iterrows():
                        bbox1_coords = tuple(bbox1[['x', 'y', 'width', 'height']].values)

                        if bbox1_coords in processed_bboxes:
                            continue  # Skip this bounding box in Case D

                        for _, bbox2 in next_detections.iterrows():
                            bbox2_coords = list(bbox2[['x', 'y', 'width', 'height']].values)
                            transformed_bbox2_coords = list(apply_shift_and_rotation(bbox2_coords, shift_next, rotation_angle_next, image_shape))

                            iou, rgb = compute_iou_and_rgb(bbox1_coords, transformed_bbox2_coords, bbox2_coords, image1_path, image2_path)

                            if iou > 0 and iou >= iou_threshold and rgb > 0.95:
                                writer.writerow([*bbox1_coords, iou])
                                print("Case D", bbox1_coords, iou, round(rgb,2))

process_bounding_box_matching(
    detect_folder_path=detect_folder_path,
    image_folder_path=image_folder_path,
    shift_results=shift_results,
    iou_threshold=0.5,  # IoU threshold
)

### Visualisation with YOLO BBs

def draw_bboxes_from_csv(image_path, csv_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading the image {image_path}")
        return

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Load the CSV file containing the bounding boxes
    if not os.path.exists(csv_path):
        print(f"CSV File is missing: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)

    # Draw each bounding box on the image
    for _, row in df.iterrows():
        try:
            # Values from the CSV
            x = row['x']
            y = row['y']
            box_width = row['width']
            box_height = row['height']
            iou = round(row['IoU'], 2)  # Round the IoU value to 2 decimal places

            # Calculate the coordinates of the top-left and bottom-right corners
            x1 = int(x - box_width / 2)  # x1 is the top-left corner
            y1 = int(y - box_height / 2)  # y1 is the top-left corner
            x2 = int(x + box_width / 2)  # x2 is the bottom-right corner
            y2 = int(y + box_height / 2)  # y2 is the bottom-right corner
            # Draw a rectangle (x1, y1 is the top-left corner, x2, y2 is the bottom-right corner)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red color and thickness = 3

            # Add the IoU value as text next to the bounding box
            label = f"IoU: {iou}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        except KeyError as e:
            print(f"KeyError: {e} in row {row}")

    # Convert the image to RGB for Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.figure(figsize=(12, 8))  # Set a larger figure size for better display
    plt.imshow(image_rgb)
    plt.axis('off')  # Turn off axes
    plt.show()


def process_folder(image_folder, csv_folder):
    # List all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg',".JPG", '.png'))]

    # Sort the images in reverse order so that the last image comes first
    image_files.sort(reverse=True)

    for image_file in image_files:
        # Full path to the image
        image_path = os.path.join(image_folder, image_file)
        
        # Remove the "detected_" prefix from the image name
        csv_file_name = image_file.replace("detected_", "")
        
        # Generate the path to the corresponding CSV file
        csv_file = f"BBM_{csv_file_name}.csv"  # Example for the CSV file name
        csv_path = os.path.join(csv_folder, csv_file)

        print(f"\nProcessing image: {image_file}")
        draw_bboxes_from_csv(image_path, csv_path)

# If images with YOLO bounding boxes are available, the function can be used to visualize the bounding boxes
user_input = input("Do you want to process the folder and visualize bounding boxes? (y/n): ").strip().lower()
if user_input == 'y':
    process_folder(detect_folder_path, detect_folder_path)
elif user_input == 'n':
    print("Skipping folder processing.")
else:
    print("Invalid input. Skipping folder processing.")

# ### Compare&Save moving Cars BBs to CSV
# Ensure that the results folder exists
os.makedirs(csv_results, exist_ok=True)

# Get all files in the folder
data = os.listdir(detect_folder_path)

# Filter *_detections.csv and BBM_*.csv files
detections_files = [f for f in data if f.endswith("_detections.csv")]
bbm_files = [f for f in data if f.startswith("BBM_") and f.endswith(".csv")]

# Create a mapping based on image identifiers
detections_dict = {re.sub(r'_detections.csv$', '', f): f for f in detections_files}
bbm_dict = {re.sub(r'BBM_', '', f).replace('.csv', ''): f for f in bbm_files}

# Process only matching pairs
for id, detection_file in detections_dict.items():
    if id in bbm_dict:
        bbm_file = bbm_dict[id]

        csv1_path = os.path.join(detect_folder_path, detection_file)
        csv2_path = os.path.join(detect_folder_path, bbm_file)

        # Load CSV files
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)

        # Check if the first dataframe has entries, if not skip
        if df1.empty:
            print(f"No entries in the first file {id}.")
            continue
        else:
            set1 = set(zip(df1["x"], df1["y"]))
        if df2.empty:
            print(f"No entries for the second file {id}.")
            continue
        else:
            set2 = set(zip(df2["x"], df2["y"]))
        # Entries that are only in the first file
        unique_entries = set1 - set2

        # If no differences, process the next file
        if not unique_entries:
            print(f"No Unique entries {id}.") 
            continue
        
        # Filter data for output
        df1_filtered = df1[df1.apply(lambda row: (row["x"], row["y"]) in unique_entries, axis=1)]

        # Save the result file
        result_file = os.path.join(csv_results, f"movingcar_BB_{id}.csv")
        df1_filtered.to_csv(result_file, index=False)
        print(f"{id} saved {result_file}")

print("All moving cars detected!")

