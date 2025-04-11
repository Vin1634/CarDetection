#!/usr/bin/env python
# coding: utf-8

# ### Detecting cars with Yolo hosted on Roboflow

# In[ ]:


import cv2
import os
import csv
from roboflow import Roboflow
import pandas as pd


# Initialize Roboflow API
rf = Roboflow(api_key="JZS2yC3dw3yIHxghZmUP") 
project = rf.workspace().project("cardetectionashaiman")
model = project.version(2).model

folder_path = r'C:\Users\felix\Documents\MT_Data\Zone10_market' 
output_path = r'C:\Users\felix\Documents\MT_Data\Yolo\Zone10_market'  # Output folder for detected images and CSVs
os.makedirs(output_path, exist_ok=True)  # Ensure output folder exists

# Get all image filenames in the folder
image_filenames = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.JPG', '.png'))]

for filename in image_filenames:
    image_path = os.path.join(folder_path, filename)
    img = cv2.imread(image_path)

    if img is not None:
        # Make prediction
        prediction = model.predict(image_path, confidence=50, overlap=50).json()
        detections = prediction['predictions']

        # Convert detections to a DataFrame for CSV output
        df = pd.DataFrame(detections)
        df['image'] = filename  # Add the image filename as a column

        # Save bounding boxes as CSV
        csv_output_path = os.path.join(output_path, f"{filename}_detections.csv")
        df.to_csv(csv_output_path, index=False)

        # Draw bounding boxes on the image
        """""
        for det in detections:
            x, y, w, h = int(det['x']), int(det['y']), int(det['width']), int(det['height'])
            x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2  # Convert YOLO format to (x1, y1, x2, y2)
            label = det.get('class', 'Object')  # Class label
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 3)

        # Save the image with bounding boxes
        image_output_path = os.path.join(output_path, f"detected_{filename}")
        cv2.imwrite(image_output_path, img)
        """""
        print(f"Processed {filename}. Saved detected image and CSV.")
    else:
        print(f"Error loading image: {filename}")

