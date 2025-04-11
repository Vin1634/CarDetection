#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import cv2
from simple_lama import SimpleLama
from PIL import Image


# In[2]:


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# In[3]:


import torch


# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Print the CUDA version
print("CUDA version:", torch.version.cuda)

# Print the device name
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))


# In[15]:


# Folder with images and CSVs
# csv folder contains the CSVs with the bounding boxes of moving cars
csv_folder = r'C:\Users\felix\Documents\GitHub\CarDetection\testzone\intersection\results\movingcars_csv'
# image folder contains the images
image_folder = r'C:\Users\felix\Documents\GitHub\CarDetection\testzone\intersection\raw_images'
# mask folder contains the masks
mask_folder = r'C:\Users\felix\Documents\GitHub\CarDetection\testzone\intersection\results\masks'
output_folder = r'C:\Users\felix\Documents\GitHub\CarDetection\testzone\intersection\results\inpainted'


# ### Create Masks

# In[14]:


import os
import cv2
import numpy as np
import pandas as pd



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

# In[16]:


from simple_lama import SimpleLama
from PIL import Image
import numpy as np
import os
from PIL import Image, ExifTags


padding = 20



# Ordner mit den Bildern und Masken
#image_folder = r'C:\Users\felix\Documents\GitHub\CarDetection\testzone\turn\raw_images'
#mask_folder = r'C:\Users\felix\Documents\GitHub\CarDetection\testzone\z1_turn\results\masks'


# Stelle sicher, dass der Ausgabe-Ordner existiert
os.makedirs(output_folder, exist_ok=True)

# SimpleLama initialisieren
lama = SimpleLama()

# Alle Bilder durchgehen
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith((".jpg", ".png")):
        image_path = os.path.join(image_folder, image_file)
        image_name = os.path.splitext(image_file)[0]  # Ohne .jpg/.png

        # Check if the image has already been inpainted
        output_path = os.path.join(output_folder, f"{image_name}_inpainted_p{padding}.JPEG")
        if os.path.exists(output_path):
            print(f"Bild {image_file} wurde bereits inpainted, überspringe...")
            continue

        # Passende Masken-Datei suchen
        mask_file = f"{image_name}_mask_p{padding}.png"
        mask_path = os.path.join(mask_folder, mask_file)
        if not os.path.exists(mask_path):
            print(f"Keine Maske für {image_file} gefunden, überspringe...")
            continue

        # Lade das Bild und die Maske
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert('L')  # In Graustufen umwandeln

        # Maske binär machen (0 oder 255)
        mask = mask.point(lambda x: 255 if x > 128 else 0)  # Alles über 128 wird 255, sonst 0
        mask = mask.convert('1')  # In binäres Format umwandeln

        # Debug: Überprüfe, ob Bild & Maske die gleiche Größe haben
        assert image.size == mask.size, f"Fehler: Bildgröße {image.size} != Maskengröße {mask.size}"

        # Inpainting ausführen
        result = lama(image, mask)

        # Lade das Originalbild mit EXIF-Daten
        original_image = Image.open(image_path)
        exif_data = original_image.info.get("exif")

        # Speichere das inpaintete Bild mit den EXIF-Daten des Originals
        output_path = os.path.join(output_folder, f"{image_name}_inpainted_p{padding}.JPEG")
        result.save(output_path, "JPEG", quality=90, optimize=True, exif=exif_data)

        # Debug: Überprüfe, ob die Datei gespeichert wurde
        assert os.path.exists(output_path), f"Fehler: Datei wurde nicht gespeichert {output_path}"

        print(f"Inpainting erfolgreich! Ergebnis gespeichert unter: {output_path}")

#run 10min for 5 images


# ### Inpainting with cv2

# import os
# import cv2
# import numpy as np
# 
# # Paths to the folders
# image_folder = r'C:\Users\felix\Documents\GitHub\CarDetection\testzone\intersection'
# mask_folder = r'C:\Users\felix\Documents\GitHub\CarDetection\testzone\intersection\results\masks'
# output_folder = r'C:\Users\felix\Documents\GitHub\CarDetection\testzone\intersection\results\inpainted'
# 
# # Ensure the output folder exists
# os.makedirs(output_folder, exist_ok=True)
# 
# # Iterate through all images
# for image_file in os.listdir(image_folder):
#     if image_file.lower().endswith((".jpg", ".png")):
#         image_path = os.path.join(image_folder, image_file)
#         image_name = os.path.splitext(image_file)[0]  # Without .jpg/.png
# 
#         # Check for corresponding mask
#         mask_file = f"{image_name}_mask_p20.png"
#         mask_path = os.path.join(mask_folder, mask_file)
#         if not os.path.exists(mask_path):
#             print(f"No corresponding mask for {image_file} found, skipping...")
#             continue
# 
#         # Load the damaged image
#         damaged_img = cv2.imread(image_path)
# 
#         # Load the mask
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
# 
#         # Inpaint the image
#         inpainted_img = cv2.inpaint(damaged_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
# 
#         # Save the inpainted image
#         output_path = os.path.join(output_folder, f"{image_name}_inpainted.jpg")
#         cv2.imwrite(output_path, inpainted_img)
#         print(f"Inpainted image saved: {output_path}")
# 

# ### Check for skipped Images 
# they have to be added to the image folder in order to calculate the orthomosaics with pix4d

# In[7]:


import os
import re

# Folder containing images
image_folder = r"C:\Users\felix\Documents\MT_Data\results\zone10_market\inpainted"

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
        print("❌ No matching images found.")
        return
    
    # Find the first and last number
    numbers.sort()
    first, last = numbers[0], numbers[-1]
    
    # Find missing numbers in sequence
    missing_numbers = [num for num in range(first, last + 1) if num not in numbers]

    if missing_numbers:
        print(f"❌ Missing image numbers: {missing_numbers}")
        missing_filenames = [f"DJI_{num:04d}.JPG" for num in missing_numbers]
        print(f"📂 Expected missing files: {missing_filenames}")
    else:
        print("✅ No missing images found. All numbers are consecutive.")

# Run the check
find_missing_images(image_folder)


