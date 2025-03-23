import dlib
import cv2
import numpy as np
import os
import pickle
import time  # Import the time module

# Load the pre-trained face detector and face recognition model from dlib
detector = dlib.get_frontal_face_detector()

# Function to extract face embedding using dlib
def get_face_embedding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Start timing face detection
    start_time = time.time()  
    faces = detector(gray)  # Detect faces
    detection_time = time.time() - start_time  # Calculate the time taken for detection
    
    return detection_time

# Directory where the dataset is stored (organized by person subdirectories)
data_dir = 'Dataset'  # Folder containing subdirectories for each person
faces = []
person_names = []  # Store names of people (instead of encoded labels)

# Variables to calculate average detection time
total_time = 0
num_images = 0

# Loop through each person's folder (subdirectory) to collect images and labels
for person_folder in os.listdir(data_dir):
    person_folder_path = os.path.join(data_dir, person_folder)
    
    if os.path.isdir(person_folder_path):
        # Load all images of the person
        for img_name in os.listdir(person_folder_path):
            img_path = os.path.join(person_folder_path, img_name)
            img = cv2.imread(img_path)
            
            # Skip non-image files
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                continue

            detection_time = get_face_embedding(img)  # Get face embeddings for the image
            
            total_time += detection_time
            num_images += 1

# Calculate the average time per image for face detection
average_detection_time = total_time / num_images if num_images > 0 else 0

print(f"Average face detection time per image: {average_detection_time} seconds")
