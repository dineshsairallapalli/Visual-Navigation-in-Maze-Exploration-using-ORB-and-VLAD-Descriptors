import cv2
import numpy as np
import os

# Path to your images folder
image_folder = "data/images_subsample"
descriptors = []

# Initialize SIFT detector
orb = cv2.ORB_create()

# Loop through all images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            keypoints = orb.detect(img,None)
            keypoints, desc = orb.compute(img, keypoints)
            if desc is not None:
                descriptors.extend(desc)

# Convert to numpy array and save
descriptors = np.array(descriptors)
np.save("orb_descriptors.npy", descriptors)
print("ORB descriptors saved to orb_descriptors.npy")