import cv2
from matplotlib import pyplot as plt
import numpy as np

def preprocess_image(img, resize_dims=(224, 224)):
    """Reads and resizes the input image."""
    # img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")
    img_resized = cv2.resize(img, resize_dims)
    return img_resized

def segment_diseased_area(img, lower_color_range, upper_color_range):
    """Applies color-based segmentation and returns the mask and cleaned mask."""
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, lower_color_range, upper_color_range)
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask_cleaned

def find_contours_and_area(img, mask_cleaned):
    """Detects contours on the cleaned mask, draws them on the image, and calculates the total area."""
    img_copy = img.copy()
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 2)
    areas = [cv2.contourArea(c) for c in contours]
    total_area = sum(areas)
    return img_copy, total_area

def predict_disease_area(area):
    """Predicts disease severity based on the total diseased area."""
    if area == 0:
        return "Healthy"
    elif area > 0 and area < 300:
        return "Mild Disease"
    elif area > 300 and area < 800:
        return "Moderate Disease"
    elif area > 800:
        return "Severe Disease"

def display_result(img, disease_severity, total_area):
    """Displays the contoured image with disease severity and area annotations."""
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Diseased Area Detection = {disease_severity}")
    plt.xlabel(f"Area = {total_area}", fontsize=12)
    plt.xticks([]), plt.yticks([])
    plt.show()

def process_image(img):
    """
    Full pipeline to preprocess, segment, detect contours, and predict disease severity.
    Returns the contoured image, total diseased area, and disease severity.
    """
    # Parameters for color segmentation
    lower_color_range = np.array([10, 100, 100])
    upper_color_range = np.array([25, 255, 255])
    
    # Preprocess the image
    img = preprocess_image(img)
    
    # Segment the diseased area
    mask_cleaned = segment_diseased_area(img, lower_color_range, upper_color_range)
    
    # Find contours and calculate area
    contoured_image, total_area = find_contours_and_area(img, mask_cleaned)
    
    # Predict disease severity
    disease_severity = predict_disease_area(total_area)
    
    return contoured_image, total_area, disease_severity
