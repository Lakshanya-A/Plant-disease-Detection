
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# Load the pre-trained model and processor for plant disease detection
processor = AutoImageProcessor.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
model = AutoModelForImageClassification.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")

def preprocess_image(img):
    # img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    # Resize the image to (224, 224) as required by the model
    img_resized = cv2.resize(img, (224, 224))
    
    # Normalize the pixel values to [0, 1]
    img_normalized = img_resized / 255.0
    
    # Convert to uint8 for OpenCV processing (important for color conversion)
    img_normalized_uint8 = (img_normalized * 255).astype(np.uint8)
    
    # Convert BGR to RGB as the model expects RGB format
    img_rgb = cv2.cvtColor(img_normalized_uint8, cv2.COLOR_BGR2RGB)
    
    return img_rgb

def predict_disease(img):
    # Preprocess image with OpenCV
    preprocessed_image = preprocess_image(img)
    
    # Prepare the image for the model using the processor
    inputs = processor(images=preprocessed_image, return_tensors="pt")
    
    # Predict the disease
    outputs = model(**inputs)
    
    # Get the predicted class
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=-1).item()  # Convert to native Python value
    predicted_label = model.config.id2label[predicted_class_idx]
    
    return predicted_label
