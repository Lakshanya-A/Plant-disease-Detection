import cv2
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from pipeline import predict_disease
from processing import process_image

# Title
st.title("PlantPulse: Plant Disease Detection")
def set_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp, .ezrtsby2, .eczjsme18, .stAppHeader, .stSidebar{{
            background: linear-gradient(rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.3)), url("{image_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        @media(prefers-color-scheme:dark){{
            .stApp, h1, h3{{
                color:#000000;
            }}
        container{{
            data-layout: "wide";
        }}
        }}
        </style>
        """,unsafe_allow_html = True
    )
set_background_image("https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjk5MC0wNWEta3MwMXF5bGQuanBn.jpg")
# Description
st.subheader("Your Digital Companion for Plant Health")
st.markdown("""
Welcome to **PlantPulse**, your trusted resource for identifying and managing plant diseases effectively.
Upload an image or capture one with your camera to begin.
""")


# Sidebar for navigation
st.sidebar.title("Navigation")
choice = st.sidebar.selectbox("Options", ["Home", "Upload Image", "Use Camera"])


def det_result(image):
    image_np = np.array(image)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:  # RGB image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    predicted_disease = predict_disease(image_np)
    contoured_image, total_area, disease_severity = process_image(image_np)
    
    # Display the annotated image
    st.image(contoured_image, caption="Disease Detection Output", width=300, channels="BGR")
    
    # Display text information
    st.markdown("### Prediction Results")
    st.write(f"**Predicted Disease:** {predicted_disease}")
    st.write(f"**Disease Severity:** {disease_severity}")
    st.write(f"**Total Diseased Area:** {total_area} pixels")
    
    
if choice == "Upload Image":
    # Image Upload Section
    uploaded_file = st.file_uploader("Upload an image of your plant", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        st.write("Image uploaded successfully!")
        det_result(image)

elif choice == "Use Camera":
    # Camera Capture Section
    st.markdown("### Capture an Image Using Your Camera")
    img_file_buffer = st.camera_input("Take a picture")
    st.markdown(
    """
    <style>
    div[data-testid="stCameraInput"] {
        max-width: 300px; /* Adjust the width here */
        margin: 0 auto; /* Center the camera area */
    }
    </style>
    """,
    unsafe_allow_html=True
)

    if img_file_buffer is not None:
        # Convert the image to a format usable with OpenCV
        image = Image.open(img_file_buffer)
        st.image(image, caption="Captured Image",width=300)
        st.write("Image captured successfully!")
        det_result(image)

else:
    # Home Page
    st.write("Explore options using the sidebar.")
