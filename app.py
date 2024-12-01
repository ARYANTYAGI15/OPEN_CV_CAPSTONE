import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
print("Current working directory:", os.getcwd())

# Load the trained model
model = load_model("Garbage_classification.h5")
class_labels = {0: 'Cardboard', 1: 'Glass', 2: 'Metal', 3: 'Paper', 4: 'Plastic', 5: 'Trash', 6: 'General Waste'}

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized / 255.0
    frame_reshaped = np.expand_dims(frame_normalized, axis=0)
    return frame_reshaped

def classify_frame(frame):
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    predicted_class = np.argmax(predictions)
    return class_labels[predicted_class]

def get_recycling_info(material):
    recycling_guidance = {
        'Cardboard': 'Recycle with paper products. Avoid wet or soiled cardboard.',
        'Glass': 'Rinse and recycle in glass bins.',
        'Metal': 'Clean and recycle in metal bins. Avoid non-recyclable metals.',
        'Paper': 'Recycle clean paper products. Avoid shredded or soiled paper.',
        'Plastic': 'Recycle based on local rules. Rinse plastics with recycling codes 1 and 2.',
        'Trash': 'Dispose in general waste. Check for alternative recycling options.',
        'General Waste': 'Dispose in general waste bins.'
    }
    return recycling_guidance.get(material, "No recycling information available.")

def capture_image_with_opencv():
    cap = cv2.VideoCapture(0)  # Initialize webcam
    if not cap.isOpened():
        st.error("Error: Could not open camera.")
        return None

    st.info("Press 's' to capture an image and 'q' to quit.")
    captured_image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Unable to read from camera.")
            break

        cv2.imshow('Capture Image (Press "s" to Save)', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save the image on pressing 's'
            captured_image = frame
            st.success("Image captured successfully!")
            break
        elif key == ord('q'):  # Quit without capturing
            st.info("Exiting without capturing.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_image

# Streamlit App
st.title("Garbage Classification and Recycling Guidance")

# Option to Upload an Image or Use Webcam
st.sidebar.title("Choose an Option")
option = st.sidebar.radio("Select Input Type:", ('Upload Image', 'Capture Image with Webcam'))

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert PIL Image to NumPy array
        image_np = np.array(image)
        label = classify_frame(image_np)
        recycling_info = get_recycling_info(label)

        st.write(f"**Classified as:** {label}")
        st.write(f"**Recycling Guidance:** {recycling_info}")

elif option == 'Capture Image with Webcam':
    if st.button("Capture Image"):
        captured_image = capture_image_with_opencv()
        if captured_image is not None:
            st.image(captured_image, caption="Captured Image", channels="BGR", use_column_width=True)

            label = classify_frame(captured_image)
            recycling_info = get_recycling_info(label)

            st.write(f"**Classified as:** {label}")
            st.write(f"**Recycling Guidance:** {recycling_info}")