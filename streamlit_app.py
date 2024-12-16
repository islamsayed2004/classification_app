
import streamlit as st
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your pretrained model (replace with your actual model path)
model = tf.keras.models.load_model('model_with_selected_classes_77val.h5')

# Define the class labels
class_labels = ['Bagel', 'Apple', 'Ball', 'Bell Pepper', 'Bed']

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.resize((128, 128))  # Resize image to match model input size (224x224 for most models)
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Check if the image has an alpha channel (RGBA) and remove it if necessary
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha channel

    # Normalize the image to the range [0, 1]
    img_array = img_array / 255.0

    # Add batch dimension (required for models)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Streamlit UI
st.title("Image Classifier")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    img_array = preprocess_image(uploaded_file)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Display the prediction
    st.write(f"Prediction: {class_labels[predicted_class_index]}")

