import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input  # Or your model's preprocessing function

# Load the trained model
model = load_model('brain_tumor_model.h5')

# Streamlit app title
st.title("Brain Tumor Detection Website")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(uploaded_file, use_container_width=True)  # Updated to use_container_width

    # Convert the image to an array
    img_array = image.img_to_array(img)
    
    # Add batch dimension (model expects a batch of images)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image
    img_array = preprocess_input(img_array)  # Use the appropriate preprocess function

    # Predict the class of the image
    prediction = model.predict(img_array)

    # Decode the prediction (assuming the model outputs a probability vector)
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']  # Adjust class names if needed
    predicted_class = class_names[np.argmax(prediction)]

    # Show the prediction result
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Prediction Confidence: {np.max(prediction)*100:.2f}%")
