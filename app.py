import streamlit as st
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

# 1. Page Configuration
st.set_page_config(page_title="Plant Doctor AI", layout="centered")
st.title("🌿 Plant Disease Detector")

# 2. Load LiteRT Model
@st.cache_resource
def load_model():
    # model.tflite file GitHub-la kandippa irukkanum
    interpreter = Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    class_names = [
        'Pepper Bacterial spot', 'Pepper healthy', 'Potato Early blight', 
        'Potato healthy', 'Potato Late blight', 'Tomato Target Spot', 
        'Tomato Mosaic virus', 'Tomato YellowLeaf Curl', 'Tomato Bacterial spot', 
        'Tomato Early blight', 'Tomato healthy', 'Tomato Late blight', 
        'Tomato Leaf Mold', 'Tomato Septoria spot', 'Tomato Spider mites'
    ]

    file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).convert('RGB').resize((224, 224))
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocessing
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        result_index = np.argmax(output)
        st.success(f"Detected: **{class_names[result_index]}**")
        st.info(f"Confidence: {np.max(output)*100:.2f}%")

except Exception as e:
    st.error(f"Error: {e}")
