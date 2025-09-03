import streamlit as st
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchsummary import summary
import io
import sys
import os

# Add src directory to path to find custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vit_model import ViT


# --- Page Configuration ---
st.set_page_config(
    page_title="Urvish Joshi | IIIT V | Histopathology Project",
    layout="wide"
)

# --- Custom CSS for Styling ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# You can create a style.css file to customize further, or embed directly
st.markdown("""
<style>
@keyframes glow {
    0% {
        box-shadow: 0 0 2px #ffffff, 0 0 5px #ffffff, 0 0 7px #007bff, 0 0 10px #007bff;
    }
    50% {
        box-shadow: 0 0 5px #ffffff, 0 0 10px #ffffff, 0 0 15px #007bff, 0 0 20px #007bff;
    }
    100% {
        box-shadow: 0 0 2px #ffffff, 0 0 5px #ffffff, 0 0 7px #007bff, 0 0 10px #007bff;
    }
}

.link-container {
    background-color: #333333; /* Dark Grey */
    color: #ffffff; /* White text */
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
    font-size: 1.1em; /* Increased font size */
    animation: glow 2.5s infinite; /* Apply the glow animation */
}
.link-container a {
    color: #ffffff !important; /* White link text, !important to override default */
    text-decoration: underline;
}
.details-container {
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 10px;
    margin-bottom: 20px;
}
.center-text {
    text-align: center;
}
.main-title {
    font-size: 2.2em; /* Further reduced font size */
    text-align: center;
}
.top-header {
    text-align: center;
    padding: 5px 0;
    margin-bottom: 15px;
    font-size: 1.3em; /* Increased font size */
}
.stButton>button {
    background-color: #4CAF50; /* Green */
    color: white;
    padding: 10px 24px;
    border-radius: 8px;
    border: none;
    font-size: 1em;
    font-weight: bold;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #45a049;
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# --- Top Header Bar ---
# st.markdown(
#     """
#     <div class="top-header">
#         <span><strong>Urvish Joshi</strong></span> &nbsp;|&nbsp; <span><em>B.Tech CSE, IIIT Vadodara</em></span>
#     </div>
#     """,
#     unsafe_allow_html=True
# )


# --- Header Section ---
st.markdown('<h1 class="main-title">Project Title : Histopathology Cancer Detection | Deep Learning (ResNet, ViT)</h1>', unsafe_allow_html=True)

# Personal Details Section (Moved)
st.markdown(
    """
    <div class="top-header">
        <span><strong>Urvish Joshi</strong></span> &nbsp;|&nbsp; <span><em>B.Tech CSE, IIIT Vadodara</em></span>
    </div>
    """,
    unsafe_allow_html=True
)

# GitHub Link Section
st.markdown(
    """
    <div class="link-container">
        <strong>GitHub Link to the Project:</strong> 
        <a href="https://github.com/urvishjoshi-19/Histopathology-Cancer-Detection-Deep-Learning-ResNet-ViT-" target="_blank">
            https://github.com/urvishjoshi-19/Histopathology-Cancer-Detection-Deep-Learning-ResNet-ViT-
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Project Details Section ---
st.markdown(
    """
    <div class="details-container">
        <h4>Project Details</h4>
        <ul>
            <li><strong>Dataset Used:</strong> Histopathologic Cancer Detection dataset from Kaggle. It contains 220,025 color images of tissue samples (96x96 pixels). <a href="https://www.kaggle.com/competitions/histopathologic-cancer-detection/data" target="_blank">Click Here</a></li>
            <li><strong>Objective:</strong> To classify histopathologic images as either containing metastatic tissue or not.</li>
            <li><strong>Model:</strong> A custom Vision Transformer (ViT) implemented from scratch in PyTorch.</li>
            <li><strong>Training Technique:</strong> The model was trained using the "One Cycle Policy," an advanced learning rate scheduling technique to improve convergence and accuracy.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)


# --- Main App Content ---

# Introduction and Project Overview
st.header('Project Overview')
st.write("""
The goal of this project is to accurately detect cancerous cells in histopathologic images of tissue samples. 
This is a challenging task because it requires identifying subtle differences between cancerous and non-cancerous cells in large, complex images. 

I have used a Vision Transformer (ViT), a powerful deep learning model, which I implemented from scratch in PyTorch. 
This interactive web app demonstrates the key components and simulated performance of my model.
""")

# --- Model Architecture ---
st.header('Vision Transformer (ViT) Model Architecture')
st.write("Here is a summary of the ViT model I implemented:")

# Capture the model summary
@st.cache_resource
def get_model_summary():
    model = ViT(image_size=96, patch_size=6,
                num_classes=2, dim=64, depth=4,
                heads=4, mlp_dim=256, pool = 'cls',
                channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.)
    
    # Use a string buffer to capture the summary output
    buffer = io.StringIO()
    # Temporarily redirect stdout
    
    old_stdout = sys.stdout
    sys.stdout = buffer
    
    summary(model, (3, 96, 96))
    
    # Restore stdout
    sys.stdout = old_stdout
    
    return buffer.getvalue()

model_summary = get_model_summary()
st.code(model_summary, language='text')

st.write("My custom ViT model is significantly smaller than traditional models like ResNet-18, yet it achieves comparable performance. This highlights the efficiency of the transformer architecture.")

# --- Training Simulation ---
st.header('Training Process Simulation')

if st.button('Run Training Simulation'):
    st.write("Simulating the training process for 5 epochs...")
    
    # Initialize placeholder for charts
    accuracy_chart = st.empty()
    loss_chart = st.empty()
    
    # Simulate the training loop
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize data for charts
    chart_data = pd.DataFrame(columns=['Epoch', 'Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss'])

    for epoch in range(1, 6):
        # Simulate training for one epoch
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1, text=f'Epoch {epoch}/5')
        
        # Fake metrics for the end of the epoch
        train_acc = 0.85 + (epoch * 0.02) + np.random.uniform(-0.01, 0.01)
        val_acc = 0.86 + (epoch * 0.02) + np.random.uniform(-0.01, 0.01)
        train_loss = 0.45 - (epoch * 0.03) + np.random.uniform(-0.01, 0.01)
        val_loss = 0.43 - (epoch * 0.03) + np.random.uniform(-0.01, 0.01)
        
        status_text.text(f"Epoch {epoch} Complete | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Update chart data
        new_data = pd.DataFrame({
            'Epoch': [epoch],
            'Training Accuracy': [train_acc],
            'Validation Accuracy': [val_acc],
            'Training Loss': [train_loss],
            'Validation Loss': [val_loss]
        })
        chart_data = pd.concat([chart_data, new_data], ignore_index=True)

        # Display charts
        accuracy_chart.line_chart(chart_data, x='Epoch', y=['Training Accuracy', 'Validation Accuracy'])
        loss_chart.line_chart(chart_data, x='Epoch', y=['Training Loss', 'Validation Loss'])

    st.success("Training simulation complete!")


# --- Live Prediction Demo ---
st.header('Live Working Prototype')
st.write("Upload an image to see a simulated prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Simulate a prediction
    time.sleep(2)
    prediction = np.random.choice(['Cancer Detected', 'No Cancer Detected'], p=[0.5, 0.5])
    confidence = np.random.uniform(0.75, 0.95)

    if prediction == 'Cancer Detected':
        st.error(f"Prediction: **{prediction}** (Confidence: {confidence:.2f})")
    else:
        st.success(f"Prediction: **{prediction}** (Confidence: {confidence:.2f})")


# --- Final Results ---
st.header('Final Model Performance')

# Display images from the README
st.image('https://user-images.githubusercontent.com/101819411/235392389-93131fd6-781a-4750-8b22-6440f80cfd45.png', caption='Training and Validation Accuracy Curves', use_container_width=True)
st.image('https://user-images.githubusercontent.com/101819411/235392575-6b7f5615-c5f1-4e67-aa3a-956e6d1650f4.png', caption='Training and Validation Loss Curves', use_container_width=True)

st.write("---")
st.write("This interactive demo was created with Streamlit.")
