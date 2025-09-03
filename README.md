# Histopathology Cancer Detection with Vision Transformers

## Overview

This project presents a deep learning solution for detecting metastatic cancer in histopathologic scans of lymph node sections. I have developed and implemented a Vision Transformer (ViT) model from scratch using PyTorch to classify images with high accuracy.

The repository includes an interactive web application built with Streamlit that showcases the fully-trained model's capabilities. You can simulate the training process and perform live predictions on your own images.

## Key Features

- **Custom Vision Transformer:** A ViT model implemented from scratch in PyTorch, showcasing a modern and efficient deep learning architecture.
- **Interactive Web Demo:** A user-friendly web interface to demonstrate the model's performance and prediction workflow.
- **High Accuracy:** The model achieves excellent performance on the classification task, as shown in the results section.
- **Clean and Organized Codebase:** The project is structured with a clear separation of source code, scripts, and documentation.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Dataset Details

This model was trained on the **Histopathologic Cancer Detection** dataset from Kaggle.
- **Total Images:** 220,025
- **Image Size:** 96x96 pixels
- **Color Space:** RGB
- **Task:** Binary Classification (Metastatic vs. Non-Metastatic)

## Interactive Demo

Follow these instructions to get the interactive demo running on your local machine.

### Prerequisites

- Python 3.8 or higher
- `pip` package manager

### Installation and Setup

1.  **Clone the Repository:**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd <YOUR_REPOSITORY_NAME>
    ```

2.  **Create a Virtual Environment:**
    It is highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    All the required packages are listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

### Running the Demo

Once the setup is complete, you can launch the Streamlit application with a single command:

```bash
streamlit run app.py
```

This will open the interactive demo in your web browser.

## Model Performance

The Vision Transformer architecture demonstrates powerful performance on this image classification task. The following graphs show the accuracy and loss curves from the training process.

![Training and Validation Accuracy](https://user-images.githubusercontent.com/101819411/235392389-93131fd6-781a-4750-8b22-6440f80cfd45.png)

![Training and Validation Loss](https://user-images.githubusercontent.com/101819411/235392575-6b7f5615-c5f1-4e67-aa3a-956e6d1650f4.png)
