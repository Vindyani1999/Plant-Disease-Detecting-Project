# Deep Learning Plant Disease Prediction

## Overview

This project utilizes deep learning techniques implemented in TensorFlow to predict plant diseases. It includes a user-friendly interface deployed using Streamlit, enabling users to upload images and utilize drag-and-drop functionality. The application supports a range of image types, including JPEG, PNG, and JPG.

## Features

- **Deep Learning Model:** Utilizes TensorFlow for robust and accurate disease prediction.
- **User-Friendly Interface:** Deployed with Streamlit, offering an intuitive experience for users.
- **Image Upload:** Supports multiple image formats for user convenience.
- **Drag-and-Drop:** Allows users to easily upload images by dragging and dropping them into the interface.

## Installation

To run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/plant-disease-prediction.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the application:
    ```bash
   streamlit run app.py

## Usage

1. Access the application through the provided URL or by running it locally.
2. Upload an image of a plant to predict the presence of diseases.
3. View the prediction results displayed by the application.

## Technologies Used

- **TensorFlow:** Deep learning framework for model development.
- **Streamlit:** Web application framework for deploying the user interface.
- **Python:** Programming language used for development.

## Steps

In this project wwe are going to build a plant disease prediction system and we are going to use convolutional nural network for the image classification.

## Workflow

### Step 1: Dataset Acquisition
![Dataset Acquisition](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/f678ba27-8129-4805-b3d7-5e1eb2d10df4)

The dataset is from Kaggle. We will get a Kaggle JSON file containing our Kaggle account credentials and retrieve the dataset through an API. (We are going to work with Google Colab)

### Step 2: Data Processing
![Data Processing](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/1af6f2b7-d6c4-48df-9fdc-3e07a1bc3732)

In this data processing step, we will use an image data generator class in TensorFlow. The images are loaded from the directory and prepared to be suitable as input for the neural network.

### Step 3: Data Splitting
![Data Splitting](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/9118cd50-aeee-453d-9342-a8a047edc236)

In this step, we will split our dataset into training and testing data. This step is connected with the data processing because we are going to build a pipeline for the training data generator and validation generator.

### Step 4: Model Building
![Model Building](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/3a9e8c94-035f-476c-b5a6-82b11f11e445)

Next, we will build our convolutional neural network with appropriate convolutional layers and all the required dense layers.

### Step 5: Model Evaluation and Saving
![Model Evaluation and Saving](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/d7f3cd68-1c9e-4264-be4e-7253ba07e30a)

The model is then evaluated to check if it is giving correct predictions and working as expected. After verifying, we can save the model as a file.

### Step 6: Model Export to Streamlit
![Model Export to Streamlit](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/7552ee8f-a19b-4af1-b3af-bcd16356e058)

Using the saved model, we will export this trained model and load it into the Streamlit app's Python script.

### Step 7: Dockerizing the Application
![Dockerizing the Application](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/f1bde005-2306-413d-b564-dd6c5c68b358)

As the first step of Dockerizing, we are going to create a Dockerfile that includes a list of instructions needed to create a Docker image.

### Step 8: Creating Docker Image and Container
![Creating Docker Image and Container](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/56d2f350-5d94-41db-a188-ff218cb4e2b0)

Using this Dockerfile, we can create the Docker image and then create the Docker container.

