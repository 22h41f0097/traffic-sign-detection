# Traffic Sign Detection and Recognition

## Project Overview

Traffic Sign Detection and Recognition is an AI/ML-based computer vision
project developed using Python.

The project uses a CNN/LeNet-style neural network model to recognize
traffic sign classes from video frames.

The application provides a graphical user interface (GUI) through which
the user can select training images, generate/load the model, select a
test video, and perform traffic sign detection.

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Tkinter
- imutils
- CNN / LeNet

## Project Workflow

The project follows these main steps:

1. Select the traffic sign training-image folder.
2. Load and preprocess the training images.
3. Generate or load the CNN/LeNet model.
4. Select a test video.
5. Process the video frames using OpenCV.
6. Resize and preprocess the video frames.
7. Use the trained model to predict the traffic sign class.
8. Display the detected/predicted sign on the video.

## Traffic Sign Categories

The project documentation describes five traffic-sign categories:

- A
- B
- C
- D
- E

The application maps the trained classes to prediction labels during
traffic-sign detection.

## Model

The project uses a CNN/LeNet-style neural network architecture.

The model contains convolution, pooling, flattening, dense and dropout
layers and uses TensorFlow/Keras for training and prediction.

The model is designed to process traffic-sign images resized to
28 x 28 pixels.

## Video Processing

OpenCV is used to read and process video frames.

Each frame is resized and converted into grayscale before being passed
to the trained model for prediction.

## Project Files

### TrafficSignDetection.py

Main Python application.

It contains:

- Tkinter GUI
- Training-image selection
- Image preprocessing
- CNN/LeNet model creation
- Model training/loading
- Test-video selection
- Traffic-sign detection

### predict.py

Python script used for prediction.

It loads the saved model architecture and weights and processes video
frames to predict traffic-sign classes.

### run.bat

Windows batch file used to start the application.

### SCREENSHOTS.docx

Contains the project documentation and screenshots showing the project
workflow and application output.

### requirements.txt

Contains the Python packages required to run the project.

## How to Run

Install the required Python packages:

    pip install -r requirements.txt

Then run:

    python TrafficSignDetection.py

Alternatively, on Windows:

    run.bat

## Dataset

The project documentation refers to the MASTIF traffic-sign datasets
and describes a dataset containing traffic-sign images divided into
five categories.

The original training dataset is not included in this GitHub repository.

## Model Files

The trained model files are not included in this repository because
the original trained model files were not available with the submitted
project source files.

The prediction script expects the trained model architecture and
weights to be available when prediction is performed.

## Project Limitations

According to the project documentation:

- The demonstrated dataset contains only five traffic-sign types.
- Real-world traffic-sign systems contain more traffic-sign categories.
- Therefore, the project should not be considered a complete
  real-world traffic-sign recognition system.
- The documentation also notes that bounding-box detection was not
  working properly in the demonstrated video output.

## Future Improvements

The project can be extended by:

- Adding more traffic-sign categories.
- Using a larger and more diverse dataset.
- Improving detection accuracy.
- Implementing proper bounding-box detection.
- Developing a web-based frontend.
- Providing a REST API for model prediction.
- Deploying the AI model as a full-stack application.

## Project Type

AI / Machine Learning / Computer Vision

## Author

Traffic Sign Detection and Recognition Project