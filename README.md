# Traffic Sign Detection and Recognition

## Project Overview
Traffic Sign Detection and Recognition is an AI/ML-based computer vision project developed using Python.

The project uses a CNN/LeNet-style neural network model to recognize traffic sign classes from video frames.

The application provides a graphical user interface (GUI) through which the user can select training images, generate/load the model, select a test video, and perform traffic sign detection.

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

## Model
The project uses a CNN/LeNet-style neural network architecture. The model contains convolution, pooling, flattening, dense and dropout layers and uses TensorFlow/Keras for training and prediction.

The model is designed to process traffic-sign images resized to 28 x 28 pixels.

## Video Processing
OpenCV is used to read and process video frames. Each frame is resized and converted into grayscale before being passed to the trained model for prediction.

## Project Files

### signalsystem.py
Main Python application containing the Tkinter GUI, training-image selection, image preprocessing, CNN/LeNet model creation, model training/loading, test-video selection, and traffic-sign detection.

### predict.py
Python script used for prediction. It loads the saved model architecture and weights and processes video frames to predict traffic-sign classes.

### run.bat
Windows batch file used to start the application.

### SCREENSHOTS.docx
Contains project documentation and screenshots showing the project workflow and application output.

### train_model.py
Python script for training the CNN/LeNet-style model using the expected five traffic-sign categories (A-E).

### requirements.txt
Contains the Python packages required to run the project.

## How to Run
Install the required packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python signalsystem.py
```

Alternatively, on Windows:

```text
run.bat
```

## Dataset
The project documentation refers to the MASTIF traffic-sign datasets and describes a dataset containing traffic-sign images divided into five categories.

The original training dataset is not included in this GitHub repository.

## Model Files
The trained model files are not included in this repository because the original trained model files were not available with the submitted project source files.

The prediction script expects the trained model architecture and weights to be available when prediction is performed.

## Project Limitations
- The demonstrated dataset contains only five traffic-sign types.
- Real-world traffic-sign systems contain more traffic-sign categories.
- The project should not be considered a complete real-world traffic-sign recognition system.
- The documentation notes that bounding-box detection was not working properly in the demonstrated video output.

## Future Improvements
- Add more traffic-sign categories.
- Use a larger and more diverse dataset.
- Improve detection accuracy.
- Implement proper bounding-box detection.
- Develop a web-based frontend.
- Provide a REST API for model prediction.
- Deploy the AI model as a full-stack application.

## Project Type
AI / Machine Learning / Computer Vision

## Author
Traffic Sign Detection and Recognition Project
