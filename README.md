# Blood Cell Detection

## Overview
This Streamlit application allows you to detect and classify blood cells in images using a YOLOv5-based model. The model is trained to identify and locate red blood cells (RBCs), white blood cells (WBCs), and platelets in medical images.

## Features
- Upload an image in JPG, JPEG, or PNG format.
- Detects RBCs, WBCs, and platelets with adjustable confidence thresholds.
- Displays the processed image with bounding boxes and labels.

## Requirements
- Python 3.10
- Streamlit
- OpenCV
- NumPy

## Installation
Follow these steps to set up and run the project:

1. **Clone the repository:**  
   `git clone https://github.com/AIOnGraph/blood-cell-detection.git`  
   `cd blood-cell-detection`

2. **Create a virtual environment:**  
   `python -m venv venv`  

   To activate the virtual environment:
   - On Windows:  
     `venv\Scripts\activate`  
   - On macOS/Linux:  
     `source venv/bin/activate`  

3. **Install required packages:**  
   `pip install -r requirements.txt`  

4. **Run the application:**  
   `streamlit run app.py`  

5. **For the demo link:**  
   [https://blood-cell-detection.streamlit.app/](https://blood-cell-detection.streamlit.app/)
