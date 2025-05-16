Facial Emotion Recognition using CNN 😄😢😠
This project is a Facial Emotion Recognition System that uses a Convolutional Neural Network (CNN) to classify emotions from human facial expressions captured via webcam. It is trained using the popular FER2013 dataset and runs live predictions using a webcam stream.

📁 Dataset
The model is trained using the FER2013 dataset, which contains 35,000+ labeled facial expression images categorized into 7 emotions:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

Download dataset here: https://www.kaggle.com/datasets/msambare/fer2013

🧠 Model Overview
CNN model built using TensorFlow / Keras

Trained on train/ images from FER2013

Validated on test/ images

Uses multiple convolutional + pooling layers followed by dense layers

Uses softmax activation for 7-class emotion classification

📸 Live Webcam Emotion Detection
After training, the model is used in detector.py to:

Access webcam feed

Detect and extract face using Haar Cascade

Predict emotion in real time

Display emotion label on screen

🚀 How to Run
Clone this repository or download the project folder.

Install dependencies by running:
pip install -r requirements.txt

To train the model, run:
python model.py

To start emotion detection with webcam, run:
python detector.py

🛠️ Requirements
* Python 3.x
* TensorFlow
* OpenCV
* NumPy
* Keras

You can install everything using the provided requirements.txt.

✨ Author
Syeda Muskan SR
Artificial Intelligence & Data Science Student @ RIT
GitHub: https://github.com/syedsmuskan
