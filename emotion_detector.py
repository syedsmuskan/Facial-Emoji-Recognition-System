import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("emotion_model.h5")

# Emotion classes (same order as in FER2013)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Nutral', 'Surprise', 'Sad']

# Start webcam
cap = cv2.VideoCapture(0)

# Load Haar Cascade face detector from OpenCV
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict emotion
            prediction = model.predict(roi)[0]
            label = emotion_labels[np.argmax(prediction)]
            label_position = (x, y - 10)

            # Display results
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Emotion Detector - Press Q to exit', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
