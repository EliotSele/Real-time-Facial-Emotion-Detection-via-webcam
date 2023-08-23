import cv2
import tensorflow as tf
print(f"Tensorflow version: {tf.__version__}")
import numpy as np
import os
import pathlib

#Change Directory
cascade_path = pathlib.Path(__file__).parent.absolute()
print(f"cascade_path: {cascade_path}")
os.chdir(cascade_path)

# Load the emotion detection model
model = tf.keras.models.load_model('best_model.h5')

# Define the emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define the webcam
cap = cv2.VideoCapture(0)
while True:
    # Capture the video
    ret, frame = cap.read()

    # Convert the video to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces in the video using haarcascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face detected, predict the emotion
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        # roi_gray = roi_gray.reshape(1, 28, 28, 1)
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        prediction = model.predict(roi)[0]
        label = emotions[prediction.argmax()]

        # Draw a rectangle around the face and display the emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting video
    cv2.imshow('Emotion Detection', frame)

    # Quit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
