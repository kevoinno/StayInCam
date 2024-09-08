import os
import numpy as np
import cv2

# Load haar-cascal XML classifier file
haar_cascade = cv2.CascadeClassifier(os.path.join('.', 'haarcascade_frontalface_default.xml')) 

# read webcam
webcam = cv2.VideoCapture(0)

# visualize webcam
while True:
    ret, frame = webcam.read()
    
    # convert image to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # use classifier to detect face
    faces_rect = haar_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=6)

    # draw borders around face on the original frame
    for (x, y ,w, h) in faces_rect:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
