from ultralytics import YOLO
import cv2
import os
import math 

# Load model with the weights
WEIGHTS_PATH = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'best.pt')
model = YOLO(WEIGHTS_PATH)

# read webcam
webcam = cv2.VideoCapture(0) # specify the # of the webcam that you want to access
webcam.set(3, 640)
webcam.set(4, 480)

# specify object classes
classNames = ['Human Face']

# visualize webcam
while True:
    ret, frame = webcam.read()
    # run the model on the frame
    results = model(frame, stream = True)

    # draw bounding boxes
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            #confidence = math.ceil((box.conf[0]*100))/100
            #print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            #print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0, 0, 255)
            thickness = 2

            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

webcam.release()
cv2.destroyAllWindows()