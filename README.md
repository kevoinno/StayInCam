# StayInCam


## Notes:

### Approach 1: Haar Cascade Classifier
    The Haar Cascade Classifier is a machine learning based approach for object detection. It's easy to use, as it is a pretrained model from the OpenCV library.

    Pros: Easy to use, computationally fast

    Cons: Not flexible for my usecase. The classifier fails when I change the angle that I look at my webcam even a little. This could result in a ping when I'm still in frame.

### Approch 2: Yolov8

Plan
- Train data on 300 epochs on AWS
- Load data in (need to hide screen so API key isn't leaked)
- Use human faces dataset from OIDv7
- get YOLOv8 model working with photos, then try to get it to work for video on webcam 
- Train on GPU instead of CPU
