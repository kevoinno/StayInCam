# StayInCam


## Notes:

### Approach 1: Haar Cascade Classifier
    The Haar Cascade Classifier is a machine learning based approach for object detection. It's easy to use, as it is a pretrained model from the OpenCV library.

    Pros: Easy to use, computationally fast

    Cons: Not flexible for my usecase. The classifier fails when I change the angle that I look at my webcam even a little. This could result in a ping when I'm still in frame.

### Approch 2: Yolov8

Plan
- create yaml file for google colab
- create notebook to train model on VS code
- push to github
- clone on colab and run
- https://www.youtube.com/watch?v=m9fH9OWn8YM&t=3304s Train YOLO model on google colab (46:21)
