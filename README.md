# StayInCam


## Notes:

### Approach 1: Haar Cascade Classifier
    The Haar Cascade Classifier is a machine learning based approach for object detection. It's easy to use, as it is a pretrained model from the OpenCV library.

    Pros: Easy to use, computationally fast

    Cons: Not flexible for my usecase. The classifier fails when I change the angle that I look at my webcam even a little. This could result in a ping when I'm still in frame.

### Approch 2: Yolov8

Plan
- create + annotate custom dataset OR use existing face dataset
https://g-vj.medium.com/custom-image-dataset-for-yolo-object-detection-using-open-image-dataset-41c128b9231b
