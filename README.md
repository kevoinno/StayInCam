# StayInCam


## Notes:

### Approach 1: Haar Cascade Classifier
    The Haar Cascade Classifier is a machine learning based approach for object detection. It's easy to use, as it is a pretrained model from the OpenCV library.

    Pros: Easy to use, computationally fast

    Cons: Not flexible for my usecase. The classifier fails when I change the angle that I look at my webcam even a little. This could result in a ping when I'm still in frame.

### Approach 2: YOLOv8 object detection
    Pros: Model can detect face at different angles
    Cons: Detection is "too strong", model can detect face when only half of face is present. This doesn't help our use case, since half our face can be out of frame, and the model still detects it.

    To work around this. We can train the model to also detect facial features such as eyes and mouth. If both eyes and the mouth are present, then we can consider the face in frame.
    
TODO:

Notes:

No need for ECR

- Need to unzip data
- Need to specify paths

1. Start a notebook instance with ml.p2.xlarge instance

2. Train model and dump results in to S3


