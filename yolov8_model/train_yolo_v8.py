from ultralytics import YOLO
import os
import torch

YAML_PATH = os.path.join("dataset_yolo_v8", "data.yaml")

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from YAML

# Train the model
results = model.train(data=YAML_PATH, epochs=3, imgsz=640)