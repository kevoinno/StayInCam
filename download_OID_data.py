from roboflow import Roboflow
from creds import ROBOFLOW_API_KEY

# load in dataset
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("stayincam").project("stayincam")
version = project.version(1)
dataset = version.download("yolov8")
                