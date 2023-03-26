from roboflow import Roboflow
from ultralytics import YOLO

# code is base on roboflow titorial for YOLOv8
# https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/

# loading our dataset from roboflow
def load_from_roboflow():
    print("started downloading dataset from roboflow")
    rf = Roboflow(api_key="BQwCglXoth5FSTCCWMZF")
    project = rf.workspace("technion-fl2u0").project("taco-train_test")
    dataset = project.version(1).download("yolov8")
    print("finished downloading dataset from roboflow")
    return dataset

# initializing a YOLOv8 model and running it on our dataset
def run_yolo_model(dataset):
    print("started YOLO")
    model = YOLO('yolov8n.pt') # initializing a pre-trained YOLOv8 model # change to yolov8x.pt
    train_results = model.train(data="/home/muradek/Deep_project/TACO-train_test-1/data.yaml", epochs=25, imgsz=640) # training the model
    val_results = model.val(data="/home/muradek/Deep_project/TACO-train_test-1/data.yaml") # metrics # evaluate model performance on the validation set
    pred_results = model.predict(source="/home/muradek/Deep_project/TACO-train_test-1/test/images")
    print("finished YOLO")
    return train_results, val_results, pred_results
