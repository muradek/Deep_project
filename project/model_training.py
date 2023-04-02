from roboflow import Roboflow
from ultralytics import YOLO

# code is base on roboflow titorial for YOLOv8
# https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/

# loading our datasets from roboflow
def load_from_roboflow():
    print("started downloading dataset from roboflow")
    rf = Roboflow(api_key="BQwCglXoth5FSTCCWMZF")
    project = rf.workspace("technion-fl2u0").project("taco-train_test")
    dataset = project.version(1).download("yolov8")
    print("finished downloading dataset from roboflow")
    return dataset

def load_datasets():
    rf = Roboflow(api_key="BQwCglXoth5FSTCCWMZF")

    train_project = rf.workspace("technion-fl2u0").project("taco_train_only")
    train_set = train_project.version(1).download("yolov8")

    test_project = rf.workspace("technion-fl2u0").project("taco_test_set")
    test_set = test_project.version(1).download("yolov8")

    return train_set, test_set


# initializing a YOLOv8 model and custom train on our dataset
def set_model(dataset, model_version):
    print("started setting" + model_version)
    model = YOLO(model_version) # initializing a pre-trained YOLOv8 model
    data = str(dataset.location) + "/data.yaml"
    train_results = model.train(data=data, epochs=10, imgsz=640, patience=5, batch=2) # training the model
    print("finished YOLO setting")
    return model, train_results

def evaluate_model(dataset, model):
    print("started evaluating")
    data = str(dataset.location) + "/data.yaml"
    val_results = model.val(data=data, save_json=True) # metrics # evaluate model performance on the validation set
    print("finished evaluating")
    return val_results

def pred(model, dataset):
    print("started testing")
    source = str(dataset.location) + "/test/images"
    pred_results = model.predict(source=source, save=True, save_json=True)
    print("finished predicting")
    return pred_results