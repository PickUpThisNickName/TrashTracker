from ultralytics import YOLO

#for training
model = YOLO("yolov8n.yaml")
results = model.train(data="config.yaml", epochs=200)
