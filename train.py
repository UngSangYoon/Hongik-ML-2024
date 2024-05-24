from ultralytics import YOLO
from datasets import load_dataset

model = YOLO("yolov9c.pt")

training_model = model.train(data = '/Users/yun-ungsang/Desktop/4-1/기계학습심화/project/data/data.yaml', epochs=1, imgsz=1024)
training_model.save('./training_model')