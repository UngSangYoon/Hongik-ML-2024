from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    model = YOLO("yolov9c.pt")

    training_model = model.train(
    project='yolov9c_custom',
    data='data/data.yaml',
    imgsz=640,
    save_period=1,
    epochs=2,
    device=0,
    batch=12,
    name='yolov9c_custom',
    plots=True,
    cos_lr=True,
    warmup_epochs=1,
    )
    model.save('./training_model')
    model.val(data='data/data.yaml')