from autodistill_yolov8 import YOLOv8
from ultralytics import YOLO


def yolo_train():
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training) # for game , YOLOv8n is enough

    results = model.train(data='dataset/data.yaml', epochs=10, imgsz=480, device=0, batch=-1)  # use gpu train  cuda:0


if __name__ == '__main__':
    try:
        yolo_train()
    except Exception as e:
        print('errors:', e)
        pass
