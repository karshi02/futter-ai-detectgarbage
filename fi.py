from ultralytics import YOLO

model = YOLO("runs/classify/train/weights/best.pt")

model.export(format="tflite")
