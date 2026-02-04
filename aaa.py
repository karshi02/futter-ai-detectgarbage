from ultralytics import YOLO

model = YOLO("yolov8m-cls.pt")

model.train(
    data=r"C:\Users\en-rm\Downloads\dataset-resized",
    epochs=50,
    imgsz=224,
    batch=16,      # CPU แนะนำ 16 หรือ 8
    device="cpu"   # ⭐ สำคัญ
)
