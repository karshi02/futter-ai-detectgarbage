# # from ultralytics import YOLO
# # import cv2

# # # 1. โหลด Model
# # # แนะนำให้โหลดไว้บน GPU (ถ้ามี) โดยเพิ่ม .to('cuda')
# # detector = YOLO("yolov8n.pt") 
# # classifier = YOLO("runs/classify/train/weights/best.pt") 

# # cap = cv2.VideoCapture(0)

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     # 2. Object Detection (ตรวจจับวัตถุ)
# #     # ใช้ stream=True เพื่อประหยัด memory กรณีประมวลผลวิดีโอ
# #     det_results = detector(frame, conf=0.5, verbose=False)

# #     for result in det_results:
# #         for box in result.boxes:
# #             # ดึงพิกัด Bounding Box
# #             x1, y1, x2, y2 = map(int, box.xyxy[0])

# #             # 3. Crop ภาพ
# #             # ตรวจสอบขอบเขตภาพเพื่อไม่ให้ Error ถ้าพิกัดออกนอกเฟรม
# #             crop = frame[max(0, y1):min(frame.shape[0], y2), 
# #                          max(0, x1):min(frame.shape[1], x2)]
            
# #             if crop.size == 0:
# #                 continue

# #             # 4. Classification (จำแนกประเภท)
# #             # YOLOv8 รับภาพจาก OpenCV (BGR) ได้โดยตรง ไม่ต้องแปลงเป็น PIL ก็ได้ครับ
# #             cls_res = classifier(crop, verbose=False)[0]

# #             cls_id = cls_res.probs.top1
# #             name = cls_res.names[cls_id]
# #             conf = cls_res.probs.top1conf.item()

# #             # 5. การแสดงผล
# #             label = f"{name} {conf*100:.1f}%"
            
# #             # วาดกรอบและข้อความ
# #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #             cv2.putText(frame, label, (x1, y1 - 10),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# #     # แสดงผลหน้าจอ
# #     cv2.imshow("YOLOv8 Dual-Stage System", frame)
    
# #     if cv2.waitKey(1) & 0xFF == ord("q"):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()
# from ultralytics import YOLO
# import cv2

# # 1. โหลด Model (พยายามใช้ GPU ถ้าเครื่องคุณมี NVIDIA)
# detector = YOLO("yolov8n.pt") 
# classifier = YOLO("runs/classify/train/weights/best.pt") 

# # ใช้การตั้งค่าเบื้องต้นสำหรับกล้อง
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # กำหนดความกว้างเพื่อไม่ให้หนักเครื่องเกินไป
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# print("กด 'q' เพื่อออกจากโปรแกรม")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 2. Object Detection (ใช้ stream=True เพื่อจัดการ memory ให้ดีขึ้น)
#     # คัดกรองเฉพาะคลาสที่คุณสนใจได้ที่นี่ เช่น classes=[0] ถ้าจะเอาแค่คน
#     det_results = detector(frame, conf=0.5, verbose=False, stream=True)

#     for result in det_results:
#         # ดึง boxes ออกมา
#         boxes = result.boxes
#         for box in boxes:
#             # ดึงพิกัด Bounding Box
#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             # 3. Crop ภาพ (ป้องกันขอบเขตภาพหลุดเฟรม)
#             h, w, _ = frame.shape
#             crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
#             if crop.size == 0:
#                 continue

#             # 4. Classification (ส่งภาพที่ตัดแล้วไปวิเคราะห์)
#             # เราใช้ verbose=False เพื่อไม่ให้ Terminal รก
#             cls_res = classifier(crop, verbose=False)[0]

#             # ตรวจสอบว่ามี probs หรือไม่ (ป้องกัน Error กรณีโมเดลทายไม่ได้)
#             if cls_res.probs is not None:
#                 cls_id = cls_res.probs.top1
#                 name = cls_res.names[cls_id]
#                 conf = cls_res.probs.top1conf.item()

#                 # 5. การแสดงผล (วาดกรอบและชื่อที่ได้จาก Classifier)
#                 label = f"{name} {conf*100:.1f}%"
                
#                 # วาดกรอบสีเขียว
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
#                 # ทำพื้นหลังข้อความให้อ่านง่ายขึ้น
#                 (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#                 cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), (0, 255, 0), -1)
#                 cv2.putText(frame, label, (x1, y1 - 7),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

#     # แสดงผลหน้าจอ
#     cv2.imshow("Detection + Classification System", frame)
    
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2

# 1. โหลด Model
detector = YOLO("yolov8n.pt") 
# classifier = YOLO("runs/classify/train/weights/best.pt") 
classifier = YOLO(r"C:\Users\karsh\Desktop\futter-ai-detectgarbage\waste_sorting\model_v14\weights\best.pt")

cap = cv2.VideoCapture(0)

# สร้างลิสต์ ID สิ่งของที่เราต้องการ (YOLO COCO dataset)
# เช่น 39: ขวด, 41: ถ้วย, 63: โน้ตบุ๊ก, 67: คีย์บอร์ด, 73: หนังสือ
# หรือถ้าอยากได้ "ทุกอย่างยกเว้นคน" (ID 0) ให้ใช้ range(1, 80)
target_classes = list(range(1, 80)) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Object Detection - สั่งให้ข้าม ID 0 (คน) โดยใช้พารามิเตอร์ classes
    det_results = detector(frame, conf=0.5, verbose=False, classes=target_classes)

    for result in det_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 3. Crop ภาพเฉพาะสิ่งของที่เจอ
            h, w, _ = frame.shape
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
            if crop.size == 0:
                continue

            # 4. ส่งไปให้ best.pt (สมองส่วนวิเคราะห์ของคุณ) คิดต่อว่าเป็นอะไร
            cls_res = classifier(crop, verbose=False)[0]

            if cls_res.probs is not None:
                cls_id = cls_res.probs.top1
                name = cls_res.names[cls_id]
                conf = cls_res.probs.top1conf.item()

                label = f"{name} {conf*100:.1f}%"
                
                # วาดกรอบเฉพาะสิ่งของ
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # เปลี่ยนเป็นสีน้ำเงินจะได้ไม่ซ้ำ
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Object Only Mode", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()