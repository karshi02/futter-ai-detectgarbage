# # # # from ultralytics import YOLO
# # # # from PIL import Image
# # # # import matplotlib.pyplot as plt

# # # # # โหลดโมเดลที่เทรนเสร็จแล้ว
# # # # model = YOLO("runs/classify/train/weights/best.pt")

# # # # # โหลดรูป
# # # # img_path = "test.jpg"   # 🔁 เปลี่ยนเป็น path รูปของคุณ
# # # # img = Image.open(img_path)

# # # # # predict
# # # # results = model(img)

# # # # # ดึงผลลัพธ์
# # # # r = results[0]
# # # # cls_id = r.probs.top1
# # # # cls_name = r.names[cls_id]
# # # # confidence = r.probs.top1conf.item()

# # # # # แสดงผล
# # # # plt.imshow(img)
# # # # plt.axis("off")
# # # # plt.title(f"Prediction: {cls_name} ({confidence*100:.2f}%)")
# # # # plt.show()

# # # # print(f"🧠 ผลทำนาย: {cls_name}")
# # # # print(f"🎯 ความมั่นใจ: {confidence*100:.2f}%")
# # # # from ultralytics import YOLO
# # # # from PIL import Image
# # # # import matplotlib.pyplot as plt

# # # # # โหลดโมเดลที่เทรนเสร็จแล้ว
# # # # model = YOLO("runs/classify/train/weights/best.pt")

# # # # # โหลดรูป
# # # # img_path = "/img/glass"   # 🔁 เปลี่ยนเป็น path รูปของคุณ
# # # # img = Image.open(img_path)

# # # # # predict
# # # # results = model(img)

# # # # # ดึงผลลัพธ์
# # # # r = results[0]
# # # # cls_id = r.probs.top1
# # # # cls_name = r.names[cls_id]
# # # # confidence = r.probs.top1conf.item()

# # # # # แสดงผล
# # # # plt.imshow(img)
# # # # plt.axis("off")
# # # # plt.title(f"Prediction: {cls_name} ({confidence*100:.2f}%)")
# # # # plt.show()

# # # # print(f"🧠 ผลทำนาย: {cls_name}")
# # # # print(f"🎯 ความมั่นใจ: {confidence*100:.2f}%")
# # # from ultralytics import YOLO
# # # from PIL import Image
# # # import matplotlib.pyplot as plt
# # # import os
# # # import random

# # # # โหลดโมเดล
# # # model = YOLO("runs/classify/train/weights/best.pt")

# # # # โฟลเดอร์ที่มีรูป (glass)
# # # image_dir = r"C:\Users\en-rm\Downloads\dataset-resized\val\glass"
# # # # ถ้าอยากสุ่มจาก train ก็เปลี่ยนเป็น train\glass

# # # # สุ่มเลือกรูป 1 รูป
# # # img_name = random.choice(os.listdir(image_dir))
# # # img_path = os.path.join(image_dir, img_name)

# # # # เปิดรูป
# # # img = Image.open(img_path)

# # # # predict
# # # results = model(img)
# # # r = results[0]

# # # cls_id = r.probs.top1
# # # cls_name = r.names[cls_id]
# # # confidence = r.probs.top1conf.item()

# # # # แสดงผล
# # # plt.imshow(img)
# # # plt.axis("off")
# # # plt.title(f"Prediction: {cls_name} ({confidence*100:.2f}%)")
# # # plt.show()

# # # print(f"📂 ไฟล์: {img_name}")
# # # print(f"🧠 ทำนายว่า: {cls_name}")
# # # print(f"🎯 ความมั่นใจ: {confidence*100:.2f}%")
# # from ultralytics import YOLO
# # from PIL import Image
# # import matplotlib.pyplot as plt
# # import os
# # import random

# # # โหลดโมเดล
# # model = YOLO("runs/classify/train/weights/best.pt")

# # # path dataset
# # BASE_DIR = r"C:\Users\en-rm\Downloads\dataset-resized\val"
# # CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# # # รวมรูปทั้งหมดจากทุก class
# # all_images = []
# # for cls in CLASSES:
# #     cls_dir = os.path.join(BASE_DIR, cls)
# #     for img in os.listdir(cls_dir):
# #         all_images.append((cls, os.path.join(cls_dir, img)))

# # print(f"📸 พบรูปทั้งหมด: {len(all_images)} รูป")

# # # สุ่มดูรูปไปเรื่อย ๆ
# # random.shuffle(all_images)

# # for true_cls, img_path in all_images:
# #     img = Image.open(img_path)

# #     results = model(img)
# #     r = results[0]

# #     pred_cls = r.names[r.probs.top1]
# #     conf = r.probs.top1conf.item()

# #     plt.imshow(img)
# #     plt.axis("off")
# #     plt.title(
# #         f"GT: {true_cls} | Pred: {pred_cls} ({conf*100:.2f}%)"
# #     )
# #     plt.show()

# #     print(f"📂 ไฟล์: {os.path.basename(img_path)}")
# #     print(f"✅ จริง: {true_cls}")
# #     print(f"🧠 ทำนาย: {pred_cls}")
# #     print(f"🎯 ความมั่นใจ: {conf*100:.2f}%")
# #     print("-" * 40)

# #     input("กด Enter เพื่อดูรูปถัดไป (Ctrl+C เพื่อออก)")
# from ultralytics import YOLO
# from PIL import Image
# import matplotlib.pyplot as plt
# import os
# import random
# import time

# # โหลดโมเดล
# model = YOLO("runs/classify/train/weights/best.pt")

# BASE_DIR = r"C:\Users\en-rm\Downloads\dataset-resized\val"
# CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# # รวมรูปทั้งหมด
# all_images = []
# for cls in CLASSES:
#     cls_dir = os.path.join(BASE_DIR, cls)
#     for img in os.listdir(cls_dir):
#         all_images.append((cls, os.path.join(cls_dir, img)))

# print(f"📸 พบรูปทั้งหมด: {len(all_images)} รูป")

# # เปิดโหมด interactive
# plt.ion()
# fig, ax = plt.subplots()

# random.shuffle(all_images)

# for true_cls, img_path in all_images:
#     img = Image.open(img_path)

#     results = model(img)ฟ
#     r = results[0]

#     pred_cls = r.names[r.probs.top1]
#     conf = r.probs.top1conf.item()

#     ax.clear()
#     ax.imshow(img)
#     ax.axis("off")
#     ax.set_title(
#         f"GT: {true_cls} | Pred: {pred_cls} ({conf*100:.2f}%)"
#     )

#     plt.draw()
#     plt.pause(0.2)   # ⭐ เปลี่ยนรูปทุก 0.2 วินาที

# plt.ioff()
# plt.show()
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

# โหลดโมเดล
model = YOLO("runs/classify/train/weights/best.pt")

BASE_DIR = r"C:\Users\en-rm\Downloads\dataset-resized\val"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# รวมรูปทั้งหมด
all_images = []
for cls in CLASSES:
    cls_dir = os.path.join(BASE_DIR, cls)
    for img in os.listdir(cls_dir):
        all_images.append((cls, os.path.join(cls_dir, img)))

print(f"📸 พบรูปทั้งหมด: {len(all_images)} รูป")

plt.ion()
fig, ax = plt.subplots()

random.shuffle(all_images)

for true_cls, img_path in all_images:
    img = Image.open(img_path)

    results = model(img)
    r = results[0]

    pred_id = r.probs.top1
    pred_name = r.names[pred_id]
    conf = r.probs.top1conf.item()

    # แสดงบนภาพ
    ax.clear()
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        f"เจอ: {pred_name} | ความมั่นใจ: {conf*100:.2f}%"
    )

    # แสดงใน terminal
    print(
        f"📂 {os.path.basename(img_path)} "
        f"=> เจอ: {pred_name} ({conf*100:.2f}%)"
    )

    plt.draw()
    plt.pause(0.2)   # ⭐ เปลี่ยนรูปทุก 0.2 วิ

plt.ioff()
plt.show()
