import os
import random
import shutil

BASE_DIR = r"C:\Users\en-rm\Downloads\dataset-resized"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
TRAIN_RATIO = 0.8

for cls in CLASSES:
    src_dir = os.path.join(BASE_DIR, cls)
    images = os.listdir(src_dir)
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for img in train_imgs:
        shutil.move(
            os.path.join(src_dir, img),
            os.path.join(BASE_DIR, "train", cls, img)
        )

    for img in val_imgs:
        shutil.move(
            os.path.join(src_dir, img),
            os.path.join(BASE_DIR, "val", cls, img)
        )

    # ลบโฟลเดอร์เดิม (ถ้าว่าง)
    if not os.listdir(src_dir):
        os.rmdir(src_dir)

print("✅ Split dataset สำเร็จแล้ว (80% train / 20% val)")
