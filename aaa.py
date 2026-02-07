import os
import shutil
import random
from ultralytics import YOLO

# --- 1. ตั้งค่า Path (เช็คให้ตรงกับเครื่องนาย) ---
root_path = r"C:\Users\karsh\Desktop\futter-ai-detectgarbage"
data_path = os.path.join(root_path, "data_split") # โฟลเดอร์ใหม่ที่จะสร้าง

# รายชื่อโฟลเดอร์ขยะที่นายมี
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# --- 2. ฟังก์ชันแบ่งไฟล์ (Auto-Split) ---
def split_data():
    for cls in classes:
        # สร้างโฟลเดอร์ปลายทาง
        os.makedirs(os.path.join(data_path, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(data_path, 'val', cls), exist_ok=True)
        
        # ดึงรายชื่อไฟล์ภาพจากโฟลเดอร์เดิม
        src_dir = os.path.join(root_path, 'val', cls) # อ้างอิงจากพาธที่นายส่งมาล่าสุด
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # สุ่มลำดับไฟล์
        random.shuffle(files)
        split_idx = int(len(files) * 0.8) # แบ่ง 80%
        
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        # ก๊อปปี้ไฟล์ไปที่ใหม่
        for f in train_files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(data_path, 'train', cls, f))
        for f in val_files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(data_path, 'val', cls, f))
        
        print(f"✅ {cls}: แบ่งเสร็จแล้ว (Train: {len(train_files)}, Val: {len(val_files)})")

# --- 3. เริ่มทำงาน ---
if __name__ == '__main__':
    print("🚀 กำลังเตรียมข้อมูล...")
    split_data()
    
    print("\n🧠 เริ่มการเทรน AI...")
    # โหลดโมเดล Small (เก่งกว่า Nano แต่ยังไว)
    model = YOLO('yolov8s-cls.pt') 
    
    results = model.train(
        data=data_path,
        epochs=30,      # ถ้ารีบ 30 รอบก็เริ่มเห็นผลแล้วครับ
        imgsz=224,      # ขนาดภาพสำหรับแยกประเภท
        batch=16,       # ปรับตาม RAM การ์ดจอ
        device='cpu',       # ใช้ GPU (ถ้าไม่มีใส่ 'cpu')
        project='waste_sorting',
        name='model_v1',
        patience=5      # ถ้าไม่ดีขึ้น 5 รอบ ให้หยุดเทรนทันที (เซฟเวลา!)
    )
    
    print(f"🎉 เสร็จแล้ว! ไฟล์โมเดลที่ดีที่สุดอยู่ที่: waste_sorting/model_v1/weights/best.pt")