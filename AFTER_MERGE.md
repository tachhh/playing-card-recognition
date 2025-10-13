# 🎉 Merge เสร็จแล้ว! ทำอะไรต่อ?

## ✅ สถานะปัจจุบัน

คุณได้ merge dataset เดิม + dataset ใหม่เรียบร้อยแล้ว! 
ไฟล์อยู่ที่: `data/merged_dataset/`

```
merged_dataset/
├── train/          (รวมรูปจาก 2 datasets)
├── valid/          (แบ่งอัตโนมัติ)
└── dataset_info.json
```

---

## 🎯 ขั้นตอนต่อไป (3 ขั้นตอน)

### 1️⃣ ✅ แก้ไข train_cnn_model.py (เสร็จแล้ว!)

ไฟล์ `train_cnn_model.py` ถูกแก้ไขให้ใช้ merged dataset อัตโนมัติแล้ว!

### 2️⃣ เทรนโมเดล (ทำตอนนี้!)

**Double-click**: `run_training.bat`

หรือรันคำสั่ง:
```bash
python train_cnn_model.py
```

**ระยะเวลา:** 1-2 ชั่วโมง (30 epochs บน CPU)

**คาดการณ์:**
- Accuracy เดิม: 81.89%
- Accuracy ใหม่: 85-92% (ขึ้นกับข้อมูลที่เพิ่ม)

### 3️⃣ ทดสอบโมเดล

หลังเทรนเสร็จ ให้รัน:

**Double-click**: `run_testing.bat` (จะสร้างให้ด้านล่าง)

หรือ
```bash
python test_cnn_model.py
```

---

## 📊 ดูข้อมูล Merged Dataset

รันคำสั่ง:
```bash
python convert_dataset.py
```

เลือก option 1: "Analyze existing dataset"
ใส่ path: `data/merged_dataset`

จะแสดง:
- จำนวน classes
- จำนวนรูปต่อ class
- การกระจายของข้อมูล

---

## 🎮 ขั้นตอนทั้งหมด (Quick Reference)

```
[✅] Download original dataset
[✅] Place new dataset
[✅] Merge datasets
[✅] Update train_cnn_model.py
[ ] Train model              <- คุณอยู่ตรงนี้!
[ ] Test model
[ ] Use with camera
```

---

## 🚀 เริ่มเทรนเลย!

**Double-click**: `run_training.bat`

หรือ

```bash
python train_cnn_model.py
```

---

## 📝 หลังเทรนเสร็จ

โมเดลใหม่จะถูกบันทึกที่:
- `models/card_classifier_cnn.pth` (โมเดลใหม่)
- `models/card_classifier_cnn_full.pth` (checkpoint)
- `models/training_history_cnn_YYYYMMDD_HHMMSS.json` (ประวัติ)

---

## 🎯 ทดสอบโมเดล

```bash
python test_cnn_model.py
```

เปรียบเทียบ accuracy:
- โมเดลเดิม: 81.89%
- โมเดลใหม่: ??.??%

---

## 📷 ใช้กับกล้อง

หลังเทรนแล้ว ไม่ต้องทำอะไรเพิ่ม!

**Double-click**: `run_camera.bat`

ระบบจะใช้โมเดลใหม่ที่มี accuracy สูงกว่าอัตโนมัติ! 🎉

---

## ⏸️ ถ้าต้องการหยุดกลางคัน

กด **Ctrl+C** ใน terminal

โมเดลที่ดีที่สุดจะถูกเซฟไว้แล้ว!

---

## 💡 Tips

### ✅ เทรนต่อได้
ถ้า accuracy ยังไม่พอ สามารถรัน:
```bash
python continue_training.py
```

### ✅ เปรียบเทียบโมเดล
เก็บโมเดลเดิมไว้:
```bash
copy models\card_classifier_cnn.pth models\card_classifier_cnn_old.pth
```

### ✅ Backup
ก่อนเทรน backup โมเดลเดิม:
- `models/card_classifier_cnn.pth` → `models/backup/`

---

## 🎉 พร้อมแล้ว!

**ขั้นตอนถัดไป:** Double-click `run_training.bat` เพื่อเริ่มเทรน! 🚀

---

Made with ❤️ for Playing Card Recognition Project
