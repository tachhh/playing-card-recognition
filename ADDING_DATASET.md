# 📚 Adding New Dataset Guide

## วิธีเพิ่ม Dataset ใหม่เพื่อปรับปรุง Accuracy

### ✅ ขั้นตอนที่ 1: หา Dataset เพิ่ม

**แหล่ง Dataset แนะนำ:**
1. **Kaggle** - https://www.kaggle.com/datasets
   - ค้นหา: "playing cards", "card detection", "card classification"
2. **Roboflow** - https://universe.roboflow.com/
   - มี dataset สำเร็จรูปพร้อม annotation
3. **Google Dataset Search** - https://datasetsearch.research.google.com/
4. **สร้างเอง** - ถ่ายภาพไพ่เพิ่ม (แนะนำ 50-100 ภาพต่อไพ่)

---

### ✅ ขั้นตอนที่ 2: ตรวจสอบ Dataset Format

รันคำสั่ง:
```bash
python convert_dataset.py
```

เลือก option 4: "Detect dataset format"

**Format ที่รองรับ:**
- ✅ Class folders: `/class_name/image.jpg`
- ✅ Train/Val split: `/train/class_name/image.jpg`
- ✅ Flat structure: `/ace_of_hearts_001.jpg`

---

### ✅ ขั้นตอนที่ 3: แปลง Format (ถ้าจำเป็น)

ถ้า format ไม่ตรง ให้รัน:
```bash
python convert_dataset.py
```

เลือก option 3: "Convert flat structure to class folders"

---

### ✅ ขั้นตอนที่ 4: รวม Datasets

รันคำสั่ง:
```bash
python train_with_new_dataset.py
```

หรือใช้ converter โดยตรง:
```bash
python convert_dataset.py
```
เลือก option 2: "Merge two datasets"

สคริปต์จะ:
- รวม dataset เดิม + dataset ใหม่
- แบ่ง train/valid อัตโนมัติ (85/15)
- สร้างโฟลเดอร์: `data/merged_dataset/`

---

### ✅ ขั้นตอนที่ 5: เทรนโมเดลใหม่

**แก้ไข `train_cnn_model.py`:**

หา code บรรทัดนี้:
```python
with open(ref_file, 'r') as f:
    dataset_path = f.read().strip()
```

เปลี่ยนเป็น:
```python
# Use merged dataset
dataset_path = r'C:\playing-card-recognition\playing-card-recognition\data\merged_dataset'
```

**รันคำสั่ง:**
```bash
python train_cnn_model.py
```

---

### ✅ ขั้นตอนที่ 6: ประเมินผล

หลังเทรนเสร็จ ให้รัน:
```bash
python test_cnn_model.py
```

เปรียบเทียบ accuracy:
- **เดิม**: 81.89%
- **ใหม่**: ???

---

## 📊 Tips สำหรับ Dataset ที่ดี

### ✅ Quantity (ปริมาณ)
- มากกว่า 100 ภาพต่อ class
- ยิ่งมากยิ่งดี (500-1000 ภาพ = excellent)

### ✅ Quality (คุณภาพ)
- ภาพชัด ไม่เบลอ
- แสงสว่างพอดี
- มุมมองหลากหลาย

### ✅ Variety (ความหลากหลาย)
- หลายพื้นหลัง (โต๊ะ, ผ้า, มือ)
- หลายมุมกล้อง (ตรง, เอียง, ไกล, ใกล้)
- หลายสภาพแสง (สว่าง, มืด, ธรรมชาติ, ไฟห้อง)

### ✅ Balance (ความสมดุล)
- แต่ละ class มีจำนวนใกล้เคียงกัน
- ไม่ควรต่างกันเกิน 2 เท่า

---

## 🎯 Expected Improvements

| Dataset Size | Expected Accuracy |
|--------------|-------------------|
| Current (8K) | 81.89% |
| +5K images   | 85-88% |
| +10K images  | 88-92% |
| +20K images  | 92-95% |

---

## ⚠️ Common Issues

### ❌ Class names ไม่ตรงกัน
**แก้:** ใช้ `convert_dataset.py` แปลงให้ตรงกับ 53 classes

### ❌ Image format ต่างกัน (.jpg, .png)
**แก้:** Converter รองรับทั้ง jpg, png, jpeg อัตโนมัติ

### ❌ Resolution ต่างกัน
**แก้:** โมเดล resize เป็น 224x224 อัตโนมัติ

### ❌ Duplicate images
**แก้:** ใช้ image hashing เช็คก่อนเทรน (advanced)

---

## 🚀 Quick Start Example

```bash
# 1. Download new dataset from Kaggle
kagglehub download username/card-dataset

# 2. Analyze format
python convert_dataset.py
# Select: 4

# 3. Merge datasets
python train_with_new_dataset.py
# Enter new dataset path

# 4. Train model
python train_cnn_model.py

# 5. Test model
python test_cnn_model.py

# 6. Use with camera
python camera_simple.py
```

---

## 📞 Need Help?

ถ้ามีปัญหา:
1. ตรวจสอบ format ของ dataset ใหม่ด้วย `convert_dataset.py`
2. ดู error message ใน console
3. ตรวจสอบว่า class names ตรงกับ 53 classes หรือไม่

---

Made with ❤️ for Playing Card Recognition Project
