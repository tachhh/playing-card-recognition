# 🔧 แก้ปัญหา Confidence ต่ำ (<10%) จากกล้อง

## 🎯 สาเหตุที่พบ

จากการ Diagnostic พบว่า **Model มีปัญหาร้ายแรง**:

```
❌ ⚠️  WARNING: All probabilities are very low!
   Max confidence: 3.24%
   Top 5: 3.24%, 3.13%, 3.05%, 2.89%, 2.71%
```

**สาเหตุหลัก**: Model ทำนายด้วย confidence เกือบเท่ากันทุก class (~3% ต่อ class จาก 53 classes)
- ถ้า model สุ่มทาย → แต่ละ class จะได้ 1/53 = 1.89%
- Model นี้ให้ ~3% → หมายความว่า **แทบจะเท่ากับการสุ่มทาย**

---

## 📊 สรุปสาเหตุที่เป็นไปได้ (เรียงตามความสำคัญ)

### ⭐ อันดับ 1: **Model ไม่ได้ฝึกจริง หรือ โหลดผิด**

**อาการ**:
- Logits variance ต่ำ (std=0.346)
- Probabilities กระจายเกือบเท่ากันทุก class
- Max confidence ~3% (ใกล้เคียง 1/53 = 1.89%)

**วิธีแก้**:
```bash
# 1. ตรวจสอบว่า model ฝึกจบจริงหรือไม่
python test_cnn_model.py
# ถ้าได้ accuracy <20% → model ไม่ได้ฝึกจริง

# 2. ฝึก model ใหม่
python train_cnn_model.py
# ต้องได้ accuracy >70% ถึงจะใช้งานได้

# 3. ตรวจสอบว่า model file ไม่เสีย
# ดูขนาด models/card_classifier_cnn.pth ต้อง >100 MB
```

---

### ⭐ อันดับ 2: **Preprocessing ไม่ตรงกับตอนฝึก**

**การตรวจสอบ**: ✅ **ผ่านแล้ว** - Transforms ตรงกัน
```python
# Training & Inference ใช้ transforms เดียวกัน:
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])
```

---

### ⭐ อันดับ 3: **Class Mapping ผิด**

**การตรวจสอบ**: ✅ **ผ่านแล้ว**
- มี 53 classes ครบ
- Index 0-52 ไม่มีหาย
- Mapping ถูกต้อง

---

### ⭐ อันดับ 4: **Domain Gap (ภาพกล้อง ≠ Training Data)**

**การตรวจสอบ**: ⚠️ **ไม่สามารถทดสอบได้** (ไม่มี training data ในเครื่อง)

**อาการ**:
- Model ทำงานดีบน training set แต่แย่บนกล้อง
- แสง, มุม, พื้นหลัง, สีต่างจาก training data

**วิธีแก้**:
```bash
# ใช้ data augmentation มากขึ้นตอนฝึก
# แก้ไขใน train_cnn_model.py:
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),  # เพิ่มจาก 10 → 20
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # เพิ่ม
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # เพิ่ม
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])
```

---

### ⭐ อันดับ 5: **Camera เป็น Noise/Static**

**การตรวจสอบ**: ✅ **ผ่านแล้ว**
- Frame statistics normal (std=69.42, mean=97.90)
- ไม่ใช่ noise (noise จะมี std>60 และ mean 100-150 ทั่วทั้งเฟรม)

---

### ⭐ อันดับ 6: **Overfitting/Underfitting**

**การตรวจสอบ**: ⏸️ **รอผลการทดสอบ**

**วิธีดู**:
```bash
# ดู training history
python test_cnn_model.py
# ถ้า train_acc สูง แต่ valid_acc ต่ำ → Overfitting
# ถ้าทั้ง train_acc และ valid_acc ต่ำ → Underfitting
```

---

## 🔨 **วิธีแก้ปัญหาแบบเร่งด่วน**

### ขั้นตอนที่ 1: ทดสอบว่า Model ทำงานได้จริงหรือไม่

```bash
# 1. ตรวจสอบว่ามี dataset อยู่หรือไม่
cd data
dir

# ถ้าไม่มี → ต้อง download ก่อน
cd ..
python download_dataset.py

# 2. ทดสอบ model
python test_cnn_model.py
```

**ผลที่คาดหวัง**:
- ✅ Test Accuracy > 70% → Model ใช้งานได้
- ❌ Test Accuracy < 20% → **Model ไม่ได้ฝึกจริง ต้องฝึกใหม่**

---

### ขั้นตอนที่ 2: ถ้า Model ไม่ดี → ฝึกใหม่

```bash
# ฝึก model ใหม่ (30 epochs, ใช้เวลา ~30-60 นาที)
python train_cnn_model.py

# หรือใช้ quick test (5 epochs, ใช้เวลา ~5-10 นาที)
python test_training_quick.py
```

**ผลที่คาดหวัง**:
- Epoch 10: ~60-70% accuracy
- Epoch 20: ~75-85% accuracy
- Epoch 30: ~80-90% accuracy

---

### ขั้นตอนที่ 3: ทดสอบใหม่ด้วยกล้อง

```bash
# ทดสอบกล้อง
python camera_simple.py
# หรือ
scripts\run_camera.bat
```

**ผลที่คาดหวัง**:
- ✅ Confidence > 70% เมื่อไพ่อยู่ในกรอบชัดเจน
- ⚠️ Confidence 30-70% เมื่อไพ่มุมเอียง/แสงไม่ดี
- ❌ Confidence < 30% → ยังมีปัญหาอื่น

---

## 🧪 การวิเคราะห์เพิ่มเติม

### ตรวจสอบ Model File

```bash
# ใน PowerShell:
dir models\card_classifier_cnn.pth

# ตรวจสอบขนาด:
# ✅ ต้อง > 100 MB (ประมาณ 100-105 MB)
# ❌ ถ้า < 10 MB → ไฟล์เสียหรือยังไม่ได้ฝึก
```

### ตรวจสอบ Training History

```bash
dir models\training_history*.json
# ดูไฟล์ล่าสุด

# ดูใน JSON จะเห็น:
# - train_acc: [10.5, 25.3, 45.2, ..., 92.1]  # ต้องเพิ่มขึ้นเรื่อยๆ
# - valid_acc: [12.1, 28.5, 48.3, ..., 81.9]  # ต้องเพิ่มขึ้นเรื่อยๆ
# - best_valid_acc: 81.89  # อยากได้ > 70%
```

---

## 🎓 คำอธิบายทางเทคนิค

### ทำไม Model ถึงให้ Confidence เท่ากันทุก Class?

**Logits Analysis**:
```
Input (random) → Model → Logits
                      ↓
                min=-0.966, max=0.311, mean=-0.286, std=0.346
                      ↓
                  Softmax
                      ↓
                Max prob = 3.24%
```

**ปัญหา**: Logits มี variance ต่ำมาก (std=0.346)
- Logits ควรมี std > 1.0 เพื่อให้ softmax แยก class ได้ชัดเจน
- std=0.346 → Logits ใกล้เคียงกันทุก class → Softmax กระจายเท่าๆ กัน

**สาเหตุที่เป็นไปได้**:
1. **Model ยังไม่ได้ฝึก** - Weights เป็น random initialization
2. **Model โหลดผิด** - State dict ไม่ match กับ architecture
3. **Training fail** - Loss ไม่ลดลง, gradient vanishing/exploding

---

### BGR vs RGB Problem?

**การตรวจสอบ**: ✅ **ไม่มีปัญหา**
- Camera: BGR → RGB conversion ถูกต้อง (`cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`)
- Training: ใช้ PIL (โหลดเป็น RGB อยู่แล้ว)

---

## 📝 Checklist การแก้ปัญหา

- [ ] 1. ตรวจสอบ `test_cnn_model.py` ว่า accuracy > 70%
- [ ] 2. ถ้า accuracy < 20% → ฝึก model ใหม่ด้วย `train_cnn_model.py`
- [ ] 3. รอ training เสร็จ (30 epochs, ~30-60 นาที)
- [ ] 4. ตรวจสอบ `models/card_classifier_cnn.pth` ขนาด > 100 MB
- [ ] 5. ทดสอบกล้องอีกครั้งด้วย `camera_simple.py`
- [ ] 6. ถ้ายังไม่ดี → เพิ่ม data augmentation แล้วฝึกใหม่
- [ ] 7. ถ้ายังไม่ดี → ถ่ายรูปไพ่ด้วยกล้องแล้วเพิ่มเข้า dataset

---

## 🚀 คำสั่งด่วน

```bash
# ขั้นตอนที่ 1: ทดสอบ
python test_cnn_model.py

# ถ้า accuracy < 20%:
# ขั้นตอนที่ 2: ฝึกใหม่
python train_cnn_model.py

# ขั้นตอนที่ 3: ทดสอบกล้อง
python camera_simple.py
```

---

## 📞 ถ้ายังแก้ไม่ได้

1. **ส่ง screenshot** ของผลลัพธ์จาก `test_cnn_model.py`
2. **ส่ง training history** (ไฟล์ `models/training_history_*.json`)
3. **บอกขนาดไฟล์** `models/card_classifier_cnn.pth`

---

**สร้างเมื่อ**: 2025-10-13  
**เวอร์ชัน**: 1.0  
**สถานะ**: ✅ วินิจฉัยเสร็จแล้ว - รอทดสอบ model
