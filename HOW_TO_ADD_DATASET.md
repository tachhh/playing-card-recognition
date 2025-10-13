# 🎯 Quick Guide: วิธีเพิ่ม Dataset

## ❌ ผิด vs ✅ ถูก

### ❌ ผิด - ใส่ path ของไฟล์
```
train_cnn_model.py           ❌ นี่คือไฟล์ ไม่ใช่ folder!
camera_simple.py             ❌
models/card_classifier.pth   ❌
```

### ✅ ถูก - ใส่ path ของ folder
```
C:\playing-card-recognition\playing-card-recognition\data\new_dataset     ✅
D:\my_cards                                                                ✅
C:\Users\YourName\Downloads\playing_cards                                  ✅
```

---

## 📝 ขั้นตอนที่ถูกต้อง

### 1. เตรียม Dataset ใหม่

วางไฟล์ภาพใน folder ใดก็ได้ เช่น:

```
C:\playing-card-recognition\playing-card-recognition\data\new_dataset\
├── ace of clubs\
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── ace of diamonds\
│   └── ...
└── ...
```

หรือ

```
D:\my_cards\
├── ace_of_clubs_01.jpg
├── ace_of_clubs_02.jpg
├── ace_of_diamonds_01.jpg
└── ...
```

### 2. คัดลอก Path ของ Folder

#### วิธีที่ 1: จาก File Explorer
1. เปิด folder ใน File Explorer
2. คลิกที่ address bar ด้านบน
3. Ctrl+C เพื่อ copy path
4. ตัวอย่าง: `C:\playing-card-recognition\playing-card-recognition\data\new_dataset`

#### วิธีที่ 2: Shift + Right Click
1. ไปที่ folder ใน File Explorer
2. กด Shift + Right Click บน folder
3. เลือก "Copy as path"
4. จะได้ path แบบมี quotes: `"C:\data\new_dataset"`
5. Paste ได้เลย (script จะลบ quotes ให้อัตโนมัติ)

### 3. รัน Script

**Double-click**: `merge_datasets.bat`

หรือ

```bash
python train_with_new_dataset.py
```

### 4. ใส่ Path

```
Enter path to NEW dataset folder: C:\playing-card-recognition\playing-card-recognition\data\new_dataset
```

หรือ (มี quotes ก็ได้)

```
Enter path to NEW dataset folder: "C:\playing-card-recognition\playing-card-recognition\data\new_dataset"
```

### 5. ยืนยัน

```
Continue? (y/n): y
```

---

## 🎓 ตัวอย่างการใช้งาน

### ตัวอย่างที่ 1: Dataset ใน Project

```
Dataset location: C:\playing-card-recognition\playing-card-recognition\data\new_dataset

Enter path to NEW dataset folder: C:\playing-card-recognition\playing-card-recognition\data\new_dataset
✅ Found 500 images in the dataset!
Continue? (y/n): y
```

### ตัวอย่างที่ 2: Dataset จาก Download

```
Dataset location: C:\Users\User\Downloads\playing_cards_v2

Enter path to NEW dataset folder: C:\Users\User\Downloads\playing_cards_v2
✅ Found 1000 images in the dataset!
Continue? (y/n): y
```

### ตัวอย่างที่ 3: Dataset จาก Drive D

```
Dataset location: D:\datasets\cards

Enter path to NEW dataset folder: D:\datasets\cards
✅ Found 750 images in the dataset!
Continue? (y/n): y
```

---

## ⚠️ ข้อผิดพลาดที่พบบ่อย

### ❌ Error: "Path not found"
**สาเหตุ**: พิมพ์ path ผิด หรือ folder ไม่มีจริง
**แก้ไข**: ตรวจสอบ path ให้ถูกต้อง copy จาก File Explorer

### ❌ Error: "This is not a folder"
**สาเหตุ**: ใส่ path ของไฟล์ (เช่น .py, .jpg)
**แก้ไข**: ใส่ path ของ folder ที่มีภาพไพ่

### ❌ Error: "No images found"
**สาเหตุ**: Folder ว่างเปล่า หรือไม่มีไฟล์ .jpg/.png
**แก้ไข**: ตรวจสอบว่า folder มีภาพไพ่จริงๆ

---

## ✅ Checklist

ก่อนรัน `merge_datasets.bat` ตรวจสอบ:

- [ ] มี folder ที่มีภาพไพ่แล้ว
- [ ] รู้ path เต็มของ folder (copy จาก File Explorer)
- [ ] ไม่ใช่ path ของไฟล์ .py หรือ .jpg
- [ ] Folder มีภาพ .jpg, .png, หรือ .jpeg

---

## 🚀 ถ้าพร้อมแล้ว

**Double-click**: `merge_datasets.bat`

และใส่ path ของ **folder** ที่มีภาพไพ่ครับ! 📁

---

Made with ❤️ for Playing Card Recognition Project
