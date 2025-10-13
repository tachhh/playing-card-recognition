# Captured Cards Directory

โฟลเดอร์นี้ใช้สำหรับเก็บรูปภาพไพ่ที่ capture จากกล้อง

## การใช้งาน

เมื่อรัน `camera_simple.py` หรือ `app/run_camera_cnn.py`:
- กด **'c'** เพื่อ capture รูปภาพไพ่
- รูปจะถูกบันทึกใน `captured_cards/` โดยอัตโนมัติ
- ชื่อไฟล์: `captured_YYYYMMDD_HHMMSS.jpg`

## โครงสร้างไฟล์

```
captured_cards/
├── captured_20251013_183045.jpg
├── captured_20251013_183102.jpg
└── ...
```

## หมายเหตุ

- โฟลเดอร์นี้ถูก ignore จาก Git (ไม่ต้องอัปโหลดรูปขึ้น GitHub)
- สามารถลบรูปเก่าได้ตามต้องการ
- รูปที่ capture จะใช้ preprocessing เดียวกับที่ใช้ train model
