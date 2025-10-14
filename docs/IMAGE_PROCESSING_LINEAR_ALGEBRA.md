# หลักการ Linear Algebra ในการประมวลผลภาพ (Image Processing)
## คู่มือเชิงลึกสำหรับการนำเสนอวิชาการ

---

## สารบัญ

1. [บทนำ: ทำไมต้องใช้ Linear Algebra](#บทนำ-ทำไมตองใช-linear-algebra)
2. [ภาพคืออะไรในมุมมองคณิตศาสตร์](#ภาพคืออะไรในมุมมองคณิตศาสตร)
3. [Matrix และการแทนค่าภาพ](#matrix-และการแทนคาภาพ)
4. [Convolution: หัวใจของ Image Processing](#convolution-หวใจของ-image-processing)
5. [Kernel และ Filter Matrix](#kernel-และ-filter-matrix)
6. [การประยุกต์ใช้จริงในโปรเจกต์](#การประยกตใชจริงในโปรเจกต)
7. [Neural Network และ Linear Algebra](#neural-network-และ-linear-algebra)
8. [สรุปและบทวิเคราะห์](#สรปและบทวิเคราะห)

---

## บทนำ: ทำไมต้องใช้ Linear Algebra

### ความสำคัญของ Linear Algebra ใน Computer Vision

**Linear Algebra (พีชคณิตเชิงเส้น)** คือพื้นฐานสำคัญของการประมวลผลภาพทั้งหมด เพราะ:

```
ภาพดิจิตอล = Matrix (เมทริกซ์)
การประมวลผล = Matrix Operations (การดำเนินการเมทริกซ์)
```

**เปรียบเทียบ:**
```
ภาษามนุษย์: "ทำให้ภาพเบลอ"
ภาษาคอมพิวเตอร์: "คูณ Matrix ภาพกับ Gaussian Kernel Matrix"
```

### การประยุกต์ใช้ใน Computer Vision

| การทำงาน | วิธี Linear Algebra |
|----------|---------------------|
| ปรับความสว่าง | Scalar Multiplication |
| หมุนภาพ | Rotation Matrix |
| ขยาย/ย่อภาพ | Scaling Matrix |
| เบลอภาพ | Convolution with Blur Kernel |
| ตรวจขอบ | Convolution with Edge Detection Kernel |
| ทำ Neural Network | Matrix Multiplication + Activation |

---

## ภาพคืออะไรในมุมมองคณิตศาสตร์

### 1. ภาพขาว-ดำ (Grayscale Image)

**คำจำกัดความ:**
```
ภาพขาว-ดำ = Matrix 2 มิติ
```

**ตัวอย่าง:** ภาพขนาด 3×3 pixels

```
ภาพจริง:     Matrix แทนค่า:
■ □ ■         [  0  255   0]
□ ■ □    =    [255   0  255]
■ □ ■         [  0  255   0]

โดยที่:
  0 = สีดำ (■)
255 = สีขาว (□)
```

**สมการทางคณิตศาสตร์:**
```
I = [I(i,j)]  where i ∈ [1,m], j ∈ [1,n]
I(i,j) ∈ [0, 255]

โดยที่:
  I = Image Matrix
  i = row index (แถวที่)
  j = column index (คอลัมน์ที่)
  I(i,j) = ค่าความสว่างของ pixel ที่ตำแหน่ง (i,j)
```

### 2. ภาพสี (Color Image - RGB)

**คำจำกัดความ:**
```
ภาพสี = 3 Matrices ซ้อนกัน (Tensor 3 มิติ)
```

**โครงสร้าง:**
```
         ┌─────────┐
         │  Red    │ ← Matrix R[m×n]
         ├─────────┤
         │  Green  │ ← Matrix G[m×n]
         ├─────────┤
         │  Blue   │ ← Matrix B[m×n]
         └─────────┘
```

**สมการ:**
```
I_rgb = [R, G, B]  where R,G,B ∈ ℝ^(m×n)

I(i,j) = [R(i,j), G(i,j), B(i,j)]
```

**ตัวอย่าง:** Pixel สีม่วง
```
สีม่วง = (128, 0, 255)

Matrix แทนค่า:
R(i,j) = 128  (แดงปานกลาง)
G(i,j) = 0    (ไม่มีเขียว)
B(i,j) = 255  (น้ำเงินเต็มที่)
```

### 3. มิติของภาพ (Image Dimensions)

**สัญกรณ์มาตรฐาน:**
```
I ∈ ℝ^(H×W×C)

โดยที่:
  H = Height (ความสูง, จำนวนแถว)
  W = Width (ความกว้าง, จำนวนคอลัมน์)
  C = Channels (จำนวนช่องสี)

ตัวอย่าง:
  Grayscale: I ∈ ℝ^(480×640×1)
  RGB:       I ∈ ℝ^(480×640×3)
```

---

## Matrix และการแทนค่าภาพ

### 1. Matrix Representation (การแทนค่าด้วย Matrix)

**ภาพ 4×4 pixels ในรูป Matrix:**

```python
import numpy as np

# ภาพ Grayscale 4×4
image = np.array([
    [ 10,  20,  30,  40],
    [ 50,  60,  70,  80],
    [ 90, 100, 110, 120],
    [130, 140, 150, 160]
])

print(image.shape)  # (4, 4)
print(image[0, 0])  # 10 (pixel ซ้ายบนสุด)
print(image[3, 3])  # 160 (pixel ขวาล่างสุด)
```

**การเข้าถึง Pixel:**
```
I(i, j) = ค่าความสว่างที่แถว i, คอลัมน์ j

Indexing (Python):
  i = 0 ถึง H-1 (แถว)
  j = 0 ถึง W-1 (คอลัมน์)
```

### 2. Vector Representation (การแทนค่าด้วย Vector)

**Flatten Matrix เป็น Vector:**
```
Matrix 3×3:              Vector (Flatten):
[a b c]                  [a b c d e f g h i]^T
[d e f]        →         
[g h i]                  ขนาด: 9×1
```

**สมการ:**
```
v = flatten(I)
v ∈ ℝ^(H×W×1)

ตัวอย่าง:
Matrix 4×4 → Vector 16×1
```

**ประยุกต์ใช้:**
- ส่งเข้า Neural Network (Fully Connected Layer)
- คำนวณระยะห่างระหว่างภาพ
- Principal Component Analysis (PCA)

### 3. Matrix Properties สำหรับภาพ

**คุณสมบัติสำคัญ:**

```
1. Non-negative: I(i,j) ≥ 0  (ไม่มีค่าติดลบ)
2. Bounded:      I(i,j) ≤ 255 (มีขอบเขต)
3. Integer:      I(i,j) ∈ ℤ   (เป็นจำนวนเต็ม, แม้จะคำนวณแบบ float)
```

---

## Convolution: หัวใจของ Image Processing

### 1. Convolution คืออะไร

**คำจำกัดความ:**

Convolution เป็นการดำเนินการทางคณิตศาสตร์ที่ **"เลื่อน Filter ไปบนภาพ"** และคำนวณผลคูณของพิกเซลที่ทับกัน

**สัญกรณ์:**
```
S(i,j) = (I ∗ K)(i,j) = ΣΣ I(m,n) · K(i-m, j-n)
                        m n

โดยที่:
  S = Output (ผลลัพธ์)
  I = Input Image (ภาพต้นฉบับ)
  K = Kernel/Filter (ตัวกรอง)
  ∗ = Convolution operator
```

### 2. ตัวอย่างการคำนวณ Convolution

**ภาพต้นฉบับ (5×5):**
```
I = [1 2 3 2 1]
    [2 3 4 3 2]
    [3 4 5 4 3]
    [2 3 4 3 2]
    [1 2 3 2 1]
```

**Kernel (3×3) - Edge Detection:**
```
K = [-1 -1 -1]
    [-1  8 -1]
    [-1 -1 -1]
```

**ขั้นตอนการคำนวณ:**

#### ขั้นที่ 1: วาง Kernel ทับบนภาพ (Position 0,0)
```
เลือกพื้นที่ 3×3:
[1 2 3]
[2 3 4]
[3 4 5]

คูณทีละตัวกับ Kernel:
(-1×1) + (-1×2) + (-1×3) +
(-1×2) + ( 8×3) + (-1×4) +
(-1×3) + (-1×4) + (-1×5)

= -1 - 2 - 3 - 2 + 24 - 4 - 3 - 4 - 5
= 0
```

#### ขั้นที่ 2: เลื่อน Kernel ไปทางขวา 1 pixel
```
เลือกพื้นที่ใหม่:
[2 3 2]
[3 4 3]
[4 5 4]

คำนวณซ้ำ...
```

**ผลลัพธ์หลัง Convolution (3×3):**
```
S = [0  0  0]
    [0  0  0]
    [0  0  0]
```
(ค่าทั้งหมดเป็น 0 เพราะภาพเรียบ ไม่มีขอบ)

### 3. Convolution กับภาพที่มีขอบ

**ภาพที่มีขอบชัด:**
```
I = [0 0 0 1 1]
    [0 0 0 1 1]
    [0 0 0 1 1]
    [0 0 0 1 1]
    [0 0 0 1 1]
```

**ใช้ Edge Detection Kernel เดียวกัน:**
```
ที่ตำแหน่งขอบ:
[0 0 1]    [-1 -1 -1]
[0 0 1] ∗  [-1  8 -1]
[0 0 1]    [-1 -1 -1]

= (-1×0) + (-1×0) + (-1×1) +
  (-1×0) + ( 8×0) + (-1×1) +
  (-1×0) + (-1×0) + (-1×1)

= 0 + 0 - 1 + 0 + 0 - 1 + 0 + 0 - 1
= -3  (ตรวจพบขอบ!)
```

### 4. Convolution Properties

**คุณสมบัติสำคัญ:**

#### 4.1 Linearity (ความเป็นเชิงเส้น)
```
(aI₁ + bI₂) ∗ K = a(I₁ ∗ K) + b(I₂ ∗ K)

การ convolve ผลรวมภาพ = ผลรวมของ convolution แต่ละภาพ
```

#### 4.2 Commutativity (สลับที่ได้)
```
I ∗ K = K ∗ I

ภาพ convolve kernel = kernel convolve ภาพ
```

#### 4.3 Associativity (เปลี่ยนลำดับได้)
```
(I ∗ K₁) ∗ K₂ = I ∗ (K₁ ∗ K₂)

convolve 2 ครั้ง = convolve kernel ที่รวมกัน
```

### 5. Padding และ Stride

**Padding:** เติมขอบภาพ
```
Original (3×3):     With Padding (5×5):
[a b c]             [0 0 0 0 0]
[d e f]      →      [0 a b c 0]
[g h i]             [0 d e f 0]
                    [0 g h i 0]
                    [0 0 0 0 0]

ทำไม? เพื่อรักษาขนาดภาพหลัง convolution
```

**Stride:** ระยะการเลื่อน Kernel
```
Stride = 1:  เลื่อน 1 pixel ต่อครั้ง (ละเอียด)
Stride = 2:  เลื่อน 2 pixels ต่อครั้ง (ลดขนาด)

Output size = (Input - Kernel + 2×Padding) / Stride + 1
```

**ตัวอย่าง:**
```
Input: 5×5
Kernel: 3×3
Padding: 0
Stride: 1

Output = (5 - 3 + 0) / 1 + 1 = 3×3
```

---

## Kernel และ Filter Matrix

### 1. ความหมายของ Kernel

**Kernel (หรือ Filter)** คือ Matrix ขนาดเล็กที่กำหนดว่า:
- จะ**ประมวลผล**ภาพอย่างไร
- จะ**ตรวจจับ**คุณลักษณะอะไร

```
Kernel = "สูตรการแปลงภาพ"
```

### 2. ประเภทของ Kernel

#### 2.1 Identity Kernel (ไม่เปลี่ยนแปลง)
```
K = [0 0 0]
    [0 1 0]
    [0 0 0]

ผลลัพธ์: ภาพเหมือนเดิมทุกประการ
I ∗ K = I
```

#### 2.2 Blur Kernel (เบลอภาพ)

**Box Blur (Average Filter):**
```
K = (1/9) × [1 1 1]
            [1 1 1]
            [1 1 1]

หลักการ: เฉลี่ยค่า 9 pixels รอบๆ
```

**ตัวอย่างการคำนวณ:**
```
Input:         Output:
[10 20 30]     
[40 50 60]  →  (10+20+30+40+50+60+70+80+90)/9 = 50
[70 80 90]
```

**Gaussian Blur (เบลอแบบ Gaussian):**
```
K = (1/16) × [1  2  1]
             [2  4  2]
             [1  2  1]

หลักการ: ถ่วงน้ำหนักตามระยะห่าง (ใกล้ = มากกว่า, ไกล = น้อยกว่า)
```

**กราฟ Gaussian:**
```
    |     ╱‾‾‾╲
    |    ╱     ╲
    |   ╱       ╲
    |  ╱         ╲
    | ╱           ╲
    |───────────────
       ตรงกลางมีน้ำหนักมากสุด
```

#### 2.3 Sharpen Kernel (ทำให้คมชัด)
```
K = [0  -1   0]
    [-1  5  -1]
    [0  -1   0]

หลักการ: เพิ่มความแตกต่างระหว่าง pixel กับเพื่อนบ้าน
```

**สมการ:**
```
Sharpen = Original + (Original - Blurred)
        = Original + Edge Enhancement
```

#### 2.4 Edge Detection Kernels

**Sobel X (ตรวจขอบแนวตั้ง):**
```
K_x = [-1  0  1]
      [-2  0  2]
      [-1  0  1]

ตรวจจับการเปลี่ยนแปลงแนวแกน X (ซ้าย-ขวา)
```

**Sobel Y (ตรวจขอบแนวนอน):**
```
K_y = [-1 -2 -1]
      [ 0  0  0]
      [ 1  2  1]

ตรวจจับการเปลี่ยนแปลงแนวแกน Y (บน-ล่าง)
```

**การรวม Sobel:**
```
G = √(G_x² + G_y²)

โดยที่:
  G_x = I ∗ K_x
  G_y = I ∗ K_y
  G = ขนาดของ Gradient (ความชันของภาพ)
```

**Laplacian (ตรวจขอบทุกทิศทาง):**
```
K = [-1 -1 -1]
    [-1  8 -1]
    [-1 -1 -1]

หรือ

K = [0  -1  0]
    [-1  4 -1]
    [0  -1  0]

หลักการ: Second Derivative (อนุพันธ์อันดับสอง)
∇²I = ∂²I/∂x² + ∂²I/∂y²
```

#### 2.5 Emboss Kernel (ภาพนูน)
```
K = [-2 -1  0]
    [-1  1  1]
    [ 0  1  2]

สร้างเอฟเฟกต์เหมือนแกะสลัก
```

### 3. การออกแบบ Kernel เอง

**หลักการออกแบบ:**

#### 3.1 ผลรวมของ Kernel
```
ΣΣ K(i,j) = 1  → รักษาความสว่าง (Blur, Sharpen)
ΣΣ K(i,j) = 0  → ตรวจจับขอบ (Edge Detection)
```

**ตัวอย่าง:**
```
Blur Kernel:
[1/9 1/9 1/9]
[1/9 1/9 1/9]  ← ผลรวม = 9/9 = 1 ✓
[1/9 1/9 1/9]

Edge Kernel:
[-1 -1 -1]
[-1  8 -1]  ← ผลรวม = -8 + 8 = 0 ✓
[-1 -1 -1]
```

#### 3.2 สมมาตร (Symmetry)
```
Kernel แบบสมมาตร → ไม่มี bias ในทิศทาง

[1 2 1]
[2 4 2]  ← สมมาตรทุกแกน
[1 2 1]
```

#### 3.3 Separable Kernel
```
Kernel 2D สามารถแยกเป็น 2 Kernels 1D:

K_2D = K_x^T × K_y

ตัวอย่าง Gaussian:
[1 2 1]       [1]
[2 4 2]  =  [2] × [1 2 1]
[1 2 1]       [1]

ข้อดี: คำนวณเร็วกว่า (O(n²) → O(n))
```

---

## การประยุกต์ใช้จริงในโปรเจกต์

### 1. ขั้นตอนการประมวลผลภาพในโปรเจกต์

```
[กล้อง: 640×480×3]
        ↓
[RGB → Grayscale]  Matrix: 640×480×1
        ↓
[Gaussian Blur]     Convolution: I ∗ K_gaussian
        ↓
[Adaptive Threshold] Matrix Operation: I > T(x,y)
        ↓
[Morphological Operations]
        ├─ Opening:  (I ⊖ K) ⊕ K
        └─ Closing:  (I ⊕ K) ⊖ K
        ↓
[Find Contours]     Edge Detection + Connected Components
        ↓
[Extract Region]    Matrix Slicing: I[y1:y2, x1:x2]
        ↓
[Resize to 224×224] Matrix Interpolation
        ↓
[Normalize]         Matrix Arithmetic: (I - μ) / σ
        ↓
[CNN Model]
```

### 2. RGB → Grayscale Conversion

**สมการมาตรฐาน (ITU-R BT.709):**
```
Gray = 0.2989×R + 0.5870×G + 0.1140×B
```

**ในรูป Matrix:**
```
[R]       [0.2989]
[G]  ·    [0.5870]  = Gray
[B]       [0.1140]

หรือ

Gray = [R G B] · [0.2989 0.5870 0.1140]^T
```

**ทำไมใช้สัดส่วนนี้?**
- ตามนุษย์มองเห็น **Green** มากที่สุด (58.70%)
- **Blue** น้อยที่สุด (11.40%)
- **Red** ปานกลาง (29.89%)

**โค้ด:**
```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# เทียบเท่ากับ:
gray = 0.114*B + 0.587*G + 0.299*R
```

### 3. Gaussian Blur

**สมการ Gaussian 2D:**
```
G(x,y) = (1/(2πσ²)) × e^(-(x²+y²)/(2σ²))

โดยที่:
  σ = standard deviation (ความกว้างของ Gaussian)
  x,y = ระยะห่างจากจุดกลาง
```

**ตัวอย่าง Kernel 5×5 (σ=1.0):**
```
K = (1/273) × [1   4   7   4  1]
              [4  16  26  16  4]
              [7  26  41  26  7]
              [4  16  26  16  4]
              [1   4   7   4  1]

ค่าตรงกลาง (41/273 ≈ 15%) สูงสุด
ค่าขอบ (1/273 ≈ 0.4%) ต่ำสุด
```

**โค้ด:**
```python
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# (5, 5) = ขนาด kernel
# 0 = ให้คำนวณ σ อัตโนมัติ
```

**ทำไมต้อง Blur?**
```
ภาพเดิม:  มี noise (จุดรบกวน)
          ↓ Gaussian Blur
ภาพใหม่:  เรียบ, ลด noise
          → Threshold ได้แม่นยำขึ้น
```

### 4. Adaptive Threshold

**สมการ:**
```
T(x,y) = mean(I(x-w:x+w, y-w:y+w)) - C

Output(x,y) = {
  255  if I(x,y) > T(x,y)
  0    otherwise
}

โดยที่:
  T(x,y) = threshold ที่แต่ละตำแหน่ง (adaptive)
  w = ขนาดหน้าต่าง (neighborhood)
  C = ค่าคงที่ลบออก
```

**ทำไมต้อง Adaptive?**
```
Global Threshold:  T = 128 ทั้งภาพ
                   ↓
                   แสงไม่สม่ำเสมอ → ผลลัพธ์ไม่ดี

Adaptive Threshold: T แต่ละบริเวณ
                    ↓
                    ปรับตามแสงท้องถิ่น → ผลลัพธ์ดีกว่า
```

**โค้ด:**
```python
thresh = cv2.adaptiveThreshold(blur, 255, 
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# 11 = block size (ขนาดหน้าต่าง 11×11)
# 2 = constant C
```

### 5. Morphological Operations

**Mathematical Morphology** ใช้ Set Theory และ Linear Algebra

#### 5.1 Erosion (การกัดเซาะ)
```
A ⊖ B = {z | (B)_z ⊆ A}

หมายความ: เก็บเฉพาะตำแหน่งที่ B พอดีกับ A

Matrix Operation:
  สำหรับแต่ละตำแหน่ง:
    if min(A[region] × B) == 1:
      Output = 1
    else:
      Output = 0
```

**ผลลัพธ์:** ภาพเล็กลง, ลบจุดเล็กๆ

#### 5.2 Dilation (การขยาย)
```
A ⊕ B = {z | (B)_z ∩ A ≠ ∅}

Matrix Operation:
  สำหรับแต่ละตำแหน่ง:
    if max(A[region] × B) == 1:
      Output = 1
    else:
      Output = 0
```

**ผลลัพธ์:** ภาพใหญ่ขึ้น, เติมรูเล็กๆ

#### 5.3 Opening (เปิด)
```
A ∘ B = (A ⊖ B) ⊕ B

1. Erosion ก่อน  → ลบจุดเล็กๆ
2. Dilation ตาม → คืนขนาดเดิม

ผลลัพธ์: ลบ noise, รักษารูปร่างหลัก
```

#### 5.4 Closing (ปิด)
```
A • B = (A ⊕ B) ⊖ B

1. Dilation ก่อน → เติมรู
2. Erosion ตาม  → คืนขนาดเดิม

ผลลัพธ์: เติมรูเล็กๆ, เชื่อมขอบที่ขาด
```

**โค้ด:**
```python
kernel = np.ones((3, 3), np.uint8)

# Opening
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Closing
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
```

**Structuring Element (Kernel):**
```
3×3 Square:      3×3 Cross:
[1 1 1]          [0 1 0]
[1 1 1]          [1 1 1]
[1 1 1]          [0 1 0]

5×5 Circle:
[0 1 1 1 0]
[1 1 1 1 1]
[1 1 1 1 1]
[1 1 1 1 1]
[0 1 1 1 0]
```

### 6. Image Normalization

**สมการ:**
```
I_norm = (I - μ) / σ

โดยที่:
  μ = mean (ค่าเฉลี่ย)
  σ = standard deviation (ส่วนเบี่ยงเบนมาตรฐาน)
```

**คำนวณ μ และ σ:**
```
μ = (1/(H×W)) × ΣΣ I(i,j)

σ = √[(1/(H×W)) × ΣΣ (I(i,j) - μ)²]
```

**สำหรับ RGB (ImageNet Statistics):**
```
μ = [0.485, 0.456, 0.406]  ← ค่าเฉลี่ยแต่ละ channel
σ = [0.229, 0.224, 0.225]  ← std แต่ละ channel

I_norm(c) = (I(c) - μ(c)) / σ(c)  for c ∈ {R,G,B}
```

**ทำไมต้อง Normalize?**
```
ก่อน: ค่า 0-255 → range กว้าง → gradient ไม่เสถียร
หลัง: ค่า ≈-2 ถึง 2 → range แคบ → training เร็ว + แม่นกว่า
```

**โค้ด:**
```python
transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Matrix Operation:
normalized = (image - mean) / std
```

---

## Neural Network และ Linear Algebra

### 1. Convolutional Layer

**สมการ:**
```
Y = σ(W ∗ X + b)

โดยที่:
  X = Input (ภาพ)
  W = Weights (Kernel/Filter ที่เรียนรู้ได้)
  b = Bias
  σ = Activation Function (ReLU, Sigmoid, etc.)
  Y = Output (Feature Map)
```

**ตัวอย่าง Conv Layer ในโปรเจกต์:**
```python
nn.Conv2d(3, 32, kernel_size=3, padding=1)

# Input:  3 channels (RGB)
# Output: 32 channels (32 feature maps)
# Kernel: 3×3
```

**Matrix Dimensions:**
```
Input:  [Batch, 3, 224, 224]
Weight: [32, 3, 3, 3]  ← 32 kernels, each 3×3×3
Bias:   [32]
Output: [Batch, 32, 224, 224]

คำนวณ:
  สำหรับแต่ละ Kernel (32 อัน):
    Output[i] = Σ (Input[c] ∗ Weight[i,c]) + Bias[i]
                c=1 to 3
```

### 2. Batch Normalization

**สมการ:**
```
BN(x) = γ × ((x - μ_B) / √(σ_B² + ε)) + β

โดยที่:
  μ_B = mean ของ batch
  σ_B = std ของ batch
  γ = scale parameter (learnable)
  β = shift parameter (learnable)
  ε = ค่าเล็กๆ ป้องกันหารด้วย 0 (เช่น 10^-5)
```

**Matrix Form:**
```
x_norm = (x - μ) / σ
y = γ ⊙ x_norm + β

⊙ = element-wise multiplication
```

**ทำไมต้องใช้?**
```
1. Normalize activation → gradient ไหลดีขึ้น
2. ลด internal covariate shift
3. ทำให้ training เร็วขึ้น
4. ลด overfitting (มีผล regularization)
```

### 3. Fully Connected Layer

**สมการ:**
```
Y = σ(WX + b)

โดยที่:
  X ∈ ℝ^(n×1)   = Input vector
  W ∈ ℝ^(m×n)   = Weight matrix
  b ∈ ℝ^(m×1)   = Bias vector
  Y ∈ ℝ^(m×1)   = Output vector
```

**ตัวอย่างในโปรเจกต์:**
```python
nn.Linear(256 * 14 * 14, 512)

# Input:  256×14×14 = 50,176 neurons (flatten)
# Output: 512 neurons
# Weight: [512, 50176]
# Bias:   [512]
```

**Matrix Multiplication:**
```
[512 neurons]  =  [512×50176] × [50176×1] + [512×1]
    Y                  W            X          b

จำนวนพารามิเตอร์ = 512 × 50,176 + 512 = 25,690,624
```

### 4. Pooling Layer

**Max Pooling:**
```
MaxPool(I, k) = max{I(i,j) | (i,j) ∈ window}

ตัวอย่าง (2×2):
Input:        Output:
[1 2 | 3 4]   [6 8]
[5 6 | 7 8]   ────→
─────────     [10 12]
[1 2 | 3 4]
[9 10| 11 12]

เลือกค่าสูงสุดในแต่ละ window
```

**สมบัติ:**
```
1. ลดขนาดภาพ (Downsampling)
2. ลดพารามิเตอร์
3. Translation Invariance (ไม่สนตำแหน่งเล็กน้อย)
4. ไม่มีพารามิเตอร์ที่เรียนรู้
```

### 5. Activation Functions

**ReLU (Rectified Linear Unit):**
```
ReLU(x) = max(0, x) = {
  x   if x > 0
  0   if x ≤ 0
}

Matrix Operation:
  ReLU(X) = X ⊙ (X > 0)
  ⊙ = element-wise multiplication
```

**Softmax (Output Layer):**
```
Softmax(x_i) = e^(x_i) / Σ e^(x_j)
                        j=1 to n

สำหรับ vector:
[x₁]       [e^x₁ / Z]
[x₂]  →    [e^x² / Z]
[x₃]       [e^x³ / Z]

โดยที่ Z = e^x₁ + e^x² + e^x³
```

**คุณสมบัติ Softmax:**
```
1. Output ∈ (0, 1)  ← ทุกค่าเป็นบวก < 1
2. Σ Output = 1     ← รวมเป็น 100%
3. Differentiable   ← หา gradient ได้
```

---

## สรุปและบทวิเคราะห์

### 1. บทบาทของ Linear Algebra ในแต่ละขั้นตอน

| ขั้นตอน | การดำเนินการ | Matrix Operation |
|---------|-------------|------------------|
| **Input** | อ่านภาพ | Matrix Representation |
| **Grayscale** | แปลงสี | Matrix-Vector Multiplication |
| **Blur** | ทำให้เบลอ | Convolution (Matrix) |
| **Threshold** | แยกวัตถุ | Element-wise Comparison |
| **Morphology** | ทำความสะอาด | Set Operations on Matrices |
| **Resize** | ปรับขนาด | Matrix Interpolation |
| **Normalize** | ทำให้เป็นมาตรฐาน | Matrix Arithmetic |
| **Conv Layer** | สกัดคุณลักษณะ | Convolution + Matrix Multiply |
| **FC Layer** | จัดประเภท | Matrix Multiplication |
| **Softmax** | ความน่าจะเป็น | Exponential + Normalization |

### 2. ความซับซ้อนทางคำนวณ (Computational Complexity)

**Convolution:**
```
Time Complexity: O(H × W × K² × C_in × C_out)

โดยที่:
  H, W = ขนาดภาพ
  K = ขนาด Kernel
  C_in = Input channels
  C_out = Output channels

ตัวอย่าง:
  224×224, Kernel 3×3, 3→32 channels
  = 224 × 224 × 9 × 3 × 32
  = 43,614,720 operations
```

**Fully Connected:**
```
Time Complexity: O(n × m)

โดยที่:
  n = Input size
  m = Output size

ตัวอย่าง:
  50,176 → 512
  = 50,176 × 512
  = 25,690,112 operations
```

**เปรียบเทียบ:**
```
Conv Layer:  น้อยกว่า FC แต่ทำหลายชั้น
FC Layer:    มากมหาศาล แต่ทำน้อยชั้น
Total:       Conv มีประสิทธิภาพกว่าใน CNN
```

### 3. ทำไม CNN ถึงได้ผล

**1. Local Connectivity:**
```
FC: แต่ละ neuron เชื่อมทุก pixel → พารามิเตอร์เยอะมาก
CNN: แต่ละ neuron เชื่อมแค่บริเวณเล็กๆ → พารามิเตอร์น้อย

ตัวอย่าง:
  Image 224×224×3 → 512 features
  FC:  150,528 × 512 = 77M parameters
  CNN: 3×3×3 × 32 = 864 parameters
```

**2. Parameter Sharing:**
```
Kernel เดียว ใช้ทั้งภาพ
→ เรียนรู้ pattern ครั้งเดียว ใช้ได้ทุกที่
→ Translation Invariance (ไม่ว่าไม้จะอยู่ไหน ก็จดจำได้)
```

**3. Hierarchical Learning:**
```
Layer 1: ขอบ, เส้น, มุม (Low-level)
Layer 2: รูปร่างเล็กๆ (Mid-level)
Layer 3: ส่วนประกอบของไม้ (High-level)
Layer 4: ไม้เล่นทั้งใบ (Very High-level)
```

### 4. การเลือก Hyperparameters

**Kernel Size:**
```
3×3:  ขนาดมาตรฐาน, เร็ว, ได้ผลดี
5×5:  receptive field กว้างกว่า แต่ช้ากว่า
1×1:  ลด/เพิ่ม channels, ไม่มี spatial information
```

**Padding:**
```
Same: Output size = Input size
Valid: Output size < Input size
```

**Stride:**
```
1: รักษารายละเอียด แต่คำนวณช้า
2: ลดขนาด 2 เท่า, เร็วกว่า, เสียรายละเอียดบ้าง
```

### 5. แนวทางการปรับปรุง

**1. Data Augmentation:**
```
I_aug = Affine(I) = [cos θ  -sin θ] × [x] + [t_x]
                    [sin θ   cos θ]   [y]   [t_y]

การหมุน, เลื่อน, ย่อ/ขยาย = Matrix Transformation
```

**2. Advanced Architectures:**
```
ResNet:   Skip Connections → Y = F(X) + X
DenseNet: Concatenate → Y = [X, F(X)]
Attention: Weighted Sum → Y = Σ α_i × X_i
```

**3. Transfer Learning:**
```
Pre-trained Weights:
  W_pretrained (ImageNet) → W_finetuned (Cards)
  
ข้อดี: เรียนรู้เร็วกว่า, ได้ผลดีกว่า
```

---

## สรุปสุดท้าย

### ข้อได้เปรียบของ Linear Algebra Approach

1. **Mathematical Foundation:** มีพื้นฐานทางคณิตศาสตร์ชัดเจน
2. **Optimization:** หา Gradient และ Backpropagation ได้ง่าย
3. **Parallelization:** คำนวณ Matrix แบบ parallel บน GPU
4. **Reproducibility:** ผลลัพธ์เหมือนกันทุกครั้ง (ถ้า seed เดียวกัน)
5. **Scalability:** ขยายขนาด Model ได้ง่าย

### Key Takeaways

```
1. ภาพ = Matrix → การประมวลผล = Matrix Operations
2. Convolution = หัวใจของ Image Processing และ CNN
3. Kernel Design = กำหนดว่าจะประมวลผลอย่างไร
4. Linear Algebra → เหตุผลว่าทำไม Deep Learning ถึงทำงาน
5. การเข้าใจคณิตศาสตร์ → ปรับแต่ง Model ได้ดีขึ้น
```

### ความรู้ที่ควรศึกษาต่อ

1. **Matrix Calculus:** การหา Gradient ของ Matrix
2. **Eigenvalues/Eigenvectors:** PCA, SVD สำหรับลดมิติ
3. **Optimization Theory:** Gradient Descent และ variants
4. **Information Theory:** Entropy, Cross-Entropy Loss
5. **Signal Processing:** Fourier Transform สำหรับ frequency domain

---

## อ้างอิง

### หนังสือแนะนำ
1. **"Deep Learning"** - Goodfellow, Bengio, Courville
2. **"Computer Vision: Algorithms and Applications"** - Szeliski
3. **"Linear Algebra and Its Applications"** - Gilbert Strang
4. **"Digital Image Processing"** - Gonzalez, Woods

### Papers สำคัญ
1. **ImageNet Classification with Deep CNNs** - Krizhevsky et al., 2012
2. **Gradient-Based Learning Applied to Document Recognition** - LeCun et al., 1998
3. **Batch Normalization** - Ioffe & Szegedy, 2015

### Online Resources
- **Khan Academy:** Linear Algebra
- **3Blue1Brown:** Neural Networks
- **Stanford CS231n:** Convolutional Neural Networks
- **MIT 18.06:** Linear Algebra by Gilbert Strang

---

**จัดทำโดย:** Playing Card Recognition Project  
**วันที่อัพเดท:** October 14, 2025  
**สำหรับ:** การนำเสนอวิชาการและเอกสารประกอบการสอน

---

© 2025 - เอกสารนี้จัดทำเพื่อการศึกษาและวิจัย
