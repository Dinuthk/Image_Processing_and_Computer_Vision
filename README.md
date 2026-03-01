# 🖼 Image Processing and Computer Vision 

## 📖 Project Description

This repository contains the complete solutions for **Assessment 01** of the Image Processing and Computer Vision course.

The assignment covers:

- Spatial domain filtering (Average, Median, Gaussian)
- Noise addition and removal
- Gaussian & Laplacian pyramids
- Wavelet decomposition
- Digital watermarking (DWT)
- Medical image analysis
- Fundus image segmentation (Classical approach)

All implementations are done using **Python with OpenCV, NumPy, and Matplotlib**.

---

## 🧠 Practical Part – Fundus Image Segmentation

### 🎯 Objective
To design and implement a **classical image processing pipeline (Non-AI)** to segment retinal vessels from fundus images and validate the results using quantitative metrics.

---

## 🔄 Segmentation Pipeline

The segmentation pipeline consists of:

1. Green channel extraction  
2. Gaussian noise reduction  
3. CLAHE contrast enhancement  
4. Adaptive thresholding  
5. Morphological operations  
6. Connected component filtering  
7. Vessel connectivity enhancement  

---

## 📊 Validation Metrics

The segmentation performance is evaluated using:

- **Dice Similarity Coefficient (DSC)**
- **Jaccard Index (IoU)**

Validation is performed on a subset of **50 annotated images** as required.

---

## 📁 Repository Structure
Assessment_01/
│
├── Question_01.py
├── Question_02.py
├── Question_03.py
├── Question_04.py
├── Question_05.py
├── Question_06.py
├── Question_07.py
├── Question_08.py
├── Question_09.py
├── Question_10.py
│
├── Fundus_Segmentation.py
├── results/
│
└── README.md
---

## 🛠 Technologies Used

- Python 3.x  
- OpenCV  
- NumPy  
- Matplotlib  

---

## 🚫 Important Note

- ❌ AI / Machine Learning / Deep Learning methods were NOT used.  
- ✅ Only classical image processing techniques were applied as required.

---

## 📚 Reference

---

## 👨‍💻 Author

**Dinuth Diruksha**  
Computer Engineering Undergraduate  
University of Ruhuna  

---
