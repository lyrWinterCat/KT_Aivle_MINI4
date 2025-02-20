# KT_Aivle_MINI4
# 📷 AI 기반 얼굴 인식 프로젝트 (Face Recognition)

## 📝 프로젝트 개요
본 프로젝트는 **사내 출입 과정에서의 대기 시간을 줄이기 위해 얼굴 인식 기술을 활용하는 시스템**을 구축하는 것을 목표로 합니다.  
기존의 카드키 방식은 **출퇴근 및 점심시간에 비효율적**이므로, **고속도로 하이패스 시스템과 유사한 실시간 얼굴 인식 시스템**을 도입하여 출입 시간을 단축하고자 합니다.  

### 🎯 **목표**
✅ 얼굴 인식 모델을 개발하여 **사내 출입 절차 자동화**  
✅ 실시간 얼굴 인식을 통해 **출퇴근 대기 시간 최소화**  
✅ **Keras 기반 FaceNet, YOLO 모델**을 비교 및 활용하여 최적의 모델 선정  

---

## 📂 데이터셋 소개

본 프로젝트에서는 **다양한 얼굴 인식 데이터셋**을 활용하여 모델을 학습합니다.  
📌 **데이터 출처**:
- [LFW Dataset (Kaggle)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
- [Face Recognition Dataset (Roboflow)](https://universe.roboflow.com/new-workspace-kuixc/face-recognition-dataset/dataset/1)

### 🔹 **데이터 종류**
| 데이터셋 | 이미지 개수 | 특징 |
|----------|------------|-----------------|
| LFW | 13,233장 | 유명인 얼굴 데이터 |
| Face Recognition 1 | 5,979장 | 일반 얼굴 데이터 |
| Face Recognition 2 | 4,983장 | 다양한 환경의 얼굴 데이터 |

---

## 🚀 프로젝트 수행 과정

### **1️⃣ 데이터 수집 및 전처리**
- **얼굴 데이터 수집**  
  - 제공된 데이터셋을 다운로드 후 **정제 및 필터링**
  - 추가적으로 **본인의 얼굴 이미지 촬영 후 데이터 수집**
- **데이터 전처리**  
  - 이미지 리사이징 (160x160)
  - 얼굴 영역 검출 및 정규화
  - 데이터 라벨링 및 Annotation 작업 (Roboflow, CVAT 등 활용)

### **2️⃣ 얼굴 인식 모델 학습**
#### 📌 **FaceNet 모델 (Keras)**
- 사전 학습된 **FaceNet 모델을 활용한 얼굴 임베딩 생성**
- 얼굴 이미지를 128차원 벡터로 변환 후, 유사도 계산을 통한 얼굴 분류  
- **주요 특징**:
  - 높은 정확도, 실시간 인식 속도 최적화

#### 📌 **YOLO 기반 얼굴 인식 모델**
- **YOLO-cls 모델 활용**  
  - 얼굴 영역 탐지 및 분류 수행
- **YOLO Object Detection 모델 적용**  
  - Annotation 데이터 기반으로 YOLO 학습
  - 객체 검출을 통해 **여러 명의 얼굴을 동시에 인식 가능**

### **3️⃣ 모델 성능 평가 및 비교**
- 모델별 평가 지표 분석:
  - **Precision (정밀도), Recall (재현율), F1-score**  
  - **Confusion Matrix 시각화**  
  - **ROC Curve & AUC Score 분석**
- **실제 카메라 환경 테스트**
  - **로컬 Webcam 테스트**를 통해 실시간 얼굴 인식 검증

---

## 🛠 사용 기술

| 기술 | 설명 |
|------|------|
| **Python** | 데이터 전처리 및 모델 학습 |
| **Keras / TensorFlow** | FaceNet 모델 학습 및 추론 |
| **YOLO (UltraLytics)** | 실시간 객체 검출 기반 얼굴 인식 |
| **OpenCV** | 이미지 전처리 및 실시간 카메라 테스트 |
| **Pandas / NumPy** | 데이터 처리 및 분석 |
| **Matplotlib / Seaborn** | 모델 성능 평가 시각화 |

---

## 🔥 실행 방법

### 
```bash
1️⃣ 필수 라이브러리 설치
pip install -r requirements.txt

2️⃣ 데이터 전처리 및 모델 학습
python preprocess.py  # 데이터 전처리 실행
python train_facenet.py  # FaceNet 모델 학습
python train_yolo.py  # YOLO 모델 학습

3️⃣ 실시간 얼굴 인식 테스트
python test_camera.py
```
## 📌 주요 개념 정리

### 🔹 **FaceNet 모델이란?**
FaceNet은 **딥러닝 기반 얼굴 인식 모델**로, 얼굴 이미지를 **고차원 벡터로 변환하여 유사도를 계산**하는 방식으로 동작합니다.  

- **입력:** `(160, 160, 3)` 크기의 얼굴 이미지  
- **출력:** `128차원 벡터 (임베딩)`  
- **유사도 계산 방법:** `cosine similarity (코사인 유사도)`

---

### 🔹 **YOLO 기반 얼굴 검출**
YOLO (**You Only Look Once**)는 **한 번의 패스만으로 객체를 탐지하는 CNN 기반 모델**입니다.  

- **YOLO-cls:** 이미지 내 얼굴을 **분류**  
- **YOLO Object Detection:** 이미지 내 **여러 개의 얼굴을 탐지 및 분류**  

YOLO 모델을 활용하면 **고속 실시간 얼굴 인식**이 가능하며, 여러 얼굴을 동시에 감지할 수 있습니다.

---

### 🔹 **성능 평가 지표**
머신러닝 및 딥러닝 모델의 성능을 평가하기 위해 아래의 주요 지표를 사용합니다.  

- **Precision (정밀도):**  
  - 모델이 얼굴이라고 예측한 것 중 실제 얼굴의 비율  
  - `Precision = TP / (TP + FP)`

- **Recall (재현율):**  
  - 실제 얼굴 중에서 모델이 얼굴이라고 맞춘 비율  
  - `Recall = TP / (TP + FN)`

- **F1-score:**  
  - Precision과 Recall의 조화 평균  
  - `F1-score = 2 × (Precision × Recall) / (Precision + Recall)`

- **ROC Curve & AUC (Receiver Operating Characteristic & Area Under Curve):**  
  - 모델의 분류 성능을 평가하는 그래프  
  - AUC 값이 **1에 가까울수록 성능이 좋음**


