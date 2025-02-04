# Setup Guide for Hand Gesture Recognition for Sign Language Detection

## Required Dependencies

To run this project, install the following dependencies:
```bash
pip install tensorflow keras torch torchvision torchaudio opencv-python mediapipe numpy matplotlib scikit-learn
```

## Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/PavanTejaSripati/hand-gesture-recognition-for-sign-language-communication.git
cd hand-gesture-recognition-for-sign-language-communication
```

### 2️⃣ Setup Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Run the Data Collection Script
```bash
python data_collection.py
```

### 4️⃣ Train the Model
```bash
python train.py
```

### 5️⃣ Test Real-time Sign Language Recognition
```bash
python real_time_detection.py
```

## Adding More Concepts for Better Understanding

### Hand Gesture Recognition Process
- **Data Collection:** Capturing gestures using webcam
- **Preprocessing:** Extracting hand keypoints using **MediaPipe**
- **Feature Extraction:** Using CNN, LSTM, and TCN models
- **Training & Evaluation:** Assessing accuracy and F1 Score
- **Real-Time Detection:** Converting gestures into text

### Current Models Implemented
1. **CNN (Convolutional Neural Network):** Extracts spatial features from images
2. **LSTM (Long Short-Term Memory):** Captures sequential dependencies in gestures
3. **TCN (Temporal Convolutional Network):** Optimized for real-time sequence modeling

### Model Used in Final Implementation
- **TCN performed best** for real-time sign language detection, achieving the highest accuracy and efficiency.

## Additional Improvements & Future Enhancements
🔹 Increase dataset diversity for **more gestures & real-world variations**.  
🔹 Optimize **hyperparameters** for better real-time performance.  
🔹 Deploy as a **web application** using Flask or FastAPI.  
🔹 Implement **gesture-to-speech conversion** for better accessibility.  
