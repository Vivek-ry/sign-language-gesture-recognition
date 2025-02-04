# Hand Gesture Recognition for Sign Language Detection

## Overview
This project focuses on developing a real-time **sign language detection system** using advanced **deep learning** techniques, including:
- **Convolutional Neural Networks (CNN)**
- **Temporal Convolutional Networks (TCN)**
- **Long Short-Term Memory (LSTM) models**

The goal is to bridge the communication gap between sign language users and non-users by accurately recognizing hand gestures and converting them into text or speech.

## Features
✅ Real-time **sign language interpretation** using webcam video
✅ **Deep learning models** (CNN, TCN, LSTM) for accurate recognition
✅ Uses **MediaPipe Holistic Model** for hand landmark detection
✅ **Data Augmentation** techniques for improving model generalization
✅ **Model Ensemble** methods for better accuracy and robustness
✅ **Performance Metrics**: Accuracy, F1 Score

## Tech Stack
- **Programming Language**: Python
- **Frameworks & Libraries**: TensorFlow, Keras, PyTorch, OpenCV, MediaPipe, scikit-learn, NumPy, Matplotlib

## Dataset & Data Processing
1. **Data Collection**: Hand gesture images/videos captured using webcam
2. **Preprocessing**:
   - Convert BGR to RGB
   - Extract **keypoints** (hand, face, body landmarks) using **MediaPipe Holistic Model**
   - Store extracted data in structured folders
3. **Data Augmentation**:
   - Image rotation, translation, and scaling
   - Generate additional training samples

## Model Architectures
### 1️⃣ **Convolutional Neural Network (CNN)**
- Extracts spatial features from video frames
- Uses multiple **Conv2D** and **MaxPooling** layers
- Final dense layers classify gestures

### 2️⃣ **Long Short-Term Memory (LSTM)**
- Captures temporal dependencies in hand movements
- Uses **three LSTM layers** for sequence processing
- Trained using **2000 epochs**

### 3️⃣ **Temporal Convolutional Network (TCN)**
- Uses **1D Convolutional layers** for sequence modeling
- Parallelizes training for efficiency
- Maintains temporal relationships in gesture recognition

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
```bash
pip install tensorflow keras torch torchvision torchaudio opencv-python mediapipe numpy matplotlib scikit-learn
```

### Running the Project
1️⃣ Clone the repository:
```bash
git clone https://github.com/Vivek-ry/sign-language-gesture-recognition.git
cd hand-gesture-recognition-for-sign-language-communication
```

2️⃣ Run the data collection script:
```bash
python data_collection.py
```

3️⃣ Train the model:
```bash
python train.py
```

4️⃣ Test real-time sign language recognition:
```bash
python real_time_detection.py
```

## Results & Performance
| Model | Accuracy | F1 Score |
|--------|---------|----------|
| CNN | 93.75% | 0.938 |
| LSTM | 91.67% | 0.918 |
| TCN | **Best Performance** | **Optimized for real-time use** |

- Achieved **high accuracy** and **real-time performance** with MediaPipe + deep learning models.
- Efficient real-time hand gesture detection and classification.

## Future Enhancements
🔹 Improve **dataset diversity** (more gestures, different backgrounds)
🔹 Optimize **model hyperparameters** for better accuracy
🔹 Deploy as a **web application** for broader accessibility
🔹 Implement **gesture-to-speech conversion**

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
