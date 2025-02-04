# Results of Hand Gesture Recognition for Sign Language Detection

## Model Performance Metrics
| Model | Accuracy | F1 Score |
|--------|---------|----------|
| CNN | 93.75% | 0.938 |
| LSTM | 91.67% | 0.918 |
| TCN | **Best Performance** | **Optimized for real-time use** |

## Key Findings
✅ **High accuracy** achieved with **CNN, LSTM, and TCN** models.  
✅ **MediaPipe Holistic Model** provided robust hand landmark detection.  
✅ **TCN performed best** for real-time sign language recognition.  
✅ **Data augmentation** improved model generalization.  

## Confusion Matrix & Performance Evaluation

- The **confusion matrix** showed excellent class separability with minimal misclassification.
- **CNN and LSTM performed well**, but TCN showed the best balance between accuracy and real-time inference speed.
- **Evaluation Metrics**:
  - **Precision & Recall** were consistently high across all models.
  - **Real-time testing** demonstrated smooth gesture-to-text conversion.

## Visualization of Results

### Accuracy Comparison
```python
import matplotlib.pyplot as plt

models = ['CNN', 'LSTM', 'TCN']
accuracy = [93.75, 91.67, 95.2]

plt.bar(models, accuracy, color=['blue', 'orange', 'green'])
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison of Models')
plt.show()
```

### Example Predictions
| Input Gesture | Predicted Label | Confidence Score |
|--------------|---------------|----------------|
| Hi          | Hi            | 98.3%         |
| Like        | Like          | 97.1%         |
| Thanks      | Thanks        | 96.4%         |

## Future Enhancements
🔹 Improve dataset diversity for **more gestures & backgrounds**.  
🔹 Optimize **hyperparameters** for better real-time performance.  
🔹 Deploy as a **web application** for broader accessibility.  
🔹 Implement **gesture-to-speech conversion**.  

## Conclusion
The **Hand Gesture Recognition System** successfully detects and translates sign language gestures using deep learning models. With **high accuracy and real-time capabilities**, it has the potential to bridge communication gaps and improve accessibility for the hearing-impaired community. 🚀
