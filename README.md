# Pneumonia Detection with PyTorch CNN

## 🧠 Overview
This project uses a convolutional neural network (CNN) built with PyTorch to classify chest X-ray images as either normal or pneumonia. The dataset used is the widely known Chest X-Ray Pneumonia Dataset from Kaggle.

## 🎯 Objective
Build a binary image classifier that distinguishes between normal lungs and pneumonia-infected lungs using PyTorch.

## 🗃 Dataset Structure
The dataset (`chest_xray/`) is structured into three folders:
- `train/`
- `val/`
- `test/`

Each contains subfolders: `NORMAL/` and `PNEUMONIA/`

## 🔧 Dependencies
```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

## 🚀 How to Run
1. Download and unzip the dataset from Kaggle into the root directory as `chest_xray/`
2. Run the script in your terminal or notebook:

```bash
python pneumonia_classifier.py
```

## 📈 Features
- Custom CNN model in PyTorch
- Data augmentation for training set
- Live tracking of loss and accuracy
- Classification report and confusion matrix

## 📊 Outputs
- Epoch-wise training loss and validation accuracy
- Final test accuracy
- Confusion matrix
- Classification report with precision, recall, F1-score

## 📌 Future Enhancements
- Integrate transfer learning (e.g., ResNet, EfficientNet)
- Add Grad-CAM visualization for interpretability
- Enable real-time image uploads for predictions
- Add early stopping and checkpoint saving
- Deploy as a web interface (Streamlit or Flask)

## 🧪 Sample Output
```
Epoch 1/5 - Loss: 0.5421, Val Acc: 0.8750
...
✅ Test Accuracy: 0.9042
📊 Classification Report:
              precision    recall  f1-score   support
    NORMAL       0.91      0.89      0.90       234
 PNEUMONIA       0.89      0.92      0.91       390
```

## 📁 File Structure
```
├── pneumonia_classifier.py
├── chest_xray/
│   ├── train/
│   ├── val/
│   └── test/
```

## Dataset
Dataset from Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
