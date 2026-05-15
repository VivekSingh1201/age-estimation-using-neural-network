# Age Detection from Images using Artificial Neural Network (ANN)

## Overview
This project implements an Age Group Classification System using an Artificial Neural Network (ANN) built with TensorFlow/Keras. The model classifies facial images into predefined age categories by learning visual patterns from image data.

The project also explores multiple deep learning optimization techniques such as:
- Optimizer comparison
- Activation function tuning
- Bias initialization experiments
- Regularization techniques
- Dropout
- Early stopping
- Cyclic learning rates
- TensorBoard visualization

This project serves as both an implementation of image classification using ANN and an experimental study of hyperparameter tuning techniques.

---

## Features
- Image preprocessing and resizing
- Label encoding and one-hot encoding
- ANN-based multiclass image classification
- TensorBoard experiment tracking
- Hyperparameter optimization experiments
- Learning rate scheduling
- Model saving and loading
- Prediction visualization
- CSV export of predictions

---

## Project Structure

```text
Age-Detection-ANN/
│
├── age_detection_train/
│   └── Train/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
│
├── age_detection_test/
│   └── Test/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
│
├── clr_callback.py
├── optimum_model.keras
├── out.csv
├── notebook.ipynb
├── requirements.txt
└── README.md
```

---

## Dataset
The dataset contains facial images grouped into age categories.
Dataset Source: Infosys Springboard

### Training Dataset
Contains:
- Image files
- Corresponding labels

Example labels:
- Young
- Middle
- Old

### Test Dataset
Contains:
- Unlabeled facial images for prediction

---

## Technologies Used
- Python 3.10+
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- ImageIO
- Scikit-learn
- TensorBoard

---

## Installation

### Clone Repository
```text
bash git clone https://github.com/your-username/age-detection-ann.git cd age-detection-ann
```

### Create Virtual Environment
```text
bash python -m venv DLenv source DLenv/bin/activate 
```

For Windows:
```text
bash DLenv\Scripts\activate 
```
---

### Install Dependencies
bash pip install -r requirements.txt 

Or manually:
bash pip install tensorflow numpy pandas matplotlib pillow imageio scikit-learn tensorboard 

---

## Model Architecture
The ANN architecture used:

```text
Input Image (32x32x3)
        ↓
Flatten Layer
        ↓
Dense Layer (500 neurons, ReLU)
        ↓
Dropout / Regularization (Optional Experiments)
        ↓
Output Layer (Softmax)
```
---

## Training
Train the model:
python model.fit(train_x,train_y,batch_size=32,epochs=50,     validation_split=0.2 ) 

---

## Hyperparameter Experiments

### Optimizers Tested
- Adam
- SGD
- RMSprop
- Adagrad
- Adadelta

### Activation Functions Tested
- ReLU
- Sigmoid
- Tanh
- Linear

### Bias Initializers
- Zero initialization
- Constant 0.01
- Constant 0.1

### Regularization Methods
- L1 (Lasso)
- L2 (Ridge)
- Elastic Net

### Dropout
Implemented dropout to reduce overfitting.

Example:
python Dropout(0.30) 

### Early Stopping
Stops training automatically when validation loss stops improving.

### Cyclic Learning Rate
Implemented CLR with triangular scheduling for learning rate optimization.

---

## TensorBoard Monitoring
Launch TensorBoard:
bash tensorboard --logdir=. 

Open browser:
bash http://localhost:6006 

Track:
- Accuracy
- Validation Accuracy
- Loss
- Validation Loss
- Learning Rate changes

---

## Saving the Model
python model.save('optimum_model.keras') 

---

## Loading the Model
python from tensorflow.keras.models import load_model  model = load_model('optimum_model.keras', compile=False) 

---

## Making Predictions
python pred = model.predict(test_x) pred = np.argmax(pred, axis=1) pred = lb.inverse_transform(pred) 

---

## Export Predictions
python test['Class'] = pred test.to_csv('out.csv', index=False) 

Output format:
csv ID,Class img1.jpg,Young img2.jpg,Middle img3.jpg,Old 

---

## Visual Prediction Example
python idx = 2481 single_image = np.expand_dims(test_x[idx], axis=0)  pred = model.predict(single_image) pred_class = np.argmax(pred, axis=1) pred_label = lb.inverse_transform(pred_class)[0] 

---

## Learning Outcomes
This project demonstrates:
- Fundamentals of Artificial Neural Networks
- Image preprocessing for deep learning
- Hyperparameter tuning
- Overfitting prevention techniques
- Model evaluation
- TensorBoard usage
- Model persistence
- Practical ANN experimentation

---

## Future Improvements
Possible upgrades:
- Replace ANN with CNN for improved accuracy
- Transfer learning with pretrained models
- Real-time webcam age detection
- Web deployment using Flask / Streamlit
- Better age range granularity
- Data augmentation

---

## Limitations
- ANN does not preserve spatial image structure
- CNN would outperform this model for image tasks
- Performance depends heavily on dataset quality
- Age prediction is grouped classification, not exact age regression

---

## Author
Vivek Kumar Singh

---

## License
This project is for educational and research purposes.
