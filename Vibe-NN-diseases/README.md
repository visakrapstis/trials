# Vibe-NN-diseases

A neural network-based disease prediction system using symptom patterns. This project demonstrates the application of deep learning in medical diagnosis assistance.

CAVE: Please note, that the data is fully AI-generated and randomly synthesized / augmented. It trully in this case only is a representation of fast pased AI generated workflow, including cursor, MCPs. 

## Project Overview

The project was developed in three main steps:

1. **Data Preparation and Augmentation**
   - Created a comprehensive database of 100 internal medicine diseases with their symptoms
   - Implemented data augmentation to generate 5,000 synthetic patient cases
   - Randomized symptom patterns to simulate real-world variability in disease presentation

2. **Neural Network Development**
   - Built a deep learning model using TensorFlow
   - Implemented a 3-layer neural network with batch normalization and dropout
   - Used early stopping to prevent overfitting
   - Achieved 98% accuracy on test data

3. **Prediction System**
   - Developed a symptom-based disease prediction system
   - Provides top 3 most likely diseases with confidence scores
   - Handles partial symptom sets effectively
   - Demonstrates robust performance even with limited symptom information

## Results

The model shows impressive performance in disease prediction:
- Training Accuracy: >99%
- Validation Accuracy: >97%
- Test Accuracy: 98.10%
- Handles both common and rare diseases effectively
- Provides confidence scores for predictions

## Technologies Used
- Python
- TensorFlow
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
