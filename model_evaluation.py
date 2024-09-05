import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

# Load the preprocessed data
with open('preprocessed_data.pkl', 'rb') as f:
    X_train_scaled, X_test_scaled, y_train, y_test = pickle.load(f)

# Load the trained model
model = load_model('neurodiversity_model.keras')

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)

# Ensure y_test and y_pred_binary have the same shape
y_test = np.array(y_test)
y_pred_binary = np.array(y_pred_binary)

# Print shapes for debugging
print(f'y_test shape: {y_test.shape}')
print(f'y_pred_binary shape: {y_pred_binary.shape}')

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy:.4f}')

# Prepare target names
target_names = ['ADHD', 'Dyslexia', 'ASD', 'Dysgraphia', 'Dyscalculia']

# Generate classification report
report = classification_report(y_test, y_pred_binary, target_names=target_names, zero_division=0)
print('Classification Report:')
print(report)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred_binary.argmax(axis=1))
print('Confusion Matrix:')
print(conf_matrix)




