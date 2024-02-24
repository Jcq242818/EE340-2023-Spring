import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load MNIST dataset
print("=> Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X = np.array(mnist['data'])
y = np.array(mnist['target'])
print(f"=> X shape: {X.shape}")

# Visualize a sample digit
sample_digit = X[0].reshape(28, 28)
plt.imshow(sample_digit, cmap='binary')
plt.axis('off')
plt.show()

# Prepare the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




# Train and evaluate models
print("=> Training and evaluating models...")
# Logistic Regression
print("=> Training Logistic Regression model...")
model = LogisticRegression()
# model.fit(X_train_scaled, y_train)


# Fit the model
model.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy on test set: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)
