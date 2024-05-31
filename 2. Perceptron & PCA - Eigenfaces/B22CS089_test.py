import numpy as np
import sys

def load_data(file_path):
    try:
        data = np.loadtxt(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def load_weights(weights_file):
    try:
        weights = np.loadtxt(weights_file)
        return weights
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

def perceptron_predict(X, weights):
    return np.where(np.dot(X, weights) >= 0, 1, 0)

# Load test data
X_test = np.loadtxt('B22CS089_test.txt')

# Normalize test data
X_test_normalized = X_test / np.linalg.norm(X_test, axis=0)

# Load weights
weights = np.loadtxt('weights.txt')

# Predict labels
y_pred = perceptron_predict(X_test_normalized, weights)

# Print labels in comma-separated form
print(','.join(map(str, y_pred)))

# Load the actual labels
actual_labels = load_data('labels.txt')

# Print predicted and actual labels
print("Predicted Labels\tActual Labels")
for pred, actual in zip(y_pred, actual_labels):
    print(f"{pred}\t\t\t{actual}")

# Calculate accuracy
accuracy = np.mean(y_pred == actual_labels) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
