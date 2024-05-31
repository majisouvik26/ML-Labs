import numpy as np
import sys

# Generate random weights for the linear function
np.random.seed(0)
w = np.random.randn(5)

# Generate synthetic dataset
num_samples = 5000
X = np.random.randint(-100, 101, (num_samples, 4))
y = np.where(np.dot(np.hstack((np.ones((num_samples, 1)), X)), w) >= 0, 1, 0)

# Save dataset to file
np.savetxt('B22CS089_data.txt', np.hstack((X, y.reshape(-1, 1))), fmt='%d')

# Combine X and y for consistent shuffling
data_combined = np.hstack((X, y.reshape(-1, 1)))

# Shuffle the combined data
np.random.shuffle(data_combined)


# Split the data into train and test sets (80% train, 20% test)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Save the train set to train.txt (with labels)
np.savetxt('B22CS089_train.txt', np.hstack((X_train, y_train.reshape(-1, 1))), fmt='%d', delimiter=' ')

# Save the test set to test.txt (without labels)
np.savetxt('B22CS089_test.txt', X_test, fmt='%d', delimiter=' ')

# Save the labels from test data to labels.txt
np.savetxt('labels.txt', y_test, fmt='%d')

print("Train and test datasets along with labels have been generated and saved.")

# Perceptron learning algorithm
def perceptron_train(X, y, num_epochs=100):
    # Initialize weights
    weights = np.zeros(X.shape[1])
    
    for _ in range(num_epochs):
        for i in range(X.shape[0]):
            if np.dot(X[i], weights) * y[i] <= 0:
                weights += y[i] * X[i]
    
    return weights

# Normalize training data
X_train_normalized = X_train / np.linalg.norm(X_train, axis=0)

# Train Perceptron
weights = perceptron_train(X_train_normalized, y_train)

# Save weights to file
np.savetxt('weights.txt', weights)

print("Training is complete and weights have been saved to weights.txt")
