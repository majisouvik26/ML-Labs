# -*- coding: utf-8 -*-
"""B22CS089_problem2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UOAFptcAdSmLO6umcNk2a1VMjte4RHkC

TASK-1
1. Load the dataset using Scikit-learn's fetch_lfw_people function
2. Split the dataset into training and testing sets using an 80:20 split ratio
"""

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
# Load the LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# deatils of the dataset
print(f"Data shape: {lfw_people.images.shape}")
print(f"Number of samples: {lfw_people.images.shape[0]}")
print(f"Number of features: {lfw_people.images.shape[1] * lfw_people.images.shape[2]}")
print(f"Number of classes: {len(lfw_people.target_names)}")
print(f"Class names: {lfw_people.target_names}")

# Display some images from the dataset
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Loop over the subplots and plot images
for i, ax in enumerate(axes.flat):
    ax.imshow(lfw_people.images[i], cmap='gray')
    ax.set_title(lfw_people.target_names[lfw_people.target[i]], fontsize=8)  # Set smaller font size for titles
    ax.axis('off')

plt.tight_layout()
plt.show()

# Flatten the images
X = lfw_people.data
# Normalize the pixel values to be between 0 and 1
X = X / 255.0



# Print the shape of the preprocessed data
print(f"Preprocessed data shape: {X.shape}")
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(lfw_people.data, lfw_people.target, test_size=0.2, random_state=42)

import pandas as pd

# Convert numpy array to pandas DataFrame
df = pd.DataFrame(data=X, columns=[f"pixel_{i}" for i in range(X.shape[1])])

# Display the head of the DataFrame
print(df.head())

print(df.tail())

import seaborn as sns
columns = df.columns[:6]

# Plot the pairplot
sns.pairplot(df[columns])
plt.tight_layout()
plt.show()

print(df.info())

"""TASK-2
1. Implement Eigenfaces using PCA
2. Set appropriate value of n_components
"""

from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Set an appropriate value for n_components
n_components = 200

# Fit PCA on the training data
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

# Apply PCA transformation to both training and testing data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# choice of n_components using explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
# Plot the curve
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

# Set the style
sns.set_style("whitegrid")

# Set the context to notebook for a smaller font size
sns.set_context("notebook")

plt.figure(figsize=(8, 6))
sns.lineplot(x=range(1, len(cumulative_explained_variance_ratio) + 1), y=cumulative_explained_variance_ratio, marker='o')
plt.xlabel('Number of Components', fontsize=12)
plt.ylabel('Cumulative Explained Variance Ratio', fontsize=12)
plt.title('Explained Variance Ratio vs. Number of Components', fontsize=14)
plt.grid(True)
plt.xlim(1, 200)
plt.show()

"""TASK-3
1. Choose a classifier for Eigendaces and train the classifier using transformed training
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

# Choose a classifier (e.g., K-Nearest Neighbors)
knn = KNeighborsClassifier(n_neighbors=5)

# Initialize the models
models = {
    "KMeans": KMeans(n_clusters=len(set(y_train)), n_init=10),
    "SVM": SVC(C=1.0,gamma='scale'),
    "DecisionTree": DecisionTreeClassifier(),
    "KNN": knn
}

# Initialize Bagging classifiers
bagging_models = {
    "BaggingKMeans": BaggingClassifier(estimator=KMeans(n_clusters=len(set(y_train)), n_init=10)),
    "BaggingSVM": BaggingClassifier(estimator=SVC(C=1.0,gamma='scale')),
    "BaggingDecisionTree": BaggingClassifier(estimator=DecisionTreeClassifier()),
    "BaggingKNN": BaggingClassifier(estimator=knn)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Testing Accuracy: {accuracy}")
    # Calculate training accuracy to check if it is overfitting or not
    y_train_pred = model.predict(X_train_pca)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"{name} Training Accuracy: {train_accuracy}")
    results[name] = accuracy


for name, model in bagging_models.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Testing Accuracy: {accuracy}")
    # Calculate training accuracy to check if it is overfitting or not
    y_train_pred = model.predict(X_train_pca)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"{name} Training Accuracy: {train_accuracy}")
    results[name] = accuracy

# Create a DataFrame for the results
import pandas as pd
results_df = pd.DataFrame(results.items(), columns=['Model', 'Accuracy'])

# Plot the results
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('Accuracy Before and After Bagging', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

# Define the pipeline with scaling and KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Define the hyperparameters to tune
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train_pca, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Evaluate the model
accuracy = grid_search.score(X_test_pca, y_test)

print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy}")

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define the pipeline with scaling and SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Define the hyperparameters to tune
param_grid = {
    'svm__C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__gamma': ['scale', 'auto']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train_pca, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Evaluate the model
accuracy = grid_search.score(X_test_pca, y_test)

print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy}")

"""TASK-4
1. Use the trained Eigenfaces classifier to
make predictions on the Eigenfaces-transformed testing data.
2. Calculate and report accuracy.
3. Visualize a subset of Eigenfaces and report the
observations.
"""

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)
y_pred = knn.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate confusion matrix for KNN

cm_knn = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_knn, annot=True, cmap='Blues', fmt='d', xticklabels=lfw_people.target_names, yticklabels=lfw_people.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for KNN')
plt.show()




# Visualize a subset of Eigenfaces
fig, axes = plt.subplots(3, 5, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(lfw_people.images.shape[1], lfw_people.images.shape[2]), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Eigenface {i+1}")
plt.tight_layout()
plt.show()

# Report observations on model performance
# You can analyze the incorrect predictions to understand where the model is failing
incorrect_indices = (y_pred != y_test)
incorrect_images = X_test[incorrect_indices]
incorrect_labels_pred = y_pred[incorrect_indices]
incorrect_labels_true = y_test[incorrect_indices]


# Print classification report
print(classification_report(y_test, y_pred, target_names=lfw_people.target_names))

# Visualize a subset of Eigenfaces with actual and predicted labels
fig, axes = plt.subplots(3, 5, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(lfw_people.images.shape[1], lfw_people.images.shape[2]), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Actual: {lfw_people.target_names[y_test[i]]}\nPredicted: {lfw_people.target_names[y_pred[i]]}")
plt.tight_layout()
plt.show()

from sklearn.metrics import accuracy_score

# Calculate training accuracy
y_train_pred = knn.predict(X_train_pca)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy}")

"""TASK-5
1. Experiment with different values of n_components in PCA and observe the impact on the performance metrics (accuracy).
"""

# Initialize a k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Fit and evaluate the k-NN classifier before PCA
knn.fit(X_train, y_train)
y_pred_before_pca = knn.predict(X_test)
accuracy_before_pca = accuracy_score(y_test, y_pred_before_pca)

print(f"Accuracy before PCA: {accuracy_before_pca}")

# Define a range of values for n_components to experiment with
n_components_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]

# Initialize a dictionary to store accuracy for each value of n_components
accuracy_scores = {}

for n_components in n_components_range:
    # Fit PCA on the training data
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

    # Apply PCA transformation to both training and testing data
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train the classifier using the transformed training data
    knn.fit(X_train_pca, y_train)

    # Make predictions on the transformed testing data
    y_pred = knn.predict(X_test_pca)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Store the accuracy for this value of n_components
    accuracy_scores[n_components] = accuracy

# Print the accuracy for each value of n_components
for n_components, accuracy in accuracy_scores.items():
    print(f"n_components={n_components}: Accuracy={accuracy}")

# Plot the impact of n_components on accuracy

plt.figure(figsize=(14, 7))
plt.plot(list(accuracy_scores.keys()), list(accuracy_scores.values()), marker='o')
plt.axhline(y=accuracy_before_pca, color='orange', linestyle='--', label='Before PCA')
plt.xlabel('n_components')
plt.ylabel('Accuracy')
plt.title('Impact of n_components on Accuracy')
plt.xticks(list(accuracy_scores.keys()), rotation=45)  # Rotate x-tick labels to avoid overlap
plt.grid(True)
# Adjust the layout to prevent clipping of tick-labels
plt.tight_layout()
# Show the plot
plt.show()