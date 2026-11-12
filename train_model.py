import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# Load Dataset
# ==========================

data = pd.read_csv("dataset/Crop_recommendation.csv")

print("Dataset Loaded Successfully")
print(data.head())

# ==========================
# Check Dataset
# ==========================

print("\nDataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum())

# ==========================
# Features and Target
# ==========================

X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# K-Means Clustering
# ==========================

print("\nRunning K-Means Clustering...")

wcss = []

for i in range(1,11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig("static/images/elbow_method.png")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

print("K-Means Clustering Completed")

# ==========================
# Build Crop -> Group Map
# ==========================
# For each crop label, find the cluster it appears in most often.
# This lets us recommend other crops that share similar soil/
# environmental conditions as the predicted crop.

data_with_clusters = data.copy()
data_with_clusters['cluster'] = clusters

crop_cluster_map = (
    data_with_clusters.groupby('label')['cluster']
    .agg(lambda s: int(s.value_counts().idxmax()))
    .to_dict()
)

print("\nCrop -> Group Map:")
print(crop_cluster_map)

# ==========================
# Logistic Regression
# ==========================

print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(max_iter=5000)

lr_model.fit(X_train, y_train)

lr_prediction = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_prediction)

print("\nLogistic Regression Accuracy:")
print(lr_accuracy)

print(classification_report(y_test, lr_prediction))

# ==========================
# Random Forest
# ==========================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_prediction = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_prediction)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

print(classification_report(y_test, rf_prediction))

# ==========================
# Compare Models
# ==========================

print("\nModel Comparison")
print("----------------------------")
print("Logistic Regression :", lr_accuracy)
print("Random Forest       :", rf_accuracy)

# ==========================
# Save Best Model
# ==========================

if rf_accuracy >= lr_accuracy:
    best_model = rf_model
    print("\nBest Model: Random Forest")
else:
    best_model = lr_model
    print("\nBest Model: Logistic Regression")

joblib.dump(best_model, "crop_model.pkl")

print("\nBest model saved successfully as crop_model.pkl")

# ==========================
# Save Clustering Artifacts
# ==========================
# Saved so the web app can recommend crops from the same
# environmental/soil group as the predicted crop.

joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(crop_cluster_map, "crop_cluster_map.pkl")

print("Clustering artifacts saved as kmeans_model.pkl and crop_cluster_map.pkl")

# ==========================
# Sample Prediction
# ==========================

sample = np.array([[90, 42, 43, 20.8, 82.0, 6.5, 202]])

prediction = best_model.predict(sample)

print("\nRecommended Crop:", prediction[0])