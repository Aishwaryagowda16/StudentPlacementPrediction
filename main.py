import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/placementdata.csv")

# Convert Yes/No to 1/0
data["PlacementTraining"] = data["PlacementTraining"].map({"Yes":1,"No":0})
data["ExtracurricularActivities"] = data["ExtracurricularActivities"].map({"Yes":1,"No":0})

# Convert target
data["PlacementStatus"] = data["PlacementStatus"].map({"Placed":1,"NotPlaced":0})

# Features and Target
X = data.drop(["StudentID","PlacementStatus"], axis=1)
y = data["PlacementStatus"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "placement_model.pkl")

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100)
print("Model Saved Successfully!")