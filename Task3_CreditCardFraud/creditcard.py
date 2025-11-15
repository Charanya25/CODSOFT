import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("📌 Loading dataset...")
df = pd.read_csv("creditcard.csv")

print("📌 Preparing features and target...")
X = df.drop("Class", axis=1)
y = df["Class"]

print("📌 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("📌 Training LIGHT Random Forest model (n_estimators=20)...")
model = RandomForestClassifier(
    n_estimators=20,     # FAST training
    max_depth=10,        # Prevents heavy computation
    n_jobs=-1,           # Use all CPU cores
    random_state=42
)

model.fit(X_train, y_train)

print("📌 Making predictions...")
pred = model.predict(X_test)

print("\n📊 Model Accuracy:", accuracy_score(y_test, pred))
print("\n📋 Classification Report:\n", classification_report(y_test, pred))
