import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 🌸 App Title
st.title("🌸 Iris Flower Classification App")
st.write("Predict the species of Iris flower based on its measurements.")

# 🪴 Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=['species'])

# 🧠 Split and scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

accuracy = accuracy_score(y_test, model.predict(X_test))

# 🌿 Sidebar Input
st.sidebar.header("Enter Flower Measurements 🌿")

sepal_length = st.sidebar.slider("Sepal length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), float(X['sepal length (cm)'].mean()))
sepal_width  = st.sidebar.slider("Sepal width (cm)", float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()), float(X['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider("Petal length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), float(X['petal length (cm)'].mean()))
petal_width  = st.sidebar.slider("Petal width (cm)", float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()), float(X['petal width (cm)'].mean()))

# 🧩 Create Input DataFrame
input_data = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})

# ⚙️ Scale the user input the same way as training data
input_scaled = scaler.transform(input_data)

# 🔮 Prediction
prediction = model.predict(input_scaled)[0]
species_name = iris.target_names[prediction]

# 📊 Display Results
st.subheader("🌼 Prediction Result")
st.success(f"Predicted Flower Species: **{species_name.capitalize()}**")
st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

# 📋 Dataset Preview
st.subheader("📊 Dataset Preview")
st.dataframe(X.head())
