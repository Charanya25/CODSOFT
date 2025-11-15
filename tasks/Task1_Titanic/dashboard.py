import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title and Description
st.title("🚢 Titanic Survival Prediction App")
st.write("""
This interactive app predicts whether a passenger **survived** or **did not survive**
on the Titanic, based on their details.
""")

# Load dataset
df = pd.read_csv("train.csv")

# Data cleaning
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Split data
X = df.drop(['Survived'], axis=1)
y = df['Survived']

# Re-check column names
st.write("✅ Model trained with columns:", list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# Sidebar for user input
st.sidebar.header("Enter Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 1, 80, 25)
sibsp = st.sidebar.slider("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.slider("Number of Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare Paid (£)", 0.0, 500.0, 32.2)
embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Convert input to dataframe
input_data = pd.DataFrame({
    'PassengerId': [0],  # dummy column to match training data if present
    'Pclass': [pclass],
    'Sex': [0 if sex == 'male' else 1],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [0 if embarked == 'S' else 1 if embarked == 'C' else 2]
})

# Align columns (fix mismatch)
missing_cols = set(X.columns) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0
input_data = input_data[X.columns]

# Prediction
prediction = model.predict(input_data)
result = "🟢 Survived" if prediction[0] == 1 else "🔴 Did Not Survive"

# Display Results
st.subheader("🎯 Prediction Result")
st.write(result)
st.metric("Model Accuracy", f"{accuracy:.2%}")

# Visualization
st.subheader("📊 Feature Importance")
importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
st.bar_chart(importance.set_index('Feature'))

# Correlation heatmap
st.subheader("📈 Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
