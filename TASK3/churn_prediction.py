import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("TASK3/data/Churn_Modelling.csv")
print("✅ Dataset Loaded!")
print(df.head())

# Check for missing values
print("\n🔍 Missing values:\n", df.isnull().sum())

# Drop any columns that won't help (like customer ID, Surname, RowNumber)
for col in ['RowNumber', 'CustomerId', 'Surname']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Convert categorical to numeric using get_dummies
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and target
X = df_encoded.drop('Exited', axis=1)
y = df_encoded['Exited']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))