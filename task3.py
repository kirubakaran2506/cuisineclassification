import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Dataset.csv")

# Handle missing values
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define target and features
TARGET_COL = "Cuisines"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Output: accuracy and top 10 cuisines with names
print("Accuracy:", accuracy)

cuisine_encoder = label_encoders[TARGET_COL]
top_10 = y.value_counts().head(10)
top_10_names = cuisine_encoder.inverse_transform(top_10.index)

print("Top 10 Cuisines:")
for name, count in zip(top_10_names, top_10.values):
    print(name, ":", count)