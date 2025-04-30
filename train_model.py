import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import joblib

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Drop 'Time' column (not relevant for prediction)
X = df.drop(['Time', 'Class'], axis=1)
y = df['Class']

# Standardize the data (feature scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to oversample the minority class (fraudulent transactions)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize RandomForestClassifier with class weights (optional here due to SMOTE)
model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=20, random_state=42)

# Train the model
model.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model using classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix for deeper understanding
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'model.pkl')
print("Model saved as 'model.pkl'")
