import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv('fitlife_mood.csv')

# Drop columns that aren't useful for prediction
df.drop(['ID', 'Date', 'Mood Before(1-10)'], axis=1, inplace=True)

# Encode the target label
label_encoder = LabelEncoder()
df['Mood After'] = label_encoder.fit_transform(df['Mood After'])

# Separate features and target
X = df.drop('Mood After', axis=1)
y = df['Mood After']

# Encode categorical features if any
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and label encoder
with open('mood_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model training complete and files saved.")
  