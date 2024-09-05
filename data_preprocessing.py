import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load the prepared dataset
df = pd.read_csv('prepared_neurodiversity_dataset.csv')

# Define features and target columns
target_columns = ['Target_ADHD', 'Target_Dyslexia', 'Target_ASD', 'Target_Dysgraphia', 'Target_Dyscalculia']
X = df.drop(columns=target_columns)
y = df[target_columns]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessed data
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('X_test_scaled.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

# Save the preprocessed data as a pickle file
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump((X_train_scaled, X_test_scaled, y_train, y_test), f)

print(f"Preprocessing complete. Data shapes: X_train_scaled: {X_train_scaled.shape}, X_test_scaled: {X_test_scaled.shape}")
print("Data preprocessing complete and saved to 'preprocessed_data.pkl'.")




