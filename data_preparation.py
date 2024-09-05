import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('neurodiversity_assessment_dataset.csv')

# Inspect the dataset
print("First few rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Replace missing values with mode for categorical data
df.fillna(df.mode().iloc[0], inplace=True)

# One-hot encode categorical columns
categorical_cols = ['Grade_Class', 'Primary_User']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Encode binary 'Yes/No' columns using Label Encoding
binary_cols = [col for col in df.columns if df[col].dtype == 'object' and set(df[col].unique()) == {'Yes', 'No'}]
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# Create target columns based on symptoms
threshold = 10
df['Target_ADHD'] = df.filter(regex='ADHD_').sum(axis=1) >= threshold
df['Target_Dyslexia'] = df.filter(regex='Dyslexia_').sum(axis=1) >= threshold
df['Target_ASD'] = df.filter(regex='ASD_').sum(axis=1) >= threshold
df['Target_Dysgraphia'] = df.filter(regex='Dysgraphia_').sum(axis=1) >= threshold
df['Target_Dyscalculia'] = df.filter(regex='Dyscalculia_').sum(axis=1) >= threshold

# Convert targets to integer
df['Target_ADHD'] = df['Target_ADHD'].astype(int)
df['Target_Dyslexia'] = df['Target_Dyslexia'].astype(int)
df['Target_ASD'] = df['Target_ASD'].astype(int)
df['Target_Dysgraphia'] = df['Target_Dysgraphia'].astype(int)
df['Target_Dyscalculia'] = df['Target_Dyscalculia'].astype(int)

# Drop non-numeric columns
non_numeric_cols = ['Overlap_Additional_Comments']
features = df.drop(columns=['Target_ADHD', 'Target_Dyslexia', 'Target_ASD', 'Target_Dysgraphia', 'Target_Dyscalculia'] + non_numeric_cols)
targets = df[['Target_ADHD', 'Target_Dyslexia', 'Target_ASD', 'Target_Dysgraphia', 'Target_Dyscalculia']]

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Combine scaled features with targets
df_prepared = pd.DataFrame(features_scaled, columns=features.columns)
df_prepared = pd.concat([df_prepared, targets.reset_index(drop=True)], axis=1)

# Save the prepared dataset
df_prepared.to_csv('prepared_neurodiversity_dataset.csv', index=False)
print("Prepared dataset saved successfully.")
