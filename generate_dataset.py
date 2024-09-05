import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 1000

# Define a mapping function for frequency scale
def encode_frequency(response):
    return {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Very Often': 4}[response]

# Generate synthetic data
data = {
    'Age': np.random.randint(6, 18, size=num_samples),
    'Grade_Class': np.random.choice(['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Grade 7', 'Grade 8', 'Grade 9', 'Grade 10', 'Grade 11', 'Grade 12'], size=num_samples),
    'Primary_User': np.random.choice(['Student', 'Parent/Guardian', 'Educator'], size=num_samples),
    # ADHD section
    'ADHD_Attention_Difficulty': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'ADHD_Need_To_Move': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'ADHD_Distracted_By_Stimuli': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'ADHD_Mentally_Restless': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'ADHD_Hard_To_Listen': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'ADHD_Impatience': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'ADHD_Unfinished_Tasks': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'ADHD_Forgot_Daily_Activities': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    # Dyslexia section
    'Dyslexia_Letter_Distinction': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dyslexia_Letter_Order': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dyslexia_Recognize_Common_Words': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dyslexia_Skip_Lines': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dyslexia_Follow_Read_Aloud': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dyslexia_Mispronounce_Words': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dyslexia_Reverse_Letters_Numbers': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dyslexia_Retain_Info': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    # ASD section
    'ASD_Prefer_Objects': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'ASD_Understand_Sarcasm': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'ASD_Predictable_Patterns': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'ASD_Repetitive_Movements': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'ASD_Focus_On_Interests': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'ASD_Adjust_Changes': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'ASD_Noisy_Environments': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'ASD_Understand_Emotions': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    # Dysgraphia section
    'Dysgraphia_Tiring_Handwriting': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dysgraphia_Form_Letters': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dysgraphia_Forget_Letters_Numbers': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dysgraphia_Rewrite_Legibility': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dysgraphia_Avoid_Handwriting': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dysgraphia_Neatness': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dysgraphia_Take_Notes': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dysgraphia_Organize_Thoughts': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    # Dyscalculia section
    'Dyscalculia_Mix_Symbols': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dyscalculia_Basic_Concepts': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dyscalculia_Tell_Time': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dyscalculia_Lose_Track_Steps': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dyscalculia_Organize_Numbers': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dyscalculia_Estimate_Concepts': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dyscalculia_Avoid_Activities': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    'Dyscalculia_Anxiety_Math': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Very Often'], size=num_samples),
    # Overlapping Conditions section
    'Overlap_Reading_Writing': np.random.choice(['Yes', 'No'], size=num_samples),
    'Overlap_Attention_Math': np.random.choice(['Yes', 'No'], size=num_samples),
    'Overlap_Multiple_Areas': np.random.choice(['Yes', 'No'], size=num_samples),
    'Overlap_Daily_Life_Affected': np.random.choice(['Yes', 'No'], size=num_samples),
    'Overlap_Strategies_Tools': np.random.choice(['Yes', 'No'], size=num_samples),
    'Overlap_Formal_Diagnoses': np.random.choice(['Yes', 'No'], size=num_samples),
    'Overlap_Learning_Needs_Met': np.random.choice(['Yes', 'No'], size=num_samples),
    'Overlap_Additional_Comments': np.random.choice(['', 'No comments'], size=num_samples)  # Comments can be empty
}

# Create DataFrame
df = pd.DataFrame(data)

# Optionally encode frequency responses if needed
frequency_columns = [col for col in df.columns if 'ADHD_' in col or 'Dyslexia_' in col or 'ASD_' in col or 'Dysgraphia_' in col or 'Dyscalculia_' in col]
for col in frequency_columns:
    df[col] = df[col].apply(encode_frequency)

# Save to CSV
df.to_csv('neurodiversity_assessment_dataset.csv', index=False)
print("Dataset generated and saved successfully.")

