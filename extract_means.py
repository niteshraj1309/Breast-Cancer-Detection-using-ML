import pandas as pd
import json

# Step 1: Load training data (replace filename if needed)
df = pd.read_csv('breast_cancer_dataset.csv')  # your training dataset

# Step 2: Drop target column (e.g., 'diagnosis')
feature_data = df.drop(columns=['diagnosis'])

# Step 3: Calculate means of all 30 features
means = feature_data.mean().to_dict()

# Step 4: Save means to JSON
with open('feature_means.json', 'w') as f:
    json.dump(means, f)

print("Feature means saved to feature_means.json ✅")
