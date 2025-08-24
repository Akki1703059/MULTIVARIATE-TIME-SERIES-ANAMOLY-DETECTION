import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


file_name = 'TEP_Train_Test.csv'

try:
    data = pd.read_csv(file_name)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found.")
    exit()

# clean the time column by  replaces any dashes with slashes.
data['Time'] = data['Time'].astype(str).str.replace('-', '/')

# Convert the now-consistent strings to datetime objects
data['Time'] = pd.to_datetime(data['Time'], format="%m/%d/%Y %H:%M")

# Normal period- given in the problem statement.
train_start = '2004-01-01 00:00:00'
train_end = '2004-01-05 23:59:00'

# Define the full analysis period for testing
analysis_start = '2004-01-01 00:00:00'
analysis_end = '2004-01-19 07:59:00'

# Select the feature columns for the model, dropping the Time column
feature_cols = data.columns.drop('Time')

# Split the data into the training and analysis sets
train_data = data[(data['Time'] >= train_start) & (data['Time'] <= train_end)]
analysis_data = data[(data['Time'] >= analysis_start) & (data['Time'] <= analysis_end)]

X_train = train_data[feature_cols]
X_analysis = analysis_data[feature_cols]

print(f"\nTraining data shape: {X_train.shape}")
print(f"Analysis data shape: {X_analysis.shape}")


# Create the Isolation Forest model
model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)

# Train the model on the normal training data
print("\nTraining the Isolation Forest model...")
model.fit(X_train)
print("Model training complete!")


scores = model.decision_function(X_analysis)

# We will use min-max scaling to transform the scores
scaler = MinMaxScaler(feature_range=(0, 100))
abnormality_scores = scaler.fit_transform(-scores.reshape(-1, 1))

# Add the new 'Abnormality_score' column to the analysis data
analysis_data['Abnormality_score'] = abnormality_scores

print("\nSuccessfully calculated Abnormality Scores!")



# Get the feature names
feature_names = X_analysis.columns



# A list to store the top features for each anomaly
top_features_list = []

# Iterate through each row in the full analysis data
for index, row in analysis_data.iterrows():
    # Find the contribution of each feature by calculating the deviation from the training data mean
    row_features = row[feature_names].values.reshape(1, -1)
    mean_of_train = X_train.mean().values
    feature_contributions = np.abs(row_features - mean_of_train)
    
    # Sort the features by their contribution (deviation)
    top_7_indices = np.argsort(feature_contributions)[0][-7:]
    top_7_features = [feature_names[i] for i in top_7_indices]
    
    # Reverse the list to get them from most to least important
    top_7_features.reverse()
    top_features_list.append(top_7_features)

# Add the top features as new columns to the DataFrame
for i in range(7):
    analysis_data[f'top_feature_{i+1}'] = [features[i] for features in top_features_list]

print("Top contributing features identified successfully.")



final_output_file = "TEP_Anomalies_Final.csv"

# Save the final DataFrame with all the new columns to a new CSV file
analysis_data.to_csv(final_output_file, index=False)

print(f"\nfinal csv is done at '{final_output_file}'!")