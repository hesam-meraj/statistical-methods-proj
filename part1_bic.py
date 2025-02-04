import pandas as pd
import numpy as np
import statsmodels.api as sm

# Assuming your dataset is in a pandas DataFrame called `df`
# and the target variable is in a column called 'target'
df = pd.read_csv("HousingData.xls") 

# Step 1: Handle missing or infinite values
df = df.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
df = df.dropna()  # Drop rows with NaN values (or use fillna() to impute)

# Step 2: Define the target variable and features
target = df['MEDV']
features = df.drop(columns=['MEDV'])

# Step 3: Initialize an empty list to store the selected features
selected_features = []

# Step 4: Initialize the best BIC to a large value
best_bic = float('inf')

# Step 5: Perform forward stepwise regression
while True:
    remaining_features = list(set(features.columns) - set(selected_features))
    new_bic = best_bic
    best_feature = None
    
    for feature in remaining_features:
        # Add the feature to the selected features
        candidate_features = selected_features + [feature]
        
        # Fit the model with the candidate features
        X = sm.add_constant(features[candidate_features])
        model = sm.OLS(target, X).fit()
        
        # Check if the BIC improves
        if model.bic < new_bic:
            new_bic = model.bic
            best_feature = feature
    
    # If no feature improves the BIC, stop the process
    if new_bic >= best_bic:
        break
    
    # Update the best BIC and add the best feature to the selected features
    best_bic = new_bic
    selected_features.append(best_feature)
    
    print(f"Added feature: {best_feature}, BIC: {best_bic}")

# Step 6: Final model with the selected features
X_final = sm.add_constant(features[selected_features])
final_model = sm.OLS(target, X_final).fit()

# Step 7: Print the summary of the final model
print(final_model.summary())