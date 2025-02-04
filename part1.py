import pandas as pd
import numpy as np
import statsmodels.api as sm

# Assuming your dataset is in a pandas DataFrame called `df`
# and the target variable is in a column called 'target'
df = pd.read_csv("HousingData.xls") 

df = df.replace([np.inf, -np.inf], np.nan)  
df = df.dropna()  

target = df['MEDV']
features = df.drop(columns=['MEDV'])

selected_features = []

best_aic = float('inf')

while True:
    remaining_features = list(set(features.columns) - set(selected_features))
    new_aic = best_aic
    best_feature = None
    
    for feature in remaining_features:
        candidate_features = selected_features + [feature]
        
        X = sm.add_constant(features[candidate_features])
        model = sm.OLS(target, X).fit()
        
        if model.aic < new_aic:
            new_aic = model.aic
            best_feature = feature
    
    if new_aic >= best_aic:
        break
    
    best_aic = new_aic
    selected_features.append(best_feature)
    
    print(f"Added feature: {best_feature}, AIC: {best_aic}")

X_final = sm.add_constant(features[selected_features])
final_model = sm.OLS(target, X_final).fit()

print(final_model.summary())