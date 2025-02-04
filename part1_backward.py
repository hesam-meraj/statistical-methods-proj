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

selected_features = list(features.columns)

while len(selected_features) > 1:  

    X = sm.add_constant(features[selected_features])
    model = sm.OLS(target, X).fit()
    
    p_values = model.pvalues[1:]  
    least_significant_feature = p_values.idxmax()
    
    candidate_features = selected_features.copy()
    candidate_features.remove(least_significant_feature)
    
    X_candidate = sm.add_constant(features[candidate_features])
    candidate_model = sm.OLS(target, X_candidate).fit()
    
    if candidate_model.aic < model.aic:
        selected_features = candidate_features
        print(f"Removed feature: {least_significant_feature}, AIC: {candidate_model.aic}")
    else:
        break  

X_final = sm.add_constant(features[selected_features])
final_model = sm.OLS(target, X_final).fit()

print(final_model.summary())