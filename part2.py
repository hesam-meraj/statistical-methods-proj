import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("HousingData.xls") 


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)


X = df.drop(columns=["MEDV"])  
y = df["MEDV"]  

X_intercept = np.c_[np.ones(X.shape[0]), X] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def ridge_closed_form(X, y, lambda_):
    """Computes Ridge Regression coefficients manually."""
    I = np.eye(X.shape[1])  
    I[0, 0] = 0  
    beta = np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y
    return beta

lambda_ = 1.0 
beta_manual = ridge_closed_form(X_intercept, y, lambda_)
print("Manual Ridge Coefficients:\n", beta_manual)

ridge = Ridge()
ridge_params = {"alpha": np.logspace(-3, 3, 50)}  # Search over λ values
ridge_cv = GridSearchCV(ridge, ridge_params, cv=5, scoring="neg_mean_squared_error")
ridge_cv.fit(X_train, y_train)

print("\nBest Ridge λ:", ridge_cv.best_params_["alpha"])

lasso = Lasso()
lasso_params = {"alpha": np.logspace(-3, 3, 50)}
lasso_cv = GridSearchCV(lasso, lasso_params, cv=5, scoring="neg_mean_squared_error")
lasso_cv.fit(X_train, y_train)

print("\nBest Lasso λ:", lasso_cv.best_params_["alpha"])

elastic_net = ElasticNet()
elastic_params = {"alpha": np.logspace(-3, 3, 50), "l1_ratio": [0.1, 0.5, 0.9]}  # Mix L1 and L2
elastic_cv = GridSearchCV(elastic_net, elastic_params, cv=5, scoring="neg_mean_squared_error")
elastic_cv.fit(X_train, y_train)

print("\nBest Elastic Net Params:", elastic_cv.best_params_)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluates a model using MSE, R², and Adjusted R²."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n, p = X_test.shape  
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    
    return mse, r2, adjusted_r2

best_ridge = ridge_cv.best_estimator_
best_lasso = lasso_cv.best_estimator_
best_elastic = elastic_cv.best_estimator_

ridge_mse, ridge_r2, ridge_adj_r2 = evaluate_model(best_ridge, X_train, X_test, y_train, y_test)
lasso_mse, lasso_r2, lasso_adj_r2 = evaluate_model(best_lasso, X_train, X_test, y_train, y_test)
elastic_mse, elastic_r2, elastic_adj_r2 = evaluate_model(best_elastic, X_train, X_test, y_train, y_test)

print("\n--- Model Performance ---")
print(f"Ridge: MSE={ridge_mse:.4f}, R²={ridge_r2:.4f}, Adjusted R²={ridge_adj_r2:.4f}")
print(f"Lasso: MSE={lasso_mse:.4f}, R²={lasso_r2:.4f}, Adjusted R²={lasso_adj_r2:.4f}")
print(f"Elastic Net: MSE={elastic_mse:.4f}, R²={elastic_r2:.4f}, Adjusted R²={elastic_adj_r2:.4f}")
