import numpy as np
import pandas as pd
import statsmodels.api as sm
from part2 import *

ridge_best = ridge_cv.best_estimator_

X_train_const = sm.add_constant(X_train)  # Add intercept
ridge_ols = sm.OLS(y_train, X_train_const).fit_regularized(L1_wt=0, alpha=ridge_best.alpha)

coef = ridge_ols.params
X_train_matrix = np.column_stack((np.ones(X_train.shape[0]), X_train))
residuals = y_train - np.dot(X_train_matrix, coef)
rss = np.sum(residuals**2)
n, p = X_train_matrix.shape
variance = rss / (n - p)
standard_errors = np.sqrt(np.diag(np.linalg.inv(X_train_matrix.T @ X_train_matrix) * variance))

lower_bound = coef - 1.96 * standard_errors
upper_bound = coef + 1.96 * standard_errors

ci_df = pd.DataFrame({"Coefficient": coef, "Lower 95% CI": lower_bound, "Upper 95% CI": upper_bound})
print("\n--- 95% Confidence Intervals for Ridge Regression ---")
print(ci_df)

significant_predictors = ci_df[(ci_df["Lower 95% CI"] > 0) | (ci_df["Upper 95% CI"] < 0)]
print("\nSignificant Predictors (95% CI does NOT include zero):")
print(significant_predictors)
