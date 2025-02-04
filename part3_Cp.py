import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split



df = pd.read_csv("HousingData.xls") 

 
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

X = df.drop(columns=["MEDV"])
y = df["MEDV"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def forward_stepwise_regression(X, y):
    selected_features = []
    remaining_features = list(X.columns)
    best_aic = float("inf")
    best_model = None
    while remaining_features:
        aic_with_features = []
        for feature in remaining_features:
            X_selected = X[selected_features + [feature]]
            X_selected = sm.add_constant(X_selected)
            model = sm.OLS(y, X_selected).fit()
            aic_with_features.append((model.aic, feature, model))
        aic_with_features.sort()
        best_new_aic, best_feature, best_new_model = aic_with_features[0]
        if best_new_aic < best_aic:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_aic = best_new_aic
            best_model = best_new_model
        else:
            break
    return selected_features, best_model

selected_features_fwd, model_fwd = forward_stepwise_regression(X_train, y_train)
coef_fwd = model_fwd.params[1:]  # Skip intercept

def backward_stepwise_regression(X, y):
    selected_features = list(X.columns)
    best_aic = float("inf")
    best_model = None
    while len(selected_features) > 1:
        aic_with_features = []
        for feature in selected_features:
            reduced_features = selected_features.copy()
            reduced_features.remove(feature)
            X_selected = X[reduced_features]
            X_selected = sm.add_constant(X_selected)
            model = sm.OLS(y, X_selected).fit()
            aic_with_features.append((model.aic, feature, model))
        aic_with_features.sort()
        best_new_aic, worst_feature, best_new_model = aic_with_features[0]
        if best_new_aic < best_aic:
            selected_features.remove(worst_feature)
            best_aic = best_new_aic
            best_model = best_new_model
        else:
            break
    return selected_features, best_model

selected_features_bwd, model_bwd = backward_stepwise_regression(X_train, y_train)
coef_bwd = model_bwd.params[1:]

lasso = Lasso()
lasso_params = {"alpha": np.logspace(-3, 3, 50)}
lasso_cv = GridSearchCV(lasso, lasso_params, cv=5)
lasso_cv.fit(X_train, y_train)
lasso_best = lasso_cv.best_estimator_
coef_lasso = pd.Series(lasso_best.coef_, index=X.columns)

ridge = Ridge()
ridge_params = {"alpha": np.logspace(-3, 3, 50)}
ridge_cv = GridSearchCV(ridge, ridge_params, cv=5)
ridge_cv.fit(X_train, y_train)
ridge_best = ridge_cv.best_estimator_
coef_ridge = pd.Series(ridge_best.coef_, index=X.columns)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

def plot_coefficients(ax, coef, title):
    """Function to plot coefficients."""
    coef = coef.sort_values()
    ax.barh(coef.index, coef.values, color=["red" if c == 0 else "blue" for c in coef.values])
    ax.set_title(title)
    ax.axvline(0, color='black', linewidth=1)

coef_fwd_series = pd.Series(coef_fwd.values, index=selected_features_fwd)
coef_bwd_series = pd.Series(coef_bwd.values, index=selected_features_bwd)

plot_coefficients(axes[0], coef_fwd_series, "Forward Stepwise Regression Coefficients")
plot_coefficients(axes[1], coef_bwd_series, "Backward Stepwise Regression Coefficients")
plot_coefficients(axes[2], coef_lasso, "Lasso Regression Coefficients")
plot_coefficients(axes[3], coef_ridge, "Ridge Regression Coefficients")

plt.tight_layout()
plt.show()

