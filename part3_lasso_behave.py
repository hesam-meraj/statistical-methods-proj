import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from part2 import *

# Train Lasso Regression
lasso = Lasso(alpha=0.1)  # Alpha controls regularization strength
lasso.fit(X_train, y_train)

# Train Ridge Regression for Comparison
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)

# Create a DataFrame to compare coefficients
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Lasso Coefficients': lasso.coef_,
    'Ridge Coefficients': ridge.coef_
})

print(coef_df)

# Plot the coefficients
plt.figure(figsize=(10,6))
plt.bar(coef_df['Feature'], coef_df['Lasso Coefficients'], alpha=0.7, label="Lasso", color='blue')
plt.bar(coef_df['Feature'], coef_df['Ridge Coefficients'], alpha=0.7, label="Ridge", color='red')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xticks(rotation=45)
plt.ylabel("Coefficient Value")
plt.title("Lasso vs. Ridge Coefficients")
plt.legend()
plt.show()
