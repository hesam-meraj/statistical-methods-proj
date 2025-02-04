from part2 import *

print("\n--- Model Performance ---")
print(f"Ridge: MSE={ridge_mse:.4f}, R²={ridge_r2:.4f}, Adjusted R²={ridge_adj_r2:.4f}")
print(f"Lasso: MSE={lasso_mse:.4f}, R²={lasso_r2:.4f}, Adjusted R²={lasso_adj_r2:.4f}")
print(f"Elastic Net: MSE={elastic_mse:.4f}, R²={elastic_r2:.4f}, Adjusted R²={elastic_adj_r2:.4f}")
