import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import OLSInfluence
import statsmodels.api as sm

# Set random seed for reproducibility
np.random.seed(42)

# Generate random data
n = 30
x = np.random.rand(n) * 10  # Random x values between 0 and 10
y = 3 * x + np.random.randn(n) * 5  # Linear relationship with noise

# Convert to DataFrame
data = pd.DataFrame({'x': x, 'y': y})

# Fit linear regression model
model = LinearRegression()
model.fit(data[['x']], data['y'])

# Predict y values
data['y_pred'] = model.predict(data[['x']])

# Calculate residuals
data['residuals'] = data['y'] - data['y_pred']

# Fit model using statsmodels to get influence measures
X = sm.add_constant(data['x'])
ols_model = sm.OLS(data['y'], X).fit()
influence = OLSInfluence(ols_model)

# Add standardized and studentized residuals to the DataFrame
data['standardized_residuals'] = influence.resid_studentized_internal
data['studentized_residuals'] = influence.resid_studentized_external

# Plot the residuals
plt.figure(figsize=(12, 6))

# Plot standardized residuals
plt.subplot(1, 2, 1)
plt.scatter(data['x'], data['standardized_residuals'], color='blue', label='Standardized Residuals')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Standardized Residuals')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.legend()

# Plot studentized residuals
plt.subplot(1, 2, 2)
plt.scatter(data['x'], data['studentized_residuals'], color='red', label='Studentized Residuals')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Studentized Residuals')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.legend()

plt.tight_layout()
plt.show()
