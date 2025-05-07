import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# Prompt for dataset choice
choice = input("Use 'student' dataset or california 'housing' dataset? (student/housing): ").strip().lower()

if choice == 'student':
    dataset_slug = "neilhoneyman/sample-multiple-linear-regression-data"
    os.system(f'kaggle datasets download -d {dataset_slug} --unzip -p ./kaggle_data')
    
    path = r"C:\Users\mtilg\.kaggle\sample_regression_data.csv"
    if not os.path.exists(path):
        print("File not found. Please check the path.")
        exit()

    df = pd.read_csv(path)
    X = df[['Study_Hours', 'Sleep_Hours']]
    y = df['Performance_Score']
else:
    california_housing = fetch_california_housing()
    X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    y = pd.Series(california_housing.target)
    X = X[['MedInc', 'AveRooms']]

# Show correlation matrix
corr_matrix = X.corr()

# Calculate VIF
X_with_const = pd.concat([pd.Series([1]*len(X), name='Intercept'), X], axis=1)
vif = pd.DataFrame()
vif["Feature"] = X_with_const.columns
vif["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

# Display correlation and VIF in a separate window
corr_text = "Correlation Matrix:\n" + str(corr_matrix.round(3)) + "\n\n"
vif_text = "Variance Inflation Factors:\n" + str(vif.round(3))
full_text = corr_text + vif_text

fig_info = plt.figure(figsize=(10, 6))
fig_info.canvas.manager.set_window_title("Multicollinearity Info")
plt.axis('off')
plt.text(0.01, 0.95, full_text, fontsize=10, va='top', ha='left', wrap=True)
plt.show()

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Visualize in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], y_test, color='blue', label='Actual Data')

x1_range = np.linspace(X_test.iloc[:, 0].min(), X_test.iloc[:, 0].max(), 100)
x2_range = np.linspace(X_test.iloc[:, 1].min(), X_test.iloc[:, 1].max(), 100)
x1, x2 = np.meshgrid(x1_range, x2_range)
z = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)

ax.plot_surface(x1, x2, z, color='red', alpha=0.5, rstride=100, cstride=100)
ax.set_xlabel(X.columns[0])
ax.set_ylabel(X.columns[1])
ax.set_zlabel('Target')
ax.set_title('Multiple Linear Regression Best Fit Line (3D)')

plt.show()