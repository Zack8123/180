# Import necessary libraries
import pandas as pd  # Import pandas library for data manipulation
from sklearn.linear_model import LinearRegression  # Import LinearRegression model
from sklearn import metrics  # Import metrics module for evaluation
import statsmodels.api as sm  # Import statsmodels library for advanced statistics

# Read the data from the spreadsheet
data = pd.read_excel(r'C:\Users\KoenigZA31\Downloads\Restaurant Revenue.xlsx')

# Separate the features (independent variables) and target variable (dependent variable)
X = data[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 'Average_Customer_Spending', 'Promotions', 'Reviews']]
y = data['Monthly_Revenue']

# Create a linear regression model
model = LinearRegression()

# Fit the model to the entire dataset
model.fit(X, y)

# Predict the monthly revenue for all data points
y_pred = model.predict(X)

# Output the predicted monthly revenue along with the actual monthly revenue
for pred, actual in zip(y_pred, y):
    print(f'Predicted Monthly Revenue: ${pred:.2f}, Actual Monthly Revenue: ${actual:.2f}')

# Compute and output the final r^2 value (coefficient of determination)
r_squared = metrics.r2_score(y, y_pred)
print(f'Final R^2 value: {r_squared:.3f}')

# Additional step: Obtain regression statistics using statsmodels
# Add a constant term to the features (intercept)
X = sm.add_constant(X)
# Fit the regression model using statsmodels
model_sm = sm.OLS(y, X).fit()
# Output the regression summary
print(model_sm.summary())
print("Go Brewers!")
print("BEANS")
