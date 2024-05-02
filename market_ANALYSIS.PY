import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load market data from CSV
market_data = pd.read_csv('market_data.csv')

# Add more features
market_data['EPS'] = np.random.randint(1, 5, market_data.shape[0])  # Adding EPS as random integers
market_data['Market Share'] = np.random.uniform(0.1, 0.9, market_data.shape[0])  # Adding Market Share as random floats

# Insights and Visualizations

# Correlation Matrix
plt.figure(figsize=(10, 8))
corr_matrix = market_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Market Data Features')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()

# Linear Regression to predict Stock Price based on Volume
X = market_data[['Volume']]
y = market_data['Stock_Price']
regressor = LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('Volume')
plt.ylabel('Stock Price')
plt.title('Linear Regression: Volume vs Stock Price')
plt.show()

# Bar chart for Market Share distribution
plt.figure(figsize=(10, 6))
sns.histplot(market_data['Market Share'], bins=20, kde=True, color='skyblue', edgecolor='black')
plt.xlabel('Market Share')
plt.ylabel('Frequency')
plt.title('Distribution of Market Share Among Companies')
plt.grid(True)
plt.show()

# Scatter plot of Revenue vs Net Income colored by Company
plt.figure(figsize=(10, 6))
sns.scatterplot(data=market_data, x='Revenue', y='Net_Income', hue='Company')
plt.xlabel('Revenue')
plt.ylabel('Net Income')
plt.title('Revenue vs Net Income for Different Companies')
plt.grid(True)
plt.legend(title='Company')
plt.show()

# Box plot of PE Ratio by Company
plt.figure(figsize=(10, 6))
sns.boxplot(data=market_data, x='Company', y='PE_Ratio')
plt.xlabel('Company')
plt.ylabel('PE Ratio')
plt.title('Distribution of PE Ratio Across Companies')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Histogram of EPS
plt.figure(figsize=(10, 6))
sns.histplot(data=market_data, x='EPS', bins=range(1, 6), discrete=True, kde=True, color='salmon', edgecolor='black')
plt.xlabel('EPS')
plt.ylabel('Frequency')
plt.title('Distribution of Earnings Per Share (EPS)')
plt.grid(True)
plt.show()
