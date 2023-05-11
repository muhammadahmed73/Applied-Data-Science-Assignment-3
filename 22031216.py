import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit


# Load the data and set the index
file_path = r'D:\Hertfordshire\Applied data science\Assignment 3\GDP\GDP.xls'
df = pd.read_excel(file_path, header=0, index_col=0)


# Drop the 'Country Code' column
df.drop(columns=['Country Code'], inplace=True)

# Check for missing values
print(df.isna().sum())

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Fill missing values with column means
df_filled = df.fillna(df.mean())

# Select the columns for clustering (features)
features = df_filled.iloc[:, 4:]  # Select all columns from 1960 to 2021

# Perform K-means clustering
k = 4  # Specify the number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(features)

# Add cluster classification as a new column
df_filled['Cluster'] = clusters

# Plotting
plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=clusters, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red')
plt.xlabel('GDP Values')
plt.ylabel('GDP Year')
plt.title('K-means Clustering of GDP')
plt.show()



gdp_values = [22.500, 30.343, 12.3432, 18.3400, 15.656500]  # Example GDP values
countries = ['United Kingdom', 'Portugal', 'India', 'United States', 'Nigeria']  # Example country names

# Create a pie chart
plt.figure(figsize=(10, 6))
plt.pie(gdp_values, labels=countries, autopct='%1.1f%%')
plt.title('GDP Distribution of Countries')
plt.axis('equal')

# Display the pie chart
plt.show()


# Fitting concept
import pandas as pd

# Load the dataset
file_path = r'D:\Hertfordshire\Applied data science\Assignment 3\GDP\GDP.xls'
df = pd.read_excel(file_path)

# Extract the required columns
columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'] + list(range(1960, 2022))
df = df[columns]

# Clean the data by dropping rows with missing values
df_cleaned = df.dropna()

# Print the cleaned dataset
print(df_cleaned.head())


print(df_cleaned['Country Name'].unique())


# Define the linear function
def linear_func(x, a, b):
    return a * x + b

# Prepare the data for fitting
x = np.array(df_cleaned.columns[4:], dtype=int)
y = np.array(df_cleaned.loc[df_cleaned['Country Name'] == 'China'].values[0][4:], dtype=float)

# Set upper and lower limits for the fitted parameters
lower_limits = [-np.inf, -np.inf]  # Lower limits for 'a' and 'b'
upper_limits = [np.inf, np.inf]    # Upper limits for 'a' and 'b'
bounds = (lower_limits, upper_limits)

# Fit the linear model using curve_fit with bounds
params, _ = curve_fit(linear_func, x, y, bounds=bounds)

# Extract the fitted parameters
a, b = params

# Predict the values for future years
future_years = np.arange(2022, 2043)
predicted_values = linear_func(future_years, a, b)

# Print the predicted values
print("Predicted values for future years:")
for year, value in zip(future_years, predicted_values):
    print(f"Year: {year}, Value: {value}")




# Plotting the predication 
# Define the linear function
def linear_func(x, a, b):
    return a * x + b

# Prepare the data for fitting
x = np.array(df_cleaned.columns[4:], dtype=int)
y = np.array(df_cleaned.loc[df_cleaned['Country Name'] == 'China'].values[0][4:], dtype=float)

# Fit the linear model using curve_fit
params, _ = curve_fit(linear_func, x, y)

# Extract the fitted parameters
a, b = params

# Predict the values for future years
future_years = np.arange(2022, 2043)
predicted_values = linear_func(future_years, a, b)

# Plot the original data and the predicted values
plt.plot(x, y, 'bo', label='Original Data')
plt.plot(future_years, predicted_values, 'r-', label='Predicted Values')

# Set plot title and labels
plt.title('Linear Fit - Predicted Values')
plt.xlabel('Year')
plt.ylabel('GDP')

# Add legend
plt.legend()

# Show the plot
plt.show()




