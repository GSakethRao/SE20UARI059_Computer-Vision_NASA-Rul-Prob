#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
# Ensure to adjust the file path accordingly
file_path = 'FT004.txt'
data = pd.read_csv(file_path, sep=' ', header=None)

# Basic preprocessing
data = data.drop([26, 27], axis=1)  # Removing empty columns
data.columns = ['unit', 'cycle'] + [f'sensor_{i}' for i in range(1, 22)]  # Naming the columns

# Calculating the RUL (Remaining Useful Life) for each row
max_cycle = data.groupby('unit')['cycle'].max()
result_frame = data.merge(max_cycle.reset_index(), left_on='unit', right_on='unit', how='left')
data['RUL'] = result_frame['cycle_y'] - result_frame['cycle_x']

# Features and target definition
features = data.drop('RUL', axis=1)
target = data['RUL']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Normalize features (important for regression models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Model prediction and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse}')

# Now you can use model.predict(new_data) to predict RUL on new, unseen data


# In[ ]:




