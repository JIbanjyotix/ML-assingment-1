# importing important libraties
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# get data from saved csv
df = pd.read_csv('C:\\Users\\Jiban\\Desktop\Rental_Data.csv')
print(df.head())

# Checking null values
print(df.isnull().sum())