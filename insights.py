from imp_lib import *
# Statical data
print(df.describe(include='all'))

# Data type of columns
print(df.dtypes)
# information about data
print(df.info())

# index()
print(df.index)

# column data
print(df.columns)

# unique values in the dataset
print(df.value_counts())

# Max values of all columns
print(df.apply(np.max))