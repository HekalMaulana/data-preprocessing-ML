# # Import Libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.impute import SimpleImputer as smp

# # Import dataset
# dataset = pd.read_csv('Data.csv')
# # Mendapatkan semua baris dengan ( : ) dan mendapatkan semua kolom kecuali kolom terakhir dengan ( :-1 ) sebagai
# # fitur variabel
# X = dataset.iloc[:, :-1].values
# # Mendapatkan Kolom Terakhir sebagai vektor variabel
# Y = dataset.iloc[:, -1].values
#
# # Mengganti dataset yang memiliki angka 0 menjadi nan
# dataset[dataset == 0] = np.nan
# print(dataset)
#
# # Taking care of missing data
# # imputer = smp(missing_values=np.nan, strategy="mean")
# # X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer as smp

# Load the dataset
dataset = pd.read_csv('pima-indians-diabetes.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Identify missing data (assumes that missing data is represented as NaN)
X[X == 0] = np.nan

# Print the number of missing entries in each column
print(np.isnan(X).sum(axis=0))

# Configure an instance of the SimpleImputer class
imputer = smp(strategy="mean")

# Fit the imputer on the DataFrame
imputer.fit(X[:, :-1])

# Apply the transform to the DataFrame
X[:, :-1] = imputer.transform(X[:, :-1])

# Convert the array back to a DataFrame for better display
# X = pd.DataFrame(X, columns=dataset.columns[:-1])

# Set print options to display numbers without scientific notation
np.set_printoptions(suppress=True)

# Set Pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Print your updated matrix of features
print(X)
