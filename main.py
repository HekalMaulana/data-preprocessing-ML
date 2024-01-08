# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer as smp
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Import dataset
dataset = pd.read_csv('Data.csv')
# Mendapatkan semua baris dengan ( : ) dan mendapatkan semua kolom kecuali kolom terakhir dengan ( :-1 ) sebagai
# fitur variabel
X = dataset.iloc[:, :-1].values
# Mendapatkan Kolom Terakhir sebagai vektor variabel
Y = dataset.iloc[:, -1].values

# Mengganti dataset yang memiliki angka 0 menjadi nan
# dataset[dataset == 0] = np.nan
# print(dataset)

# Taking care of missing data
imputer = smp(missing_values=np.nan, strategy="mean")
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Encoding Categorical Data

# Encoding Independent Variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

print(X)

# Encoding Dependent Variable
le = LabelEncoder()
Y = le.fit_transform(Y)

print(Y)

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
