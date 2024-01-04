# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer as smp

# Import dataset
dataset = pd.read_csv('Data.csv')
# Mendapatkan semua baris dengan ( : ) dan mendapatkan semua kolom kecuali kolom terakhir dengan ( :-1 ) sebagai
# fitur variabel
X = dataset.iloc[:, :-1].values
# Mendapatkan Kolom Terakhir sebagai vektor variabel
Y = dataset.iloc[:, -1].values

# Mengganti dataset yang memiliki angka 0 menjadi nan
dataset[dataset == 0] = np.nan

# Taking care of missing data
imputer = smp(missing_values=np.nan, strategy="mean")
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
