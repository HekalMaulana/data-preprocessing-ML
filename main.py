# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer as smp

dataset = pd.read_csv('Data.csv')
# Mendapatkan semua baris dengan ( : ) dan mendapatkan semua kolom kecuali kolom terakhir dengan ( :-1 ) sebagai
# fitur variabel
X = dataset.iloc[:, :-1].values
# Mendapatkan Kolom Terakhir sebagai vektor variabel
Y = dataset.iloc[:, -1].values

# Taking care of missing data
imputer = smp(missing_values=np.nan, strategy="mean")
