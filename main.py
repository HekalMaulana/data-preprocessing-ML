# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
# Mendapatkan semua baris dengan ( : ) dan mendapatkan semua kolom kecuali kolom terakhir dengan ( :-1 )
x = dataset.iloc[:, :-1]
