#Model uczenia maszynowego

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('dane/ObesityDataSet_raw_and_data_sinthetic.csv')

# Sprawdzenie braków danych
missing_values = data.isnull().sum()
print("Braki danych:")
print(missing_values)
