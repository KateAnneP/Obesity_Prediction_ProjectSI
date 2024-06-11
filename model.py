#Model uczenia maszynowego

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Wczytywanie danych
data = pd.read_csv('dane/ObesityDataSet_raw_and_data_sinthetic.csv')
dc = data.copy() #Kopia danych, jakby się coś zepsuło przypadkiem
# print(data)
# print("Początek danych: ", data.head(20))
# print("Koniec danych: ", data.tail(20))

# Sprawdzenie braków danych
missing_values = dc.isnull().sum()
# print("Braki danych:")
#print(missing_values)

#Sprawdzanie duplikatów
duplicates = dc.duplicated().sum()
print(f"Duplikaty: {duplicates}")
dc = dc.drop_duplicates()
print(dc.duplicated().sum())

#Kolumny kategoryczne i numeryczne
categorical_cols = []
numeric_cols = []

for col in dc.columns:
    if pd.api.types.is_numeric_dtype(dc[col]):
        numeric_cols.append(col)
    elif pd.api.types.is_object_dtype(dc[col]):
        categorical_cols.append(col)

# print(f"Kolumny kategoryczne: {categorical_cols}")
# print(f"Kolumny numeryczne: {numeric_cols}")

categorical_features = categorical_cols[:-1] #Kolumny kategoryczne bez kolumny z decyzją

#--- Kodowanie danych ---
label_encoder_target = LabelEncoder()
dc['NObeyesdad_Encoded'] = label_encoder_target.fit_transform(dc['NObeyesdad'])
dc['NObeyesdad_Encoded'].value_counts()

#Kodowanie wszystkich kolumn kategorycznych
label_encoders = {}

for col in categorical_features:
    #print(dc[col])
    #dc[col] = label_encoder.fit_transform(dc[col])
    #print(dc[col])
    le = LabelEncoder()
    dc[col] = le.fit_transform(dc[col])
    label_encoders[col] = le

#--- Oddzielanie części cech i decyzji ---
X = dc.iloc[:, :-2].values
y = dc['NObeyesdad_Encoded'].values

#Wydzielenie z danych części testowej i treningowej
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Skalowanie cech
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- UCZENIE ---
# Model - Lasy losowe
model = RandomForestClassifier()

# Cross-walidacja
cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # 5-fold cross-walidacja
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność lasów losowych: ", accuracy)

# Zapisywanie modelu i skalera
import pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)

with open('label_encoder_target.pkl', 'wb') as le_target_file:
    pickle.dump(label_encoder_target, le_target_file)