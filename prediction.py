import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# FAVC: Frequent consumption of high caloric food (Częste spożywanie wysokokalorycznej żywności)
# FCVC: Frequency of consumption of vegetables (Częstotliwość spożywania warzyw)
# NCP: Number of main meals per day (Liczba głównych posiłków dziennie)
# CAEC: Consumption of food between meals (Spożywanie jedzenia między posiłkami)
# SMOKE: Smoking habit (Nawyki palenia)
# CH2O: Daily water consumption (Dzienne spożycie wody)
# SCC: Calories consumption monitoring (Monitorowanie spożycia kalorii)
# FAF: Physical activity frequency (Częstotliwość aktywności fizycznej)
# TUE: Time using technology devices (Czas spędzony na korzystaniu z urządzeń technologicznych)
# CALC: Consumption of alcohol (Spożywanie alkoholu)
# MTRANS: Transportation method (Metoda transportu)
# NObeyesdad: Obesity level (Poziom otyłości)

# Wczytanie modelu i skalera
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

with open('label_encoder_target.pkl', 'rb') as le_target_file:
    label_encoder_target = pickle.load(le_target_file)

# --- WCZYTYWANIE DANYCH I PREDYKCJA ---
data = pd.read_csv('dane/ObesityDataSet_raw_and_data_sinthetic.csv')
dc = data.copy()

# Lista cech
feature_names = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
                 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
                 'CALC', 'MTRANS']

#Kolumny kategoryczne i numeryczne
categorical_cols = []
numeric_cols = []

for col in dc.columns:
    if pd.api.types.is_numeric_dtype(dc[col]):
        numeric_cols.append(col)
    elif pd.api.types.is_object_dtype(dc[col]):
        categorical_cols.append(col)

categorical_features = categorical_cols[:-1] #Kolumny kategoryczne bez kolumny z decyzją

#Wczytywanie danych
user_data = {}
print("Dane do predykcji: ")
for feature in feature_names:
    user_data[feature] = input(f"Podaj wartość dla {feature}:")

#numeric_cols, categorical_features - nazwy zmiennych

user_df = pd.DataFrame([user_data])

# Kodowanie kolumn kategorycznych
for col in categorical_features:
    user_df[col] = label_encoders[col].transform(user_df[col])

# Skalowanie cech
user_data_scaled = scaler.transform(user_df)

# Predykcja na danych wejściowych
user_prediction = model.predict(user_data_scaled)

# Konwersja predykcji do oryginalnych etykiet
user_prediction_label = label_encoder_target.inverse_transform(user_prediction)

# Wyświetlenie wyników
print("Predykcja dla wprowadzonych danych:")
print(user_prediction_label[0])