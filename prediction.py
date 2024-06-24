import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

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
scaler = joblib.load('scaler.joblib')
label_encoders = joblib.load('label_encoders.joblib')
label_encoder_target = joblib.load('label_encoder_target.joblib')
model = load_model('model.keras')

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
# for feature in feature_names:
#     user_data[feature] = input(f"Podaj wartość dla {feature}:")

user_data[feature_names[0]] = input("Podaj swoją płeć (m/k): ")
if (user_data[feature_names[0]] == 'm'):
    user_data[feature_names[0]] = 'Male'
else:
    user_data[feature_names[0]] = 'Female'
user_data[feature_names[1]] = int(input("Podaj swój wiek: "))
user_data[feature_names[2]] = float(input("Podaj swoją wzrost w metrach (np. 1.76): "))
user_data[feature_names[3]] = float(input("Podaj swoją wagę w kg (np. 54.4): "))
user_data[feature_names[4]] = input("Czy miałeś/aś przypadki otyłości w rodzinie? (t/n): ")
if (user_data[feature_names[4]] == 't'):
    user_data[feature_names[4]] = 'yes'
else:
    user_data[feature_names[4]] = 'no'
user_data[feature_names[5]] = input("Czy spożywasz często wysokokaloryczną żywność? (t/n): ")
if (user_data[feature_names[5]] == 't'):
    user_data[feature_names[5]] = 'yes'
else:
    user_data[feature_names[5]] = 'no'
user_data[feature_names[6]] = int(input("Jak często spożywasz warzywa do posiłków? (1 - rzadko, 3 - często): "))
user_data[feature_names[7]] = int(input("Ile posiłków dziennie jesz? (1 - 1/2 posiłki, 2 - 3/4 posiłki, 3 - 5/6 posiłków, 4 - 7/8 posiłków): "))
user_data[feature_names[8]] = input("Czy podjadasz między posiłkami? (nigdy, czasem, często, zawsze): ")
if (user_data[feature_names[8]] == 'nigdy'):
    user_data[feature_names[8]] = 'no'
elif (user_data[feature_names[8]] == 'czasem'):
    user_data[feature_names[8]] = 'Sometimes'
elif (user_data[feature_names[8]] == 'często'):
    user_data[feature_names[8]] = 'Frequently'
else:
    user_data[feature_names[8]] = 'Always'
user_data[feature_names[9]] = input("Czy palisz papierosy? (t/n): ")
if (user_data[feature_names[9]] == 't'):
    user_data[feature_names[9]] = 'yes'
else:
    user_data[feature_names[9]] = 'no'
user_data[feature_names[10]] = int(input("Ile wody dziennie pijesz? (1 - 0/1 litr, 2 - 2/3 litry, 3 - 4 litry i więcej): "))
user_data[feature_names[11]] = input("Czy monitorujesz dzienne spożycie kalorii? (t/n):")
if (user_data[feature_names[11]] == 't'):
    user_data[feature_names[11]] = 'yes'
else:
    user_data[feature_names[11]] = 'no'
user_data[feature_names[12]] = (input("Jak często uprawiasz sporty? (nigdy, czasem, często, bardzo często): "))
if (user_data[feature_names[12]] == 'nigdy'):
    user_data[feature_names[12]] = 0
elif (user_data[feature_names[12]] == 'czasem'):
    user_data[feature_names[12]] = 1
elif (user_data[feature_names[12]] == 'często'):
    user_data[feature_names[12]] = 2
else:
    user_data[feature_names[12]] = 3
user_data[feature_names[13]] = input("Ile czasu spędzasz przed ekranem urządzeń elektronicznych? (mało, średnio, dużo): ")
if (user_data[feature_names[13]] == 'mało'):
    user_data[feature_names[13]] = 0
elif (user_data[feature_names[13]] == 'średnio'):
    user_data[feature_names[13]] = 1
elif (user_data[feature_names[13]] == 'dużo'):
    user_data[feature_names[13]] = 2
user_data[feature_names[14]] = input("Jak często spożywasz alkohol? (nigdy, czasem, często): ")
if (user_data[feature_names[14]] == 'nigdy'):
    user_data[feature_names[14]] = 'no'
elif (user_data[feature_names[14]] == 'czasem'):
    user_data[feature_names[14]] = 'Sometimes'
elif (user_data[feature_names[14]] == 'często'):
    user_data[feature_names[14]] = 'Frequently'
user_data[feature_names[15]] = int(input("Jakiego środka transportu zazwyczaj używasz? (1 - chodzenie pieszo, 2 - transport publiczny, 3 - samochód, 4 - motocykl): "))
if (user_data[feature_names[15]] == 1):
    user_data[feature_names[15]] = 'Walking'
elif (user_data[feature_names[15]] == 2):
    user_data[feature_names[15]] = 'Public_Transportation'
elif (user_data[feature_names[15]] == 3):
    user_data[feature_names[15]] = 'Automobile'
elif (user_data[feature_names[15]] == 4):
    user_data[feature_names[15]] = 'Motorbike'

print(user_data)


#numeric_cols, categorical_features - nazwy zmiennych

user_df = pd.DataFrame([user_data])

# Kodowanie kolumn kategorycznych
for col in categorical_features:
    user_df[col] = label_encoders[col].transform(user_df[col])

# Skalowanie cech
user_data_scaled = scaler.transform(user_df)

# Predykcja na nowych danych
user_prediction = model.predict(user_data_scaled)
user_prediction_class = np.argmax(user_prediction, axis=1)

# Konwersja predykcji do oryginalnych etykiet
user_prediction_label = label_encoder_target.inverse_transform(user_prediction_class)

# Wyświetlenie wyników
print("\n\nPredykcja dla wprowadzonych danych:")
print(user_prediction_label[0])