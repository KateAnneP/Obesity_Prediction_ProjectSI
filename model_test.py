#Test różnych modeli uczenia maszynowego

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#Wczytywanie danych
data = pd.read_csv('dane/ObesityDataSet_raw_and_data_sinthetic.csv')
dc = data.copy() #Kopia danych, jakby się coś zepsuło przypadkiem
# print(data)
# print("Początek danych: ", data.head(20))
# print("Koniec danych: ", data.tail(20))

# Sprawdzenie braków danych
missing_values = data.isnull().sum()
# print("Braki danych:")
#print(missing_values)

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
label_encoder = LabelEncoder()
dc['NObeyesdad_Encoded'] = label_encoder.fit_transform(dc['NObeyesdad'])
dc['NObeyesdad_Encoded'].value_counts()

#Kodowanie wszystkich kolumn kategorycznych
for col in categorical_features:
    #print(dc[col])
    dc[col] = label_encoder.fit_transform(dc[col])
    #print(dc[col])

#--- Oddzielanie części cech i decyzji ---
X = dc.iloc[:, :-2].values
y = dc['NObeyesdad_Encoded'].values

#Wydzielenie z danych części testowej i treningowej
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Skalowanie cech
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# --- PORÓWNANIE MODELI ---
accuracies = []
models = []

# 1. Regresja logistyczna
# Inicjalizacja i trening modelu regresji logistycznej
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Predykcja na zbiorze testowym
y_pred = model.predict(X_test)
# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność regresji logistycznej:", accuracy)
models.append(model)
accuracies.append(accuracy)

#. 2. Drzewa decyzyjne
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność drzew decyzyjnych:", accuracy)
models.append(model)
accuracies.append(accuracy)

# 3. Lasy losowe
model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność lasów losowych: ", accuracy)
models.append(model)
accuracies.append(accuracy)

# 4. Gradient boosting
model = GradientBoostingClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność gradient boosting: ", accuracy)
models.append(model)
accuracies.append(accuracy)

# 5. Support Vector Machnines
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność SVM:", accuracy)
models.append(model)
accuracies.append(accuracy)

# 6. K-najbliższych sąsiadów
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność k sąsiadów:", accuracy)
models.append(model)
accuracies.append(accuracy)

# 7. Sieci neuronowe
# model = Sequential()
# model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)
#
# _, accuracy = model.evaluate(X_test, y_test)
# print("Dokładność SN:", accuracy)
# models.append(model)
# accuracies.append(accuracy)

# Podsumowanie dokładności
best_accuracy = accuracies[0]
for i in range(0, len(models)):
    if accuracies[i] > best_accuracy:
        best_accuracy = accuracies[i]
        best_model = models[i]
print(f"Najwyższą dokładność ma model: {best_model}, dokładność: {best_accuracy}")
# Wnioski: Najwyższą dokładność wykazuje algorytm lasów losowych