#Test różnych modeli uczenia maszynowego

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

#Wczytywanie danych
data = pd.read_csv('dane/ObesityDataSet_raw_and_data_sinthetic.csv')
dc = data.copy() #Kopia danych, jakby się coś zepsuło przypadkiem

# Sprawdzenie braków danych
missing_values = data.isnull().sum()
print("Braki danych:")
print(missing_values)

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

#Skalowanie cech
scaler = StandardScaler()
X = scaler.fit_transform(X)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- PORÓWNANIE MODELI ---
accuracies = []
models = []

def evaluate_model(model, X, y):
    fold_accuracies = []
    for train_index, val_index in skf.split(X, y):
        X_fold_train, X_fold_val = X[train_index], X[val_index]
        y_fold_train, y_fold_val = y[train_index], y[val_index]

        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        accuracy = accuracy_score(y_fold_val, y_pred)
        fold_accuracies.append(accuracy)

    return np.mean(fold_accuracies)

# 1. Regresja logistyczna
# Inicjalizacja i trening modelu regresji logistycznej
model = LogisticRegression(max_iter=1000)
accuracy = evaluate_model(model, X, y)
print("Dokładność regresji logistycznej:", accuracy)
models.append(model)
accuracies.append(accuracy)

#. 2. Drzewa decyzyjne
model = DecisionTreeClassifier()
accuracy = evaluate_model(model, X, y)
print("Dokładność drzew decyzyjnych:", accuracy)
models.append(model)
accuracies.append(accuracy)

# 3. Lasy losowe
model = RandomForestClassifier()
accuracy = evaluate_model(model, X, y)
print("Dokładność lasów losowych: ", accuracy)
models.append(model)
accuracies.append(accuracy)

# 4. Gradient boosting
model = GradientBoostingClassifier()
accuracy = evaluate_model(model, X, y)
print("Dokładność gradient boosting: ", accuracy)
models.append(model)
accuracies.append(accuracy)

# 5. Support Vector Machnines
model = SVC()
accuracy = evaluate_model(model, X, y)
print("Dokładność SVM:", accuracy)
models.append(model)
accuracies.append(accuracy)

# 6. K-najbliższych sąsiadów
model = KNeighborsClassifier()
accuracy = evaluate_model(model, X, y)
print("Dokładność k sąsiadów:", accuracy)
models.append(model)
accuracies.append(accuracy)

# 7. Sieć neuronowa
def create_model(input_size, hidden_size, output_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size, input_shape=(input_size,), activation='relu'),
        tf.keras.layers.Dense(hidden_size, activation='relu'),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])
    return model

# Parametry sieci
input_size = X.shape[1]
hidden_size = 64
output_size = len(np.unique(y))  # Liczba klas
num_epochs = 500
batch_size = 32
learning_rate = 0.001

def evaluate_neural_network(X, y):
    fold_accuracies = []
    for train_index, val_index in skf.split(X, y):
        X_fold_train, X_fold_val = X[train_index], X[val_index]
        y_fold_train, y_fold_val = y[train_index], y[val_index]

        # Konwersja etykiet do kategorii
        if output_size > 1:
            y_fold_train = tf.keras.utils.to_categorical(y_fold_train, output_size)
            y_fold_val = tf.keras.utils.to_categorical(y_fold_val, output_size)

        # Inicjalizacja modelu
        model = create_model(input_size, hidden_size, output_size)

        # Kompilacja modelu
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy' if output_size > 1 else 'binary_crossentropy',
                      metrics=['accuracy'])

        # Trenowanie modelu
        model.fit(X_fold_train, y_fold_train, epochs=num_epochs, batch_size=batch_size, verbose=0)

        # Testowanie modelu
        y_pred = model.predict(X_fold_val)
        if output_size > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_fold_val_classes = np.argmax(y_fold_val, axis=1)
        else:
            y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)
            y_fold_val_classes = y_fold_val

        accuracy_score(y_fold_val_classes, y_pred_classes)
        fold_accuracies.append(accuracy)

    return np.mean(fold_accuracies)


accuracy = evaluate_neural_network(X, y)
print("Dokładność SN:", accuracy)
models.append(None)
accuracies.append(accuracy)

# Podsumowanie dokładności
best_accuracy = accuracies[0]
for i in range(0, len(models)):
    if accuracies[i] > best_accuracy:
        best_accuracy = accuracies[i]
        best_model = models[i]
print(f"Najwyższą dokładność ma model: {best_model}, dokładność: {best_accuracy}")