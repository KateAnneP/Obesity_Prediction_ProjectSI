# Model uczenia maszynowego

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Wczytywanie danych
data = pd.read_csv('dane/ObesityDataSet_raw_and_data_sinthetic.csv')
dc = data.copy()

# Sprawdzenie braków danych
missing_values = dc.isnull().sum()
# print("Braki danych:")
#print(missing_values)

# Sprawdzanie duplikatów
duplicates = dc.duplicated().sum()
print(f"Duplikaty: {duplicates}")
dc = dc.drop_duplicates()
print(f"Duplikaty po usunięciu: {dc.duplicated().sum()}")

# Kolumny kategoryczne i numeryczne
categorical_cols = []
numeric_cols = []

for col in dc.columns:
    if pd.api.types.is_numeric_dtype(dc[col]):
        numeric_cols.append(col)
    elif pd.api.types.is_object_dtype(dc[col]):
        categorical_cols.append(col)

categorical_features = categorical_cols[:-1] # Kolumny kategoryczne bez kolumny z decyzją

#--- Kodowanie danych ---
label_encoder_target = LabelEncoder()
dc['NObeyesdad_Encoded'] = label_encoder_target.fit_transform(dc['NObeyesdad'])
dc['NObeyesdad_Encoded'].value_counts()

# Kodowanie wszystkich kolumn kategorycznych
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    dc[col] = le.fit_transform(dc[col])
    label_encoders[col] = le

#--- Oddzielanie części cech i decyzji ---
X = dc.iloc[:, :-2].values
y = dc['NObeyesdad_Encoded'].values

#Skalowanie cech
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --- UCZENIE ---
#------------------------------------------------------------------------
# Model sieci
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

# Krosswalidacja
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
f1_scores = []
precisions = []
recalls = []
confusion_matrices = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Konwersja etykiet do kategorii
    if output_size > 1:
        y_train = tf.keras.utils.to_categorical(y_train, output_size)
        y_test = tf.keras.utils.to_categorical(y_test, output_size)

    # Inicjalizacja modelu
    model = create_model(input_size, hidden_size, output_size)

    # Kompilacja modelu
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy' if output_size > 1 else 'binary_crossentropy',
                  metrics=['accuracy'])

    # Trenowanie modelu
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0)

    # Testowanie modelu
    y_pred = model.predict(X_test)
    if output_size > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
    else:
        y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)
        y_test_classes = y_test

    accuracies.append(accuracy_score(y_test_classes, y_pred_classes))
    f1_scores.append(f1_score(y_test_classes, y_pred_classes, average='weighted'))
    precisions.append(precision_score(y_test_classes, y_pred_classes, average='weighted'))
    recalls.append(recall_score(y_test_classes, y_pred_classes, average='weighted'))

    # Obliczanie macierzy pomyłek
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    confusion_matrices.append(cm)

# Obliczenie średniej i odchylenia standardowego dla każdej miary
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

mean_precision = np.mean(precisions)
std_precision = np.std(precisions)

mean_recall = np.mean(recalls)
std_recall = np.std(recalls)

print(f'Średnia dokładność: {mean_accuracy:.4f} (± {std_accuracy:.4f})')
print(f'Średnia F1: {mean_f1:.4f} (± {std_f1:.4f})')
print(f'Średnia precyzja: {mean_precision:.4f} (± {std_precision:.4f})')
print(f'Średnia czułość: {mean_recall:.4f} (± {std_recall:.4f})')

# Wyświetlanie macierzy pomyłek dla każdej iteracji krosswalidacji
for i, cm in enumerate(confusion_matrices):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Macierz pomyłek dla fold {i + 1}')
    plt.xlabel('Predykcja')
    plt.ylabel('Rzeczywistość')
    plt.show()


# Zapisywanie modelu i skalera
import joblib

# Zapisywanie skalera
joblib.dump(scaler, 'scaler.joblib')

# Zapisywanie encoderów
joblib.dump(label_encoders, 'label_encoders.joblib')
joblib.dump(label_encoder_target, 'label_encoder_target.joblib')

# Zapisywanie modelu
model.save('model.keras')

