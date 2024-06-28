# Model uczenia maszynowego

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.src.layers import Dense, Dropout, GRU, Concatenate
from keras.src.regularizers import regularizers
from numpy import concatenate
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
# print(f"Duplikaty: {duplicates}")
dc = dc.drop_duplicates()
# print(f"Duplikaty po usunięciu: {dc.duplicated().sum()}")

# Kolumny kategoryczne i numeryczne
categorical_cols = []
numeric_features = []

for col in dc.columns:
    if pd.api.types.is_numeric_dtype(dc[col]):
        numeric_features.append(col)
    elif pd.api.types.is_object_dtype(dc[col]):
        categorical_cols.append(col)

categorical_features = categorical_cols[:-1] # Kolumny kategoryczne bez kolumny z decyzją

#Kolejność klas decyzyjnych
order_of_classes = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
dc['NObeyesdad'] = pd.Categorical(dc['NObeyesdad'], categories=order_of_classes, ordered=True)
dc['NObeyesdad_Encoded'] = dc['NObeyesdad'].cat.codes   #Kodowanie klas decyzyjnych do numerów

# Kodowanie wszystkich kolumn kategorycznych
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    dc[col] = le.fit_transform(dc[col])
    label_encoders[col] = le

#--- Oddzielanie części cech i decyzji ---
X = dc.iloc[:, :-2].values
y = dc['NObeyesdad_Encoded'].values

# --- UCZENIE ---
# Funkcja do zwracania straty
class PrintProgress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {logs["loss"]:.4f}')

#------------------------------------------------------------------------
# Parametry sieci
input_size = X.shape[1]
num_epochs = 200
batch_size = 32
learning_rate = 0.001
hidden_size = 256
output_size = len(np.unique(y))  # Liczba klas

# Krosswalidacja
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
f1_scores = []
precisions = []
recalls = []
confusion_matrices = []

# Wybieranie tylko kolumn kategorycznych
categorical_features = data.select_dtypes(include=['category', 'object']).columns
# Liczba cech kategorycznych
num_categorical_features = len(categorical_features)

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Konwersja etykiet do kategorii
    if output_size > 1:
        y_train = tf.keras.utils.to_categorical(y_train, output_size)
        y_test = tf.keras.utils.to_categorical(y_test, output_size)

    # Skalowanie danych
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Definicja modelu
    input_categorical = Input(shape=(input_size,))
    input_numerical = Input(shape=(input_size,))

    dense1_numerical = Dense(64, activation='relu')(input_numerical)
    dense2_numerical = Dense(64, activation='relu')(dense1_numerical)

    concatenated_inputs = Concatenate()([input_categorical, input_numerical])

    combined_output = Concatenate()([concatenated_inputs, dense2_numerical])
    dense_combined = Dense(64, activation='relu')(combined_output)
    output = Dense(output_size, activation='softmax')(dense_combined)

    model = Model(inputs=[input_categorical, input_numerical], outputs=output)

    model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

    history = model.fit([X_train, X_train], y_train, epochs=num_epochs, batch_size=batch_size,
                               validation_split=0.1, verbose=0, callbacks=[PrintProgress()])

    # Testowanie modelu
    y_pred = model.predict([X_test, X_test])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)


    validation_loss = history.history['val_loss']
    final_validation_loss = validation_loss[-1]
    print(f"Final validation loss: {final_validation_loss}")

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
# for i, cm in enumerate(confusion_matrices):
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'Macierz pomyłek dla fold {i + 1}')
#     plt.xlabel('Predykcja')
#     plt.ylabel('Rzeczywistość')
#     plt.show()



