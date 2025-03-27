import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento del dataset
file_path = 'C:\\Users\\RBoiani\\OneDrive - BDO Italia SPA\\Desktop\\Banking progetto\\German_Credit_Dataset_normalized.xlsx'   # Inserisci qui il percorso del tuo file Excel
df_real = pd.read_excel(file_path)

# Separazione delle caratteristiche (X) e del target (y)
X = df_real.drop(columns=['ID', 'Risk'])
y = df_real['Risk']

# Suddivisione del dataset in training (70%) e validation (30%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Applicazione di SMOTE per bilanciare la classe minoritaria
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# Definizione della funzione per creare e addestrare la rete neurale
def create_and_train_model_balanced(X_train, y_train, X_val, y_val):
    best_recall = 0
    best_model = None
    optimal_layers = 0
    optimal_neurons = 0
    for layers in range(2, 6):  # Prova da 2 a 5 strati nascosti
        for neurons in [32, 64, 128, 256]:  # Prova con 32, 64, 128, 256 neuroni
            # Creiamo il modello di rete neurale
            model = Sequential()
            model.add(Dense(neurons, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
            model.add(Dropout(0.5))

            # Aggiunta degli strati nascosti
            for _ in range(layers - 1):
                model.add(Dense(neurons, activation='relu', kernel_regularizer=l2(0.001)))
                model.add(Dropout(0.5))

            model.add(Dense(1, activation='sigmoid'))  # Strato di output
            # Compilazione del modello
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Recall()])

            # Callback per ridurre il learning rate e prevenire overfitting
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.01, restore_best_weights=True)

            # Addestramento del modello con i dati bilanciati
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                epochs=100, batch_size=64, callbacks=[lr_scheduler, early_stopping], verbose=0)

            # Calcolo della recall sul set di validazione
            val_predictions = (model.predict(X_val) >= 0.5).astype(int)
            recall = recall_score(y_val, val_predictions)

            # Salvataggio del modello con la recall migliore
            if recall > best_recall:
                best_recall = recall
                best_model = model
                optimal_layers = layers
                optimal_neurons = neurons

    return best_model, best_recall, optimal_layers, optimal_neurons


# Creazione e addestramento del modello ottimale con il dataset bilanciato
model_balanced, best_recall_balanced, optimal_layers_balanced, optimal_neurons_balanced = create_and_train_model_balanced(
    X_resampled, y_resampled, X_val, y_val)

# Previsioni sul set di validazione
val_predictions_balanced = (model_balanced.predict(X_val) >= 0.5).astype(int)

# Calcolo della matrice di confusione
conf_matrix_balanced = confusion_matrix(y_val, val_predictions_balanced)

# Salvataggio dei risultati finali in un file CSV
results_df_balanced = pd.DataFrame({
    'ID': df_real.loc[X_val.index, 'ID'],
    'Valore Reale': y_val.values,
    'Predizione': val_predictions_balanced.flatten()
})
results_df_balanced.to_csv('C:\\Users\\RBoiani\\OneDrive - BDO Italia SPA\\Desktop\\Banking progetto\\risultati_prestiti.csv', index=False)

# Visualizzazione della matrice di confusione con i numeri
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_balanced, annot=True, fmt="d", cmap='Blues', xticklabels=['Approvato', 'Rifiutato'],
            yticklabels=['Approvato', 'Rifiutato'])
plt.title('Matrice di Confusione - Modello Bilanciato')
plt.xlabel('Previsione')
plt.ylabel('Valore Reale')
plt.show()

# Stampa dei risultati ottimali
print(f"Best Recall: {best_recall_balanced}")
print(f"Optimal Layers: {optimal_layers_balanced}")
print(f"Optimal Neurons per Layer: {optimal_neurons_balanced}")
print(f"Confusion Matrix:\n{conf_matrix_balanced}")