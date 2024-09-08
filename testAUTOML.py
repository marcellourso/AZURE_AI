import numpy as np
import azure.ai.ml
import azure.ai
import pickle  # o joblib se preferisci joblib
import joblib
import matplotlib.pyplot as plt


# 1. Carica il modello salvato in formato .pkl
with open('model.pkl', 'rb') as f:  # Sostituisci con il percorso corretto
    model = pickle.load(f)

# 2. Genera valori di input da 0 a 6.28 (0 a 2π) a passi di 0.1 radianti
radians = np.arange(0, 6.28, 0.1).reshape(-1, 1)  # Aggiungi la dimensione corretta

# 3. Esegui inferenze sul modello per ciascun valore di input
predictions = model.predict(radians)

# 4. Crea il grafico delle inferenze
plt.plot(radians, predictions, label='Predizioni del modello')
plt.xlabel('Radianti')
plt.ylabel('Valore predetto (Seno)')
plt.title('Inferenze del modello da 0 a 6.28 radianti')
plt.legend()
plt.grid(True)
plt.show()
