import pickle
import pandas as pd

# Supponiamo che il modello sia stato salvato come model.pkl
model_file = "model.pkl"

# Carica il modello Pickle
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Esempio di nuovi dati per l'inferenza (ad esempio, da un CSV o in formato DataFrame)
# Nota: sostituisci questi dati con quelli che il tuo modello si aspetta
new_data = pd.DataFrame({
    'Pregnancies': [1],
    'PlasmaGlucose': [85],
    'DiastolicBloodPressure': [66],
    'TricepsThickness': [29],
    'SerumInsulin': [0],
    'BMI': [26.6],
    'DiabetesPedigree': [0.351],
    'Age': [31]
})

new_data = new_data.to_numpy()

# Esegui inferenza con il modello caricato
predictions = model.predict(new_data)

# Mostra i risultati
print(f"Predizioni: {predictions}")
