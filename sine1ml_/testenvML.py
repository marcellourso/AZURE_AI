import pickle
import numpy as np
from sklearn.pipeline import Pipeline

# Carica il modello dal file .pkl
model_path = 'model.pkl'  # Sostituisci con il percorso del tuo file .pkl
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    pickle.dump(model,f)

print(type(model))

if isinstance(model, Pipeline):
    print("È una pipeline.")
    # Verifica i passaggi della pipeline
    print(model.named_steps)
    # Accedi al modello finale della pipeline