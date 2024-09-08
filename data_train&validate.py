import numpy as np
import pandas as pd

# Numero di campioni
nsamples = 1000

# Genera angoli in radianti (x)
x_values = np.linspace(0, 2 * np.pi, nsamples)

# Genera valori di seno con rumore (y)
y_values = np.sin(x_values) + 0.1 * np.random.randn(nsamples)

# Crea un DataFrame
data = pd.DataFrame({'x': x_values, 'y': y_values})

# Dividi in training (80%) e validation (20%)
train_size = int(0.8 * nsamples)
train_data = data[:train_size]
validation_data = data[train_size:]

# Salva i dati su file CSV
train_data.to_csv('train_data.csv', index=False)
validation_data.to_csv('validation_data.csv', index=False)

print("Dati di training e validation generati e salvati.")
