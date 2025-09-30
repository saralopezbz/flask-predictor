import requests
import json
 
# URL de la API
url = "http://127.0.0.1:5000/predict"
 
# Ejemplos de características conocidas del dataset Iris
data_examples = [
    {"features": [5.1, 3.5, 1.4, 0.2]},  # Setosa
    {"features": [7.0, 3.2, 4.7, 1.4]},  # Versicolor
    {"features": [6.3, 3.3, 6.0, 2.5]}   # Virginica
]
 
# Realizar las solicitudes POST
for i, data in enumerate(data_examples, start=1):
    response = requests.post(url, json=data)
    print(f"Ejemplo {i} - Predicción: {response.json()}")
 