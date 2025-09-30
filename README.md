# 🌸 Flask Predictor - Clasificador de Iris

Este proyecto es una **API REST** construida con **Flask** para predecir la especie de una flor **Iris** a partir de sus medidas (largo y ancho de sépalo y pétalo).

Incluye:

-  Un modelo de Machine Learning entrenado (`RandomForestClassifier`)
-  Pruebas automáticas en `Python` con `requests`
- Evaluación continua con **GitHub Actions**
-  Contenedor **Docker** para despliegue local
-  Colección de pruebas en **Postman**

---

## 🚀 Cómo ejecutar el proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/saralopezbz/flask-predictor.git
cd flask-predictor


2. Instalar dependencias

pip install -r requirements.txt

3. Ejecutar el servidor Flask

python app.py

El servidor se inicia en: http://localhost:5000

Cómo hacer predicciones
Endpoint: /predict

Método: POST
Formato del JSON de entrada:

{
  "features": [5.1, 3.5, 1.4, 0.2]
}


Respuesta esperada:
{
  "prediction": "setosa",
  "confidence": 1.0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}

🧪 Pruebas automáticas
Ejecutar pruebas localmente
python test_api.py


Se validan:

Conexión al endpoint principal

Predicciones correctas

Manejo de errores comunes

Estructura de la respuesta