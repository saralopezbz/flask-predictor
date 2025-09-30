# API de Predicción con Flask y Docker

Este proyecto consiste en una API desarrollada con Flask para hacer predicciones utilizando un modelo de machine learning entrenado con datos de Iris.

## 📦 Requisitos

* Docker instalado
* Python 3.11+ (solo si deseas ejecutar sin Docker)

## 🚀 Ejecutar con Docker

### 1. Descargar imagen desde Docker Hub

```bash
docker pull saralopezbz/flask-predictor
```

### 2. Ejecutar el contenedor

```bash
docker run -p 5000:5000 saralopezbz/flask-predictor
```

La API estará disponible en: [http://localhost:5000](http://localhost:5000)

### Endpoint principal

* **POST** `/predict`

  * Enviar un JSON con las características de una flor Iris para predecir su especie.
  * Ejemplo:

    ```json
    {
      "features": [5.1, 3.5, 1.4, 0.2]
    }
    ```

## 🔧 Para desarrolladores (opcional)

Si deseas correrlo sin Docker:

```bash
pip install -r requirements.txt
python app.py
```

## 🧪 Probar la API

Puedes usar Postman o ejecutar los scripts de prueba:

```bash
python tryapi.py
python test_api.py
```

---

## ⚙️ Integración Continua con GitHub Actions

Este repositorio incluye una acción de GitHub para verificar automáticamente que la API funcione correctamente.

### 📄 Archivo de configuración

* Ruta: `.github/workflows/test.yml`

### 🔍 ¿Qué hace?

1. Instala dependencias con `requirements.txt`
2. Ejecuta el servidor Flask
3. Lanza pruebas automatizadas con `test_api.py`

### ✅ Resultado

Cada push al repositorio ejecutará esta acción y mostrará si las pruebas pasan o fallan directamente en GitHub.

---

### 📤 Imagen en Docker Hub

📦 Imagen publicada en Docker Hub:
🔗 [https://hub.docker.com/r/saralopezbz/flask-predictor](https://hub.docker.com/r/saralopezbz/flask-predictor)

Comando para usarla directamente:

```bash
docker run -p 5000:5000 saralopezbz/flask-predictor
```

---

### 👩‍💻 Autora

**Sara López**

Proyecto para Evaluación Modular: *Desarrollo de una API REST para predicción con ML usando Flask y Docker.*
