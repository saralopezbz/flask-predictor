# Imagen base con Python 3.11
FROM python:3.11-slim

# Establecer directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos necesarios
COPY requirements.txt .
COPY modelo.pkl .
COPY app.py .
COPY train_model.py .
COPY test_api.py .
COPY postman_collection.json .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto donde correrá Flask
EXPOSE 5000

# Comando por defecto: ejecutar la API
CMD ["python", "app.py"]
