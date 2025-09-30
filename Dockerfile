# Imagen base oficial con Python 3.11
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios
COPY requirements.txt .
COPY modelo.pkl .
COPY app.py .
COPY train_model.py .
COPY test_api.py .
COPY tryapi.py .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que corre Flask
EXPOSE 5000

# Comando para ejecutar la API Flask
CMD ["python", "app.py"]
