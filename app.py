"""
API REST con Flask para clasificación usando modelo de Machine Learning.
Implementa endpoint para predicción con dataset Iris.
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from typing import Dict, Any, List, Union


# Configuración de la aplicación
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Variable global para el modelo
model_data = None


def load_model(model_path: str = "modelo.pkl") -> Dict[str, Any]:
    """
    Carga el modelo y metadatos desde archivo.
    
    Args:
        model_path: Ruta al archivo del modelo
        
    Returns:
        Dict con modelo y metadatos
        
    Raises:
        FileNotFoundError: Si no se encuentra el archivo del modelo
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Archivo del modelo no encontrado: {model_path}")
    
    return joblib.load(model_path)


def validate_input(data: Dict[str, Any]) -> tuple[bool, Union[str, List[float]]]:
    """
    Valida los datos de entrada de la solicitud.
    
    Args:
        data: Diccionario con datos de la solicitud
        
    Returns:
        Tuple (es_válido, mensaje_error_o_features)
    """
    # Verificar que existe la clave 'features'
    if 'features' not in data:
        return False, "Falta la clave 'features' en el JSON"
    
    features = data['features']
    
    # Verificar que features es una lista
    if not isinstance(features, list):
        return False, "El campo 'features' debe ser una lista"
    
    # Verificar longitud correcta (dataset Iris tiene 4 características)
    expected_length = model_data['feature_count']
    if len(features) != expected_length:
        return False, f"Se esperan {expected_length} características, se recibieron {len(features)}"
    
    # Verificar que todos los elementos son numéricos
    try:
        numeric_features = [float(x) for x in features]
    except (ValueError, TypeError):
        return False, "Todas las características deben ser valores numéricos"
    
    # Verificar que no hay valores infinitos o NaN
    if any(not np.isfinite(x) for x in numeric_features):
        return False, "Las características no pueden ser infinitas o NaN"
    
    return True, numeric_features


@app.route('/', methods=['GET'])
def home():
    """
    Endpoint principal que confirma que la API está funcionando.
    
    Returns:
        JSON con mensaje de confirmación
    """
    return jsonify({
        "message": "API lista",
        "status": "operational",
        "model_info": {
            "target_classes": model_data['target_names'].tolist() if model_data else [],
            "feature_count": model_data['feature_count'] if model_data else 0
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para realizar predicciones.
    Espera un JSON con formato: {"features": [val1, val2, val3, val4]}
    
    Returns:
        JSON con la predicción o mensaje de error
    """
    try:
        # Verificar que el contenido es JSON
        if not request.is_json:
            return jsonify({
                "error": "Content-Type debe ser application/json"
            }), 400
        
        # Obtener datos del request
        data = request.get_json()
        
        if data is None:
            return jsonify({
                "error": "JSON inválido o vacío"
            }), 400
        
        # Validar entrada
        is_valid, result = validate_input(data)
        
        if not is_valid:
            return jsonify({
                "error": result
            }), 400
        
        # Convertir a array numpy
        features_array = np.array(result).reshape(1, -1)
        
        # Realizar predicción
        prediction = model_data['model'].predict(features_array)
        prediction_proba = model_data['model'].predict_proba(features_array)
        
        # Obtener nombre de la clase predicha
        predicted_class = model_data['target_names'][prediction[0]]
        
        # Crear respuesta con información adicional
        response = {
            "prediction": predicted_class,
            "prediction_id": int(prediction[0]),
            "confidence": float(np.max(prediction_proba)),
            "probabilities": {
                class_name: float(prob)
                for class_name, prob in zip(
                    model_data['target_names'], 
                    prediction_proba[0]
                )
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"Error en predicción: {str(e)}")
        return jsonify({
            "error": "Error interno del servidor"
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Manejo de rutas no encontradas."""
    return jsonify({
        "error": "Endpoint no encontrado",
        "available_endpoints": ["/", "/predict"]
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Manejo de métodos HTTP no permitidos."""
    return jsonify({
        "error": "Método no permitido para este endpoint"
    }), 405


def initialize_app():
    """
    Inicializa la aplicación cargando el modelo.
    """
    global model_data
    
    try:
        print("Cargando modelo...")
        model_data = load_model()
        print("Modelo cargado exitosamente")
        print(f"Clases disponibles: {model_data['target_names']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Por favor, ejecuta primero 'python train_model.py' para entrenar el modelo")
        exit(1)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        exit(1)


if __name__ == '__main__':
    # Inicializar aplicación
    initialize_app()
    
    # Configuración para desarrollo
    print("Iniciando servidor Flask...")
    print("API disponible en: http://localhost:5000")
    print("Endpoint de predicción: http://localhost:5000/predict")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )