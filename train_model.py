"""
Script de entrenamiento del modelo de clasificación.
Entrena un modelo RandomForest con el dataset Iris y lo guarda.
"""

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_iris_model():
    """
    Entrena un modelo RandomForest con el dataset Iris.
    
    Returns:
        tuple: (modelo entrenado, accuracy del test, nombres de clases)
    """
    print("Cargando dataset Iris...")
    
    # Cargar dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Clases disponibles: {iris.target_names}")
    
    # Dividir en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Entrenando modelo RandomForest...")
    
    # Entrenar modelo
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=3
    )
    
    model.fit(X_train, y_train)
    
    # Evaluar modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy en test: {accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    return model, accuracy, iris.target_names


def save_model(model, target_names, filename="modelo.pkl"):
    """
    Guarda el modelo entrenado y metadatos.
    
    Args:
        model: Modelo entrenado
        target_names: Nombres de las clases
        filename: Nombre del archivo donde guardar
    """
    model_data = {
        'model': model,
        'target_names': target_names,
        'feature_count': 4,  # Dataset Iris tiene 4 características
        'feature_names': [
            'sepal_length', 'sepal_width', 
            'petals_length', 'petal_width'
        ]
    }
    
    joblib.dump(model_data, filename)
    print(f"\nModelo guardado en: {filename}")


if __name__ == "__main__":
    # Entrenar modelo
    model, accuracy, target_names = train_iris_model()
    
    # Guardar modelo
    save_model(model, target_names)
    
    print("\n¡Entrenamiento completado exitosamente!")
    print(f"El modelo está listo para usar en la API Flask.")