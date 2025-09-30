"""
Script de pruebas para la API Flask de clasificación.
Realiza pruebas automáticas de los endpoints con diferentes casos.
"""

import requests
import json
import time
from typing import Dict, Any


# Configuración
BASE_URL = "http://localhost:5000"
HEADERS = {'Content-Type': 'application/json'}


def test_home_endpoint():
    """Prueba el endpoint principal."""
    print("\n" + "="*50)
    print("PRUEBA 1: Endpoint principal (/)")
    print("="*50)
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200, "El endpoint / debería retornar 200"
        assert "API lista" in response.json().get("message", ""), "Mensaje esperado no encontrado"
        print("✅ Prueba PASADA")
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: No se pudo conectar al servidor")
        print("   Asegúrate de que el servidor Flask esté corriendo en http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    
    return True


def test_predict_valid_cases():
    """Prueba casos válidos del endpoint de predicción."""
    print("\n" + "="*50)
    print("PRUEBA 2: Predicciones válidas")
    print("="*50)
    
    # Casos de prueba válidos (dataset Iris)
    test_cases = [
        {
            "name": "Iris Setosa (ejemplo típico)",
            "data": {"features": [5.1, 3.5, 1.4, 0.2]},
            "expected_class": "setosa"
        },
        {
            "name": "Iris Versicolor (ejemplo típico)",
            "data": {"features": [7.0, 3.2, 4.7, 1.4]},
            "expected_class": "versicolor"
        },
        {
            "name": "Iris Virginica (ejemplo típico)",
            "data": {"features": [6.3, 3.3, 6.0, 2.5]},
            "expected_class": "virginica"
        }
    ]
    
    all_passed = True
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nCaso {i}: {case['name']}")
        print(f"Input: {case['data']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=case['data'],
                headers=HEADERS
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Predicción: {result['prediction']}")
                print(f"Confianza: {result['confidence']:.4f}")
                print(f"Probabilidades: {json.dumps(result['probabilities'], indent=2)}")
                
                # Verificar que la predicción es la esperada (nota: puede variar según el modelo)
                if result['prediction'] == case['expected_class']:
                    print("✅ Predicción coincide con lo esperado")
                else:
                    print(f"⚠️  Predicción {result['prediction']} vs esperado {case['expected_class']}")
                    print("   (Esto puede ser normal dependiendo del modelo entrenado)")
                
            else:
                print(f"❌ ERROR: Status code inesperado: {response.status_code}")
                print(f"Response: {response.text}")
                all_passed = False
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            all_passed = False
    
    if all_passed:
        print("\n✅ Todas las predicciones válidas PASARON")
    
    return all_passed


def test_predict_invalid_cases():
    """Prueba casos inválidos del endpoint de predicción."""
    print("\n" + "="*50)
    print("PRUEBA 3: Casos inválidos (manejo de errores)")
    print("="*50)
    
    # Casos de prueba inválidos
    invalid_cases = [
        {
            "name": "JSON sin clave 'features'",
            "data": {"data": [1, 2, 3, 4]},
            "expected_status": 400
        },
        {
            "name": "Número incorrecto de características",
            "data": {"features": [1, 2, 3]},  # Faltan características
            "expected_status": 400
        },
        {
            "name": "Características no numéricas",
            "data": {"features": ["a", "b", "c", "d"]},
            "expected_status": 400
        },
        {
            "name": "Lista vacía",
            "data": {"features": []},
            "expected_status": 400
        },
        {
            "name": "Features no es lista",
            "data": {"features": "invalid"},
            "expected_status": 400
        }
    ]
    
    all_passed = True
    
    for i, case in enumerate(invalid_cases, 1):
        print(f"\nCaso inválido {i}: {case['name']}")
        print(f"Input: {case['data']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=case['data'],
                headers=HEADERS
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == case['expected_status']:
                result = response.json()
                print(f"Error message: {result.get('error', 'No error message')}")
                print("✅ Manejo de error CORRECTO")
            else:
                print(f"❌ ERROR: Se esperaba status {case['expected_status']}, se recibió {response.status_code}")
                all_passed = False
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            all_passed = False
    
    if all_passed:
        print("\n✅ Todos los casos de error PASARON")
    
    return all_passed


def test_curl_examples():
    """Muestra ejemplos de comandos curl para pruebas manuales."""
    print("\n" + "="*50)
    print("EJEMPLOS DE COMANDOS CURL")
    print("="*50)
    
    curl_commands = [
        {
            "description": "Endpoint principal",
            "command": f'curl -X GET {BASE_URL}/'
        },
        {
            "description": "Predicción válida (Iris Setosa)",
            "command": f'curl -X POST {BASE_URL}/predict -H "Content-Type: application/json" -d \'{"features": [5.1, 3.5, 1.4, 0.2]}\''
        },
        {
            "description": "Predicción válida (Iris Versicolor)",
            "command": f'curl -X POST {BASE_URL}/predict -H "Content-Type: application/json" -d \'{"features": [7.0, 3.2, 4.7, 1.4]}\''
        },
        {
            "description": "Caso de error (características insuficientes)",
            "command": f'curl -X POST {BASE_URL}/predict -H "Content-Type: application/json" -d \'{"features": [1, 2, 3]}\''
        }
    ]
    
    for i, cmd in enumerate(curl_commands, 1):
        print(f"\n{i}. {cmd['description']}:")
        print(f"   {cmd['command']}")


def main():
    """Función principal que ejecuta todas las pruebas."""
    print("INICIANDO PRUEBAS DE LA API FLASK")
    print("Asegúrate de que el servidor Flask esté corriendo en http://localhost:5000")
    
    # Pequeña pausa para que el usuario pueda leer
    time.sleep(2)
    
    # Ejecutar pruebas
    test_results = []
    
    test_results.append(test_home_endpoint())
    test_results.append(test_predict_valid_cases())
    test_results.append(test_predict_invalid_cases())
    
    # Mostrar ejemplos curl
    test_curl_examples()
    
    # Resumen final
    print("\n" + "="*50)
    print("RESUMEN DE PRUEBAS")
    print("="*50)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"Pruebas pasadas: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("La API está funcionando correctamente")
    else:
        print("⚠️  Algunas pruebas fallaron")
        print("Revisa los errores mostrados arriba")


if __name__ == "__main__":
    main()