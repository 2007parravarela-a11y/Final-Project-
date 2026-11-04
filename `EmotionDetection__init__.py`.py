import requests
import json
import unittest

# ==========================================
# 1. FUNCIÓN DE DETECCIÓN DE EMOCIONES
# ==========================================
def emotion_detector(text_to_analyze):
    """
    Analiza el texto proporcionado usando Watson NLP para detectar emociones
    y maneja errores de entradas vacías o códigos de estado 400.
    """
    # Verificar si el texto está vacío o en blanco
    if not text_to_analyze or text_to_analyze.strip() == "":
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    header = {"grpc-metadata-mm-model-id": "watson_ru_nlp-emotion_workflow_bimodal_stock"}
    input_json = { "raw_document": { "text": text_to_analyze } }
    
    response = requests.post(url, json=input_json, headers=header)
    
    # Manejo de errores para códigos de estado 400
    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }
        
    formatted_response = json.loads(response.text)
    
    # Extraer las emociones de la respuesta
    emotions = formatted_response['emotionPredictions'][0]['emotion']
    anger = emotions['anger']
    disgust = emotions['disgust']
    fear = emotions['fear']
    joy = emotions['joy']
    sadness = emotions['sadness']
    
    # Encontrar la emoción dominante
    emotion_scores = {
        'anger': anger,
        'disgust': disgust,
        'fear': fear,
        'joy': joy,
        'sadness': sadness
    }
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    
    return {
        'anger': anger,
        'disgust': disgust,
        'fear': fear,
        'joy': joy,
        'sadness': sadness,
        'dominant_emotion': dominant_emotion
    }


# ==========================================
# 2. PRUEBAS UNITARIAS
# ==========================================
class TestEmotionDetector(unittest.TestCase):
    def test_emotion_detector(self):
        # Prueba para alegría (joy)
        result_1 = emotion_detector("I am glad this happened")
        self.assertEqual(result_1['dominant_emotion'], 'joy')

        # Prueba para ira (anger)
        result_2 = emotion_detector("I am really mad about this")
        self.assertEqual(result_2['dominant_emotion'], 'anger')

        # Prueba para asco (disgust)
        result_3 = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(result_3['dominant_emotion'], 'disgust')

        # Prueba para miedo (fear)
        result_4 = emotion_detector("I am so afraid about this")
        self.assertEqual(result_4['dominant_emotion'], 'fear')

        # Prueba para tristeza (sadness)
        result_5 = emotion_detector("I am so sad about this")
        self.assertEqual(result_5['dominant_emotion'], 'sadness')


# ==========================================
# 3. EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == '__main__':
    # Puedes probar la función directamente aquí si lo deseas
    print("Ejecutando prueba rápida de la función:")
    print(emotion_detector("I am glad this happened"))
    
    print("\nEjecutando pruebas unitarias...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)