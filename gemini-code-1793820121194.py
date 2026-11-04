from EmotionDetection.emotion_detection import emotion_detector
import unittest

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

if __name__ == '__main__':
    unittest.main()