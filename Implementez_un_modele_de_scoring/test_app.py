import unittest
from app import app  # Import your Flask application

class TestFlaskApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = app.test_client()  # Create a test client
        cls.app.testing = True  # Enable testing mode

    def test_predict_valid_client(self):
        # Test with a valid client ID
        response = self.app.post('/predict', json={'SK_ID_CURR': 264862})  
        self.assertEqual(response.status_code, 200)
        self.assertIn('decision', response.get_json())  # Check that the 'decision' key is in the response

    def test_predict_valid_client_2(self):
        test_data = {
            'SK_ID_CURR': 264862  # Remplacez par un ID client valide
        }
        
        # Vérifier que les données de test sont au bon format
        self.assertIn('SK_ID_CURR', test_data)
        self.assertIsInstance(test_data['SK_ID_CURR'], int)
    
        response = self.client.post('/predict', json=test_data)
        self.assertEqual(response.status_code, 200)

    #def test_predict_invalid_client(self):
        # Test with an invalid client ID
        #response = self.app.post('/predict', json={'SK_ID_CURR': 99999})  # ID that does not exist
        #self.assertEqual(response.status_code, 404)
        #self.assertIn('error', response.get_json())  # Check that the 'error' key is in the response

    #def test_predict_missing_client_id(self):
        # Test without the client ID
        #response = self.app.post('/predict', json={})
        #self.assertEqual(response.status_code, 400)
        #self.assertIn('error', response.get_json())  # Check that the 'error' key is in the response

if __name__ == '__main__':
    unittest.main()  # Run the tests
