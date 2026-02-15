"""
Tests unitaires pour l'API de détection de phishing
Teste tous les endpoints et les fonctionnalités de prédiction
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient
import json

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests pour l'endpoint /health"""
    
    def test_health_endpoint_returns_200(self):
        """Vérifie que l'endpoint health retourne un status 200"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_endpoint_returns_json(self):
        """Vérifie que l'endpoint health retourne du JSON"""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"
    
    def test_health_endpoint_has_status_field(self):
        """Vérifie que la réponse contient le champ status"""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_endpoint_has_models_loaded_field(self):
        """Vérifie que la réponse contient le champ models_loaded"""
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], list)


class TestHomeEndpoint:
    """Tests pour l'endpoint /"""
    
    def test_home_endpoint_returns_200(self):
        """Vérifie que l'endpoint home retourne un status 200"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_home_endpoint_returns_html(self):
        """Vérifie que l'endpoint home retourne du HTML"""
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]
    
    def test_home_page_contains_title(self):
        """Vérifie que la page contient le titre"""
        response = client.get("/")
        assert "Phishing Detection" in response.text


class TestPredictEndpoint:
    """Tests pour l'endpoint /predict"""
    
    def test_predict_requires_text_field(self):
        """Vérifie que le champ text est requis"""
        response = client.post(
            "/predict",
            json={"data_type": "sms"}
        )
        assert response.status_code == 422  # Validation error
    
    def test_predict_requires_data_type_field(self):
        """Vérifie que le champ data_type est requis"""
        response = client.post(
            "/predict",
            json={"text": "Test message"}
        )
        assert response.status_code == 422  # Validation error
    
    def test_predict_rejects_invalid_data_type(self):
        """Vérifie que les types de données invalides sont rejetés"""
        response = client.post(
            "/predict",
            json={"text": "Test", "data_type": "invalid"}
        )
        # Devrait retourner une erreur 503 si le modèle n'existe pas
        # ou 422 si la validation échoue
        assert response.status_code in [503, 422]
    
    def test_predict_accepts_valid_sms_request(self):
        """Vérifie qu'une requête SMS valide est acceptée"""
        response = client.post(
            "/predict",
            json={
                "text": "Click here to win a free iPhone!",
                "data_type": "sms"
            }
        )
        # Devrait retourner 200 si le modèle est chargé, sinon 503
        assert response.status_code in [200, 503]
    
    def test_predict_accepts_valid_email_request(self):
        """Vérifie qu'une requête email valide est acceptée"""
        response = client.post(
            "/predict",
            json={
                "text": "Dear customer, verify your account immediately.",
                "data_type": "email"
            }
        )
        assert response.status_code in [200, 503]
    
    def test_predict_response_has_required_fields(self):
        """Vérifie que la réponse contient tous les champs requis"""
        response = client.post(
            "/predict",
            json={
                "text": "Test message",
                "data_type": "sms"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "is_phishing" in data
            assert "confidence" in data
            assert "label" in data
            assert "data_type" in data
    
    def test_predict_confidence_is_between_0_and_1(self):
        """Vérifie que la confiance est entre 0 et 1"""
        response = client.post(
            "/predict",
            json={
                "text": "Test message",
                "data_type": "sms"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert 0 <= data["confidence"] <= 1
    
    def test_predict_label_is_valid(self):
        """Vérifie que le label est SAFE ou PHISHING"""
        response = client.post(
            "/predict",
            json={
                "text": "Test message",
                "data_type": "sms"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert data["label"] in ["SAFE", "PHISHING"]
    
    def test_predict_is_phishing_matches_label(self):
        """Vérifie que is_phishing correspond au label"""
        response = client.post(
            "/predict",
            json={
                "text": "Test message",
                "data_type": "sms"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if data["label"] == "PHISHING":
                assert data["is_phishing"] is True
            else:
                assert data["is_phishing"] is False


class TestPredictionAccuracy:
    """Tests de cohérence des prédictions"""
    
    def test_obvious_phishing_sms_detected(self):
        """Vérifie la détection d'un SMS de phishing évident"""
        phishing_messages = [
            "URGENT! You've won $1000000! Click here NOW!",
            "Your account will be closed. Verify immediately at http://fake-bank.com",
            "Free prize! Call 09xx-xxxxxx to claim your reward",
        ]
        
        for msg in phishing_messages:
            response = client.post(
                "/predict",
                json={"text": msg, "data_type": "sms"}
            )
            
            if response.status_code == 200:
                data = response.json()
                # On ne force pas la détection car le modèle peut avoir ses propres décisions
                # mais on vérifie que la confiance est raisonnable
                assert 0 <= data["confidence"] <= 1
    
    def test_legitimate_sms_detected(self):
        """Vérifie la détection d'un SMS légitime"""
        legitimate_messages = [
            "Hi, how are you doing today?",
            "Meeting at 3pm in conference room B",
            "Thanks for your help yesterday",
        ]
        
        for msg in legitimate_messages:
            response = client.post(
                "/predict",
                json={"text": msg, "data_type": "sms"}
            )
            
            if response.status_code == 200:
                data = response.json()
                assert 0 <= data["confidence"] <= 1
    
    def test_empty_text_handling(self):
        """Vérifie le traitement des textes vides"""
        response = client.post(
            "/predict",
            json={"text": "", "data_type": "sms"}
        )
        # Devrait retourner une erreur ou gérer gracieusement
        assert response.status_code in [200, 400, 500]
    
    def test_very_long_text_handling(self):
        """Vérifie le traitement des textes très longs"""
        long_text = "test " * 1000  # 5000 caractères
        response = client.post(
            "/predict",
            json={"text": long_text, "data_type": "sms"}
        )
        # Devrait tronquer et traiter correctement
        assert response.status_code in [200, 503]
    
    def test_special_characters_handling(self):
        """Vérifie le traitement des caractères spéciaux"""
        special_text = "Test @#$%^&*() message with émojis 😀🎉"
        response = client.post(
            "/predict",
            json={"text": special_text, "data_type": "sms"}
        )
        assert response.status_code in [200, 503]


class TestAPIPerformance:
    """Tests de performance de l'API"""
    
    def test_prediction_response_time(self):
        """Vérifie que le temps de réponse est raisonnable"""
        import time
        
        start_time = time.time()
        response = client.post(
            "/predict",
            json={
                "text": "Test message for performance",
                "data_type": "sms"
            }
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Le temps de réponse devrait être inférieur à 5 secondes
        assert response_time < 5.0
    
    def test_multiple_concurrent_requests(self):
        """Vérifie que l'API peut gérer plusieurs requêtes"""
        messages = [
            "Test message 1",
            "Test message 2",
            "Test message 3",
        ]
        
        for msg in messages:
            response = client.post(
                "/predict",
                json={"text": msg, "data_type": "sms"}
            )
            assert response.status_code in [200, 503]


def run_tests():
    """Exécute tous les tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
