from fastapi.testclient import TestClient
from app.main import app

def test_home():
    with TestClient(app) as client:
        response = client.get('/')
        assert response.status_code == 200
        assert "message" in response.json()

def test_health():
    with TestClient(app) as client:
        response = client.get('/health')
        assert response.status_code == 200
        json_resp = response.json()
        assert "status" in json_resp
        assert json_resp["status"] in ["healthy", "model file not loaded", "preprocessing file not loaded"]

def test_predict():
    dummy_input = {
        "landmarks": [0.1] * 63 
    }
    with TestClient(app) as client:
        response = client.post('/predict', json=dummy_input)
        assert response.status_code == 200
        json_resp = response.json()
        assert "prediction" in json_resp
        assert isinstance(json_resp["prediction"], str)