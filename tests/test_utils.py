from app.model import prediction   
import joblib

preprocess = joblib.load("models/scaling.pkl")
model = joblib.load("models/SVMClassifier.pkl")

def test_prediction():
    sample_input = [[1] * 63]  
    result = prediction(preprocess, model, sample_input)
    assert result is not None
    assert isinstance(result, str)
    expected_gestures = [
        "up", "down", "left", "right", "null"
    ]
    assert result in expected_gestures
