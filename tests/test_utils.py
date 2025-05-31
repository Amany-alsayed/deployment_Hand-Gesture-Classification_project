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
        "call", "dislike", "fist", "four", "like", "mute", "ok", 
        "one", "palm", "peace", "peace inv.", "rock", "stop", 
        "stop inv.", "three", "three 2", "two up", "two up inv.", "Unknown"
    ]
    assert result in expected_gestures
