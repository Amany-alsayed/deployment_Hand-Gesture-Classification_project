from app.utils import recenter_landmarks,normalize_landmarks
import pandas as pd
def prediction(preprocess,model,data):
    gesture_labels = {
    0: "call", 1: "dislike", 2: "fist", 3: "four", 4: "like",
    5: "mute", 6: "ok", 7: "one", 8: "palm", 9: "peace",
    10: "peace inv.", 11: "rock", 12: "stop", 13: "stop inv.",
    14: "three", 15: "three 2", 16: "two up", 17: "two up inv."
    }
    feature_names = [f"{axis}{i+1}" for i in range(21) for axis in ["x", "y", "z"]]
    landmarks_df = pd.DataFrame(data, columns=feature_names)
    landmarks_df = recenter_landmarks(landmarks_df)
    landmarks_df = normalize_landmarks(landmarks_df)
    landmarks_scaled = preprocess.transform(landmarks_df)
    prediction = model.predict(landmarks_scaled)[0]
    gesture_name = gesture_labels.get(int(prediction), "Unknown")
    return gesture_name