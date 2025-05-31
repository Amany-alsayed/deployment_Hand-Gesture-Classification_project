import numpy as np
import pandas as pd

def recenter_landmarks(landmarks):
  landmarks=landmarks.copy()
  wrist_x, wrist_y = landmarks.iloc[:,0], landmarks.iloc[:,1]
  # Subtract wrist coordinates from all landmarks
  for i in range(21):  # 21 landmarks
      landmarks.iloc[:,i * 3] -= wrist_x
      landmarks.iloc[:,i * 3 + 1] -= wrist_y

  return landmarks


def normalize_landmarks(landmarks):
    mid_finger_x, mid_finger_y = landmarks.iloc[:,9 * 3], landmarks.iloc[:,9 * 3 + 1]
    norm=np.sqrt(mid_finger_x**2,mid_finger_y**2)
    # Prevent division by zero
    norm[norm==0]=1
    # Normalize landmarks
    for i in range(21):  # 21 landmarks
        landmarks.iloc[:,i * 3] /= norm
        landmarks.iloc[:,i * 3 + 1] /= norm
    return landmarks

