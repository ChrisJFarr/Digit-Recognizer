from keras.models import load_model
import numpy as np
import pandas as pd

# Load model
model = load_model("model_batch.h5")

# Load test data
test = np.load("test_data/test_data.npy") / 255.

# Perform predictions
predictions = [np.argmax(pred) for pred in model.predict(test)]

# Create submission file
submission = pd.read_csv("sample_submission.csv")
submission["Label"] = predictions

# Save submission
submission.to_csv("submission.csv", index=False)
