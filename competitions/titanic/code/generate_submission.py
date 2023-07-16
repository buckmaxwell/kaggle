import sys
import os
from glob import glob
from datetime import datetime

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model


# append the path three directories up to import the variable store
sys.path.append("/Users/maxbuck/Documents/kagcomp")

from shared.state_store import StateStore
from competitions.titanic.code.run import (
    preprocess_data,
    identify_missing_values,
)

# Load model latest model (in ../models) and prefixed with a date like, 20230715201613_
glob_path = os.path.join("../models", "*.keras")
latest = None
for path in glob(glob_path):
    date_str = os.path.basename(path).split("_")[0]
    date = datetime.strptime(date_str, "%Y%m%d%H%M%S")
    if latest is None or date > latest[0]:
        latest = (date, path)

model_path = latest[1]
model_date_str = latest[0].strftime("%Y%m%d%H%M%S")
print(f"Using model {model_path}")
model = load_model(model_path)

# Load state store
state_store = StateStore(load=True)
print(f"State: {state_store.get()}")

# Load test data
test_df = pd.read_csv("../data/test.csv")

# Preprocess test data
# Be sure to fill missing values and drop columns exactly as you did with your training data
test_df = preprocess_data(test_df, state_store)


identify_missing_values(test_df)  # Should be empty

# Ensure that test data columns align with what the model was trained on
final_columns = state_store.get("column_names")
for col in final_columns:
    if col not in test_df.columns and col != "Survived":
        test_df[col] = 0

# Print number of columns in df_test
print(f"Number of columns in test data: {len(test_df.columns)}")

# Normalize test data into all floats
test_df = test_df.astype(np.float32)


# Predict survival for test data
predictions = model.predict(test_df)
predictions = [1 if pred >= 0.5 else 0 for pred in predictions]

# Create submission DataFrame
submission_df = pd.DataFrame(
    {
        "PassengerId": pd.read_csv("../data/test.csv")["PassengerId"],
        "Survived": predictions,
    }
)

# Save submission DataFrame to csv
submission_df.to_csv(f"../submissions/{model_date_str}_submission.csv", index=False)
