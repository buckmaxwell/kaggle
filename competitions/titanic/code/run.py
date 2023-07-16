# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# 1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
import os
import sys
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# append the path three directories up to import the variable store
sys.path.append("/Users/maxbuck/Documents/kagcomp")

from shared.state_store import StateStore


def load_data():
    df = pd.read_csv("../data/train.csv")
    return df


def drop_useless_columns(df):
    df = df.drop(columns=["PassengerId", "Name", "Ticket"])
    return df


def handle_missing_values(df, state_store):
    df = df.drop(
        columns="Cabin"
    )  # Since the majority of rows in the set are missing Cabin we can ignore

    # Fill missing 'Age' values with median age (imputation) since the value is continuous; over nearly 1/5 of the rows are missing Age
    median_age = df["Age"].median()
    state_store.set("median_age", median_age)
    df["Age"] = df["Age"].fillna(state_store.get("median_age"))

    # Fill missing 'Embarked' values with the mode
    mode_embarked = df["Embarked"].mode()[0]
    state_store.set("mode_embarked", mode_embarked)
    df["Embarked"] = df["Embarked"].fillna(state_store.get("mode_embarked"))
    return df


def identify_missing_values(df):
    print(df.isnull().sum())


def encode_categorical_variables(df, state_store):
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=["Sex", "Embarked"])

    # We save these as our test set will need to have the same columns
    state_store.set("column_names", df.columns.tolist())
    return df


if __name__ == "__main__":
    state_store = StateStore()

    df_train = load_data()
    df_train = drop_useless_columns(df_train)
    df_train = handle_missing_values(df_train, state_store)
    df_train = encode_categorical_variables(df_train, state_store)
    identify_missing_values(df_train)  # Should be empty

    # Load and preprocess the test data
    # df_test = load_test_data()
    # df_test = handle_missing_values(df_test, state_store)
    # df_test = encode_categorical_variables(df_test, state_store)

    ## Make sure test data has the same columns as the training data
    # df_test = df_test.reindex(columns=state_store.get("column_names"), fill_value=0)

    # Separate features and target from the training data
    y_train = df_train["Survived"]
    X_train = df_train.drop("Survived", axis=1)

    # Split the training data into a training set and a validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Normalize the data into all floats
    X_train = X_train.astype("float32")
    y_train = y_train.astype("float32")
    X_val = X_val.astype("float32")
    y_val = y_val.astype("float32")

    # Define the model
    model = Sequential(
        [
            Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

    # Save the model
    model.save(
        f"../models/{datetime.now().strftime('%Y%m%d%H%M%S')}_titanic_model.keras"
    )
