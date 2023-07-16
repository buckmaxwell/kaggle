# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# 1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
import os
import sys
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


from keras.optimizers import Adam
from kerastuner import RandomSearch


# append the path three directories up to import the variable store
sys.path.append("/Users/maxbuck/Documents/kagcomp")

from shared.state_store import StateStore
from shared.basic_hyper_model import BasicHyperModel


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
    df = pd.get_dummies(df, columns=["Sex", "Embarked", "Pclass", "Title"])

    # We save these as our test set will need to have the same columns
    state_store.set("column_names", df.columns.tolist())
    return df


def add_title(df):
    # full$Titles <- gsub("Dona|Lady|Madame|the Countess", "Lady", full$Titles)
    # full$Titles <- gsub("Don|Jonkheer|Sir", "Sir", full$Titles)

    df["Title"] = (
        df["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    )

    # Since Don, Jonkheer, and Sir are all of similar usage, and each represent
    # only one first-class man, I combined them into the category “Sir”. Dona,
    # Lady, Madame, and the Countess each only represent one first-class woman,
    # so I combined them into the category “Lady”.
    df["Title"] = df["Title"].replace(
        ["Dona", "Lady", "Madame", "the Countess"], "Lady"
    )
    df["Title"] = df["Title"].replace(["Don", "Jonkheer", "Sir"], "Sir")
    # df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    # df["Title"] = df["Title"].replace("Mme", "Mrs")
    return df

    # df["Title"] = (
    #    df["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    # )
    # return df


if __name__ == "__main__":
    state_store = StateStore()

    df_train = load_data()
    df_train = handle_missing_values(df_train, state_store)
    df_train = add_title(df_train)
    df_train = encode_categorical_variables(df_train, state_store)
    df_train = drop_useless_columns(df_train)
    identify_missing_values(df_train)  # Should be empty

    df_train.head()

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

    ####
    # Build the model with the optimal hyperparameters
    input_shape = (X_train.shape[1],)  # Shape of the input features
    hypermodel = BasicHyperModel(input_shape)

    # Then, we pass this HyperModel to the tuner
    tuner = RandomSearch(
        hypermodel,
        objective="val_accuracy",
        max_trials=30,
        directory="tunerlogs",
        project_name="titanic",
    )

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"The hyperparameter search is complete. The optimal number of units in the first densely-connected layer is {best_hps.get('units')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}."
    )
    model = Sequential()
    model.add(Dense(best_hps.get("units"), activation=best_hps.get("dense_activation")))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(learning_rate=best_hps.get("learning_rate")),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Retrain the model
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

    # Save the model
    # model.save("my_model")
    ####

    # Define the model
    # model = Sequential(
    #    [
    #        Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    #        Dense(32, activation="relu"),
    #        Dense(1, activation="sigmoid"),
    #    ]
    # )

    ## Compile the model
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    ## Train the model
    # history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

    # Save the model
    model.save(
        f"../models/{datetime.now().strftime('%Y%m%d%H%M%S')}_titanic_model.keras"
    )
