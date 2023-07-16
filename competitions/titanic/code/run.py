# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# 1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
import os
import sys
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model


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


def get_best_hyperparameters(X_train, y_train):
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

    tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

    # Get the optimal hyperparameters
    return tuner.get_best_hyperparameters(num_trials=1)[0]


def create_model(best_hps):
    model = Sequential()

    model.add(Dense(best_hps.get("units"), activation=best_hps.get("dense_activation")))
    model.add(
        Dense(best_hps.get("units"), activation=best_hps.get("dense_activation"))
    )  # additional hidden layer
    model.add(
        Dense(best_hps.get("units"), activation=best_hps.get("dense_activation"))
    )  # additional hidden layer
    model.add(Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=best_hps.get("learning_rate")),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def handle_missing_values(df, state_store):
    df = df.drop(
        columns="Cabin"
    )  # Since the majority of rows in the set are missing Cabin we can ignore

    # Fill missing 'Age' values with median age (imputation) since the value is continuous; over nearly 1/5 of the rows are missing Age
    median_age = state_store.get("median_age")
    if not median_age:
        median_age = df["Age"].median()
        state_store.set("median_age", median_age)
    df["Age"] = df["Age"].fillna(median_age)

    # Fill missing 'Embarked' values with the mode
    mode_embarked = state_store.get("mode_embarked")
    if not mode_embarked:
        mode_embarked = df["Embarked"].mode()[0]
        state_store.set("mode_embarked", mode_embarked)
    df["Embarked"] = df["Embarked"].fillna(mode_embarked)

    # Fill missing 'Fare' values with the median
    median_fare = state_store.get("median_fare")
    if not median_fare:
        median_fare = df["Fare"].median()
        state_store.set("median_fare", median_fare)
    df["Fare"] = df["Fare"].fillna(median_fare)

    return df


def identify_missing_values(df):
    print(df.isnull().sum())


def encode_categorical_variables(df, state_store):
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=["Sex", "Embarked", "Pclass", "Title"])

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
    return df


def preprocess_data(df, state_store):
    df = handle_missing_values(df, state_store)
    df = add_title(df)
    df = encode_categorical_variables(df, state_store)
    df = drop_useless_columns(df)

    return df


if __name__ == "__main__":
    state_store = StateStore()

    df_train = load_data()

    # df_train = handle_missing_values(df_train, state_store)
    # df_train = add_title(df_train)
    # df_train = encode_categorical_variables(df_train, state_store)
    # df_train = drop_useless_columns(df_train)
    df_train = preprocess_data(df_train, state_store)

    # We save these as our test set will need to have the same columns
    state_store.set("column_names", df_train.columns.tolist())

    print(f"State: {state_store.get()}")

    # identify_missing_values(df_train)  # Should be empty

    # df_train.head()

    # number of folds
    n_splits = 5

    # KFold object
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # extract your features and labels from the dataframe
    X = df_train.drop("Survived", axis=1)
    y = df_train["Survived"]

    # print number of columns in df_train
    print(f"Number of columns in df_train: {len(df_train.columns)}")

    best_hps = None
    best_score = 0

    # iterate over each fold
    for i, (train_index, val_index) in enumerate(kfold.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Normalize the data into all floats
        X_train = X_train.astype("float32")
        y_train = y_train.astype("float32")
        X_val = X_val.astype("float32")
        y_val = y_val.astype("float32")

        best_hps = get_best_hyperparameters(X_train, y_train)

        # create a new model
        model = create_model(best_hps)
        model.fit(
            X_train,
            y_train,
            epochs=100,
            # verbose=0,
        )

        # evaluate the model
        scores = model.evaluate(
            X_val,
            y_val,
            # verbose=0,
        )

        # save the best model
        if scores[1] > best_score:
            best_score = scores[1]
            best_hps = best_hps

    # Now that we have the best hyperparameters, retrain on the whole dataset
    final_model = create_model(best_hps)
    # Normalize the whole dataset
    X_train = X.astype("float32")
    y_train = y.astype("float32")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # fit the final model
    final_model.fit(
        X_train,
        y_train,
        epochs=100,
        # test_size=0.2,
        # random_state=42,
    )

    print(f"Number of features (columns) in X: {X.shape[1]}")

    # Save the best model
    final_model.save(
        f"../models/{datetime.now().strftime('%Y%m%d%H%M%S')}_titanic_model.keras"
    )
