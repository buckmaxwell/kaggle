# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# 1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
import os
import sys
import pandas as pd

# append the path three directories up to import the variable store
sys.path.append("/Users/maxbuck/Documents/kagcomp")

from shared.state_store import StateStore


def load_data():
    df = pd.read_csv("../data/train.csv")
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


def encode_categorical_variables(df):
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=["Sex", "Embarked"])

    # We save these as our test set will need to have the same columns
    state_store.set("column_names", df.columns.tolist())
    return df


if __name__ == "__main__":
    state_store = StateStore()

    df = load_data()
    df = handle_missing_values(df, state_store)
    df = encode_categorical_variables(df)
    identify_missing_values(df)  # Should be empty

    # To retrieve the column names later, you can do:
    column_names = state_store.get("column_names")
    print(column_names)
