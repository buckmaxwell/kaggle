from kerastuner import HyperModel
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# First, we define a HyperModel class
class BasicHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        model.add(
            Dense(
                units=hp.Int("units", 8, 64, step=8),
                activation=hp.Choice(
                    "dense_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu",
                ),
            )
        )
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model
