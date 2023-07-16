import pickle


class StateStore:
    def __init__(self, filename="state.pickle", load=False):
        self.filename = filename
        if load:
            self.state_store = {}
        else:
            with open(filename, "rb") as f:
                self.state_store = pickle.load(f)

        with open(filename, "wb") as f:
            pickle.dump(self.state_store, f)

    def set(self, name, value):
        with open(self.filename, "rb") as f:
            self.state_store = pickle.load(f)
        self.state_store[name] = value
        with open(self.filename, "wb") as f:
            pickle.dump(self.state_store, f)

    def get(self, name=None):
        with open(self.filename, "rb") as f:
            self.state_store = pickle.load(f)

        if name:
            return self.state_store[name]
        else:
            return self.state_store
