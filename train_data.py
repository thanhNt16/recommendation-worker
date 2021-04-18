class Data():
    def __init__(self):
        self.data = dict()

    def get_data(self, key):
        return self.data[key]
    def set_data(self, key, value):
        self.data[key] = value

train_data = Data()