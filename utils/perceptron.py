import numpy as np


class Perceptron:
    def __init__(self, value=0,weight=0):
        self.value = value
        self.weight = weight

    def update_weight(self,value):
        self.weight = value
    def output_data(self):
        return self.value*self.weight
    def update(self, value):
        self.value = value

    def get_data(self, values):
        self.update(np.sum(values))

    def get_weight(self):
        return self.weight