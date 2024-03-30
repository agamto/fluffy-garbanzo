from perceptron import Perceptron
import numpy as np
from utilactivation import utilactivation

class Linear:
    def __init__(self, perceptron_number=10,activationfunc=utilactivation.relu, input_data=None, input_layer=False):
        weights = np.random.rand(perceptron_number)
        self.layer = []
        self.activationfunc = activationfunc
        for i in range(perceptron_number):
            if input_layer:
                self.layer.append(Perceptron(weight=weights[i], value=input_data[i]))
            else:
                self.layer.append(Perceptron(weight=weights[i],value=weights[i]))

    def forward(self):
        output = 0
        for i in self.layer:
            output += i.output_data()
        self.activationfunc(output)
        return output
    def get_values(self):
        values = []
        for i in self.layer:
            values.append(i.output_data())
        return values

    def update_values(self, x):
        for i in range(len(self.layer)):
            self.layer[i].update(x)

    def update_weights(self,x):
        for i in range(len(self.layer)):
            self.layer[i].update_weight(x[i])

    def get_weights(self):
        x  = []
        for i in range(len(self.layer)):
            x.append(self.layer[i].get_weight())
        return x