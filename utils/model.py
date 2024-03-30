import matplotlib.pyplot as plt

from utiloptimizers import adam
from linear import Linear
import numpy as np
from utilLossFunction import utilLossFunction
from utilactivation import utilactivation
class nn:
    def __init__(self,layers,optimizer=adam):
        self.layers = layers
        self.optimizer = optimizer()
        self.t = 1
    def forward(self,x):
        if len(self.layers)== 1:
            x = self.layers[0].get_values(x)
            return x
        x = self.layers[0].forward()

        for i in range(1,len(self.layers)-1):
            self.layers[i].update_values(x)
            x = self.layers[i].forward()

        return self.layers[len(self.layers)-1].get_values()

    def backwards(self):
        for layer in range(len(self.layers)-1):
            weights = np.array(self.layers[layer].get_weights())
            bias = np.array(self.layers[layer].get_values())
            self.optimizer =adam()
            w,b = self.optimizer.update(t=self.t, w=weights, b=bias,dw=np.gradient(weights),db=np.gradient(bias))
            self.layers[layer].update_weights(w)
            #layer.update_values(b)

        self.t += 1
    def print_model(self):
        for i in self.layers:
            print(i.get_values())
inputdata = np.random.rand(10)
needed = [10,10,10,10,10,10,10,10,10,10]
linear1 = Linear(perceptron_number=10,input_data=inputdata,input_layer=True)
linear2 = Linear(perceptron_number=40)
Linear4 = Linear(perceptron_number=40,activationfunc=utilactivation.sigmoid)
linear3 = Linear(perceptron_number=1)
optimizer = adam(eta=0.001)
model = nn([linear1,linear2,Linear4,linear3],optimizer=adam)
nPerClust = 100
blur = 1

A = [ 1, 1 ]
B = [ 5, 1 ]

# generate data
a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur ]
b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur ]

# true labels
labels_np = np.vstack((np.zeros((nPerClust,1)),np.ones((nPerClust,1))))

# concatanate into a matrix
data_np = np.hstack((a,b)).T

# convert to a pytorch tensor
data = np.array(data_np,dtype=float)
labels = np.array(labels_np,dtype=float)

# show the data
fig = plt.figure(figsize=(5,5))
plt.plot(data[np.where(labels==0)[0],0],data[np.where(labels==0)[0],1],'bs')
plt.plot(data[np.where(labels==1)[0],0],data[np.where(labels==1)[0],1],'ko')
plt.title('The qwerties!')
plt.xlabel('qwerty dimension 1')
plt.ylabel('qwerty dimension 2')
plt.show()
epochs = 1000
losses = []
for i in range(epochs):
    outputs = model.forward(data)
    loss = utilLossFunction.cross_entropy(y_pred=outputs,y_true=needed)
    losses.append(loss)
    model.backwards()
print(losses)
