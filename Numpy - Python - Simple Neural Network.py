# full batch gradient descent (i.e. not SGD)

# https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1

import numpy as np
from numpy import exp, array, random, dot
training_set_inputs = array([[0, 1, 2], [0, 0, 2], [1, 1, 1], [1, 0, 1]])
training_set_outputs = array([[1, 0, 1, 0]]).T
random.seed(1)

#Initialization
W = random.random((3, 1))
B = random.random((1, 1))

for iteration in range(10000):
    # Sigmoid function, P. 40
    yHat = 1 / (1 + exp(-(dot(training_set_inputs, W)+B)))
    # gradient of mean square loss: grad0 = (yHat-training_set_outputs) (P. 79)
    # gradient of Sigmoid: grad = grad0 * yHat * (1 - yHat) (P. 131);
    # full batch gradient descent
    grad=(yHat-training_set_outputs) * yHat * (1 - yHat)
    # gradient of linear layer (P. 135)
    d_W=dot(training_set_inputs.T, grad)
    # just sum up grad to form d_B (P. 137)
    d_B=np.sum(grad,axis=0)
    LearnRate=0.5
    # gradient descent method (P. 95)
    W -= LearnRate*d_W
    B -= LearnRate*d_B
        
print(1 / (1 + exp(-(dot(array([0, 1, 0]), W)+B))))


