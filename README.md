# Assesment
Programming Test: Learning Activations on Neural Networks

The addition of layers called the hidden layers, between the input and the output layer to overcome the problem of a perceptron model that couldn’t work well with
the non-linear separable data gave rise to the multilayer perceptron. The model defines consists of one input layer, one hidden layer and one output layer. 
The dataset Bank Notes dataset consists of four features:- Variance, Skewness, Kurtosis and Entropy. It’s a binary classification problem consisting of values 0 and 1. Each row of the dataset is fed as an input to the input layer. So the input layer consists of four neurons, one for each feature. Once the layer receives the input, the neurons have to be activated. Each layer is associated with a set of weights and bias. Weights are like the coefficients which determine the importance of each feature in the dataset and bias is used to shift the solution space from the origin. If no bias, the solution is not linearly separable. 

The net input is calculated using the formula: Zi = xiwi + b where xi is the weights, wi is the weights and b is the bias. We activate the neuron with the help of the activation function. Activation functions are used to control the output of each layer. That is, activation functions are like the threshold value of the neuron. The activation function used here is the non-linear probabilistic-based function called the softmax. Softmax gives the probability of each data point belonging to each class. The function is given by: F(Zi)=  e^Zi/(∑e^Zk ) . Once the neuron is activated it is sent as an input to the next layer. We need to keep in mind that all the neurons are activated simultaneously and produce the output.  This forward movement of the outputs from each layer to the next layer as an input up-to the output layer is the forward propagation. We then calculate the loss which is given by the difference between the actual and the predicted values in the output layer.

If the algorithm only computed the weighted sum of the inputs, propagated results to the output layer, and stopped there, it will not be able to learn the model correctly. In the sense, it will not be able to assign optimal weights to the inputs.No actual learning takes place. Optimal weights are required to minimize the cost function. The main goal of any neural network model is to reduce the cost function. Therefore we perform backpropagation.

In backpropagation we iteratively adjust the weights in the network with the goal of minimizing the cost function (loss).We try to optimize the weights with respect to the loss function. The optimization algorithm used is gradient descent optimization. Here, we try to find the local minima that reduce the cost function with the learning rate. The learning rate is defined as by what percent the weight should be changed to reduce the error. The learning rate (alpha) chosen here is 0.05. Lower the learning rate, then the gradient will converge very slowly, and higher the learning rate, the gradient may oscillate and might not reach the local minima. Thus, a minimal value should be chosen as the learning rate. The error signal propagates from the output layer to the hidden layers and at each neuron updating its weights and biases. 
Wi(new) = Wi(old)  - alpha(loss)

The model is trained with the train set for a fixed number of epochs, learning rate. The network is trained for a fixed number of epochs and within each epoch updating the weights and bias for the inputs. The loss is calculated as the sum of the differences between the actual and the predicted values. The model trained for 10 epochs obtained a loss of 1098.
-->Epoch: 0 -->Alpha: 0.05 -->Error Sum: [[1098.]]
-->Epoch: 1 -->Alpha: 0.05 -->Error Sum: [[1098.]]
-->Epoch: 2 -->Alpha: 0.05 -->Error Sum: [[1098.]]
-->Epoch: 3 -->Alpha: 0.05 -->Error Sum: [[1098.]]
-->Epoch: 4 -->Alpha: 0.05 -->Error Sum: [[1098.]]
-->Epoch: 5 -->Alpha: 0.05 -->Error Sum: [[1098.]]
-->Epoch: 6 -->Alpha: 0.05 -->Error Sum: [[1098.]]
-->Epoch: 7 -->Alpha: 0.05 -->Error Sum: [[1098.]]
-->Epoch: 8 -->Alpha: 0.05 -->Error Sum: [[1098.]]
-->Epoch: 9 -->Alpha: 0.05 -->Error Sum: [[1098.]]
Hidden Layer Weights:
[{'weights': [array([[0.89325224]]), array([[0.00711141]]), array([[0.57705045]]), array([[0.67724837]]), array([[0.72156333]])], 'output': array([[1.]]), 'delta': array([[0.]])}, 

{'weights': [array([[0.12168445]]), array([[0.35442703]]), array([[0.96085067]]), array([[0.35528546]]), array([[0.67407736]])], 'output': array([[1.]]), 'delta': array([[0.]])},

 {'weights': [array([[0.88702744]]), array([[0.53873579]]), array([[0.06564393]]), array([[0.46829042]]), array([[0.84875228]])], 'output': array([[1.]]), 'delta': array([[0.]])}]

Output Layer:

 [{'weights': [array([[0.01433756]]), array([[0.22923161]]), array([[0.20989991]]), array([[0.24969275]])], 'output': array([[1.]]), 'delta': array([[0.]])}, 
{'weights': [array([[0.3225386]]), array([[0.67651523]]), array([[0.5306517]]), array([[0.89082956]])], 'output': array([[1.]]), 'delta': array([[0.]])}]

The model is then tested using the test set with using the weights of the trained model. Out of the 274 values used for testing, 157 values were predicted correctly. 



