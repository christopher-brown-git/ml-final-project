import pandas as pd
import numpy as np
import os
import tqdm
from data_loader import create_data, load_data
import random

iterations = 1000
rows_of_data = 100000

def mean_negative_loglikelihood(Y, pYhat):
    """
    Function to compute the mean negative loglikelihood

    Y is a vector of the true labels. This is a classification problem so Y will contain 0 or 1's
    pYhat is a vector containging the estimated probabilities; each entry i is p(Y_i=1 | ... )
    """
    n = len(Y)
    return -np.mean((Y * np.log(pYhat)) + (np.ones(n) - Y) * np.log(np.ones(n) - pYhat))

def accuracy(Y, Yhat):
    """
    Function for computing accuracy

    Y: vector of true labels
    Yhat: vector of estimated 0/1 probabilities
    """
    #return np.sum(Y==Yhat)/len(Y)
    acc = 0
    for y, yhat in zip(Y, Yhat):
        if y == yhat: acc += 1
    
    return acc/len(Y) * 100

def sigmoid(V):
    """
    Activation function
    Input: vector of floats
    Output: vector of probabilities (values between 0 and 1)
    """
    return 1/(1+np.exp(-V))

class LogisticRegression:

    def __init__(self, learning_rate=0.1, lamda=None):
        """
        Constructor for Logistic Regression class. The learning rate is 
        a positive integer that controls the step size in gradient descent.

        Lamda is a positive integer controlling how strong regularization is. 
        """
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.theta = None
    
    def calculate_gradient(self, Xmat, Y, theta_p, tiny=1e-5):
        """
        Helper function to compute the gradient vector at a point theta_p
        """
        num_rows, num_cols = Xmat.shape

        #create gradient vector
        grad_vec = np.zeros(num_cols)

        for i in range(0, len(grad_vec)):
            nudge = [0]*len(grad_vec)
            nudge[i] = tiny

            if self.lamda == None:
                grad_vec[i] = (mean_negative_loglikelihood(Y, sigmoid(Xmat @ (theta_p + nudge))) - 
                               mean_negative_loglikelihood(Y, sigmoid(Xmat @ theta_p)))/tiny
            else:
                l2_reg_no_nudge = self.lamda * np.sum((theta_p + nudge)**2)
                l2_reg_nudge = self.lamda * np.sum(theta_p**2)

                grad_vec[i] = ((l2_reg_no_nudge + mean_negative_loglikelihood(Y, sigmoid(Xmat @ (theta_p + nudge)))) 
                                - (l2_reg_nudge + mean_negative_loglikelihood(Y, sigmoid(Xmat @ theta_p))))/tiny

        return grad_vec
    
    def fit(self, Xmat, Y, max_iterations=iterations, tolerance=1e-6, verbose=False):
        """
        Train a logistic regression model using the training data in Xmat and Y
        """

        num_rows, num_cols = Xmat.shape

        #initialilze theta and theta_new randomly
        theta = np.random.uniform(-1, 1, num_cols)
        theta_new = np.random.uniform(-1, 1, num_cols)

        #do gradient descent until convergence
        for iteration in tqdm.tqdm(range(0, max_iterations), total=max_iterations):
            if verbose:
                print("Iteration", iteration, "theta=", theta)

            theta_new = theta - self.learning_rate * self.calculate_gradient(Xmat, Y, theta)

            if np.mean(np.abs(theta_new-theta)) < tolerance:
                break
            theta = theta_new

            #iteration += 1
                
        self.theta = theta_new.copy()

    def predict(self, Xmat):
        """"
        Predict 0/1 for each row in the data matrix Xmat using the following rule:
        pYhat = 1 if p(Y=1 | X) > 0.5 else 0
        """

        num_rows, num_cols = Xmat.shape
        predictions = sigmoid(Xmat @ self.theta)

        ret = [0]*num_rows

        for i in range(num_rows):
            if predictions[i] > 0.5:
                ret[i] = 1
            else:
                ret[i] = 0
        
        return ret


class Value:
    def __init__(self, data=None, label="", _parents=set(), _operation=""):
        """
        Constructor for a value node
        Value node contains data/the value, gradient, the parents node, 
        and the operation applied to the parents node to produce the value
        """

        #initialize data randomly if none was provided
        self.data = random.uniform(-1, 1) if data is None else data

        self.grad = 0
        self.label = label

        #initialize the operation used to create the value/data
        #initialize what the backward pass does
        self._parents = set(_parents)
        self._operation = _operation
        self._backward = lambda: None
    
    def __repre__(self):
        """
        Helper class for printing the value
        """
        return f"Value(data={self.data}, label={self.label})"
    
    def __add__(self, other):
        """
        Implementing the plus operator
        """

        #ensure other is always a Value object
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _parents=(self, other), _operation='+')

        #assign how gradients are backpropagated

        #for addition, you get the constant 1 because
        #for 2 nodes x_1 and x_2 each with child x_3 and loss function L
        #let x_1 be self
        #dL/dx_1 = dL/dx_3 * dx_3/dx_1 and dx_3/dx_1 = d/dL (x_1 + x_2) = 1 + 0 = 1

        def _backward():
            self.grad += 1*out.grad
            other.grad += 1*out.grad
        
        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Implement multiplication
        """

        #ensures other is always a Value node
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _parents=(self, other), _operation='*')

        #how gradients get assigned
        #let x_3 be the child node of x_1 and x_2 and L the loss
        #let x_1 be self
        # dL/dx_1 = dL/dx_3 * dx_3/dx_1 = out.grad * d/dx_1 (x_1 * x_2) = out.grad * other.data
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        """
        Implement power operator **
        """

        #ensure other is always a Value object
        assert isinstance(other, (int, float)), "only support int/float powers now"
        out = Value(self.data ** other, _parents=(self,), _operation="**" + str(other))

        #dL/dx_1 = dL/dx_2 * dx_2/dx_1
        def _backward():
            self.grad += (other * (self.data**(other-1))) * out.grad        
        
        out._backward = _backward

        return out

    def relu(self):
        """
        Implementation of the ReLU activation function
        """

        #ensure we always have a Value object
        out = Value(0 if self.data < 0 else self.data, _parents=(self,), _operation=("ReLU"))

        #if x_1 is the parent of x_2 and L is the loss
        # dL/dx_1 = dL/dx_2 * dx_2/dx_1 = out.grad * dx_2/dx_1
        def _backward():
            self.grad += (out.data > 0) * out.grad
        
        out._backward = _backward
        return out
    
    def exp(self):
        """
        Implement exponent (e^x where x is the data in a node)
        """

        #ensures we always have a Value object
        out = Value(np.exp(self.data), _parents=(self,), _operation='e')
        
        #if x_1 is the parent of x_2 and L is the loss,
        #dL/dx_1 = dL/dx_2 * dx_2/dx_1 = out.grad * e^(x_1) = out.grad * out.data
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out
    
    def log(self):
        """
        Implement log function
        """

        #ensures always have a Value node
        out = Value(np.log(self.data), _parents=(self,), _operation='log')

        #if x_1 is the parent of x_2 and L is the loss,
        #dL/dx_1 = dL/dx_2 * dx_2/dx_1 = out.grad * 1/x_1 = out.grad * 1/self.data
       
        def _backward():
            self.grad += 1/self.data * out.grad
        
        out._backward = _backward

        return out
    
    def backward(self):
        """
        Call _backward() method for each node in the neural network
        Does so in reverse toplogical order, i.e. backpropagation
        """

        topo_ord = []
        visited = set()

        def build_topo_ord(node):
            if node not in visited:
                visited.add(node)
                for parent in node._parents:
                    build_topo_ord(parent)
                topo_ord.append(node)

        build_topo_ord(self)

        self.grad = 1.0
        for node in reversed(topo_ord):
            node._backward()
    
    def __float__(self): return float(self.data)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

def sigmoid(value, scale=0.5):
    """
    sigmoid activation function
    """
    val1 = Value(-scale)
    return 1/(1 + (val1*value).exp())

def negative_log_likelihood(y, pY1):
    """
    Negative loglikelihood for a single example of data, i.e., 
    Y and p(Y=1 | ...)
    """
    val1 = Value(1)
    return -(y * pY1.log() + (val1 - y) * (val1 - pY1).log())

class Neuron: 
    """
    Represents a single neuron in a neural network
    A neuron computs a linear combination of its inputs and an intercept
    and then passes it through a non-linear activation function
    """

    def __init__(self, n_inputs):
        """
        Constructor: initializes a parameter for each input, in
        addition to one for an intercept
        """
        self.theta = [Value(random.uniform(-1, 1) for _ in range(n_inputs))]
        self.intercept = Value(random.uniform(-1, 1))
    
    def __call__(self, x, relu=False, dropout_proba=0.1, train_mode=False):
        """
        Implementing call operator
        """
        if train_mode:
            d = [0]*len(self.theta)
            for i in range(len(d)):
                if random.uniform(0, 1) > dropout_proba:
                    d[i] = 1
            
            out = sum([self.theta[i]*x[i]*d[i] for i in range(len(self.theta))]) + self.intercept
        else:
            out = sum([self.theta[i]*x[i]*(1 - dropout_proba) for i in range(len(self.theta))]) + self.intercept

        if relu:
            return out.relu()
        return sigmoid(out)
    
    def parameters(self):
        """
        Returns a list of all parameters in the neuron
        """
        return self.theta + [self.intercept]

class Layer:
    """
    Class for implementing 1 layer of a neural network
    """

    def __init__(self, n_inputs, n_outputs):
        """
        Constructor to initialize the layer with neurons
        """
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]
    
    def __call__(self, x, relu=True, dropout_proba=0.1, train_mode=False):
        """
        Implementing the function call operator, ()
        """

        outputs = [n(x, relu, dropout_proba, train_mode=train_mode) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
    
    def parameters(self):
        """
        Method to return a list of every parameter in the layer
        """
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    """
    This class implements a multilayer perceptron
    """

    def __init__(self, n_features, layer_sizes, learning_rate=0.01, dropout_proba=0.1):
        """
        Constructor initializing layers of appropriate width
        """

        layer_sizes = [n_features] + layer_sizes
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        self.dropout_proba = dropout_proba
        self.learning_rate = learning_rate

        self.train_mode = True #determines when we are training vs testing--when to use dropout or not
    
    def __call__(self, x):
        """
        call, (), operator that calls each layer in the neural network, 
        using the outputs of a layer as the inputs to the next layer
        """

        out = x
        for layer in self.layers[0:len(self.layers)-1]:
            out = layer(out, relu=True, dropout_proba=self.dropout_proba, train_mode=self.train_mode)
        return self.layers[-1](out, relu=False)
    
    def parameters(self):
        """
        Returns a list of parameters of the neural network
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def _zero_grad(self):
        """
        This method sets the gradients of every parameter to 0
        """
        for p in self.parameters():
            p.grad = 0
    
    def fit(self, Xmat_train, Y_train, Xmat_val=None, Y_val=None, max_epochs=100, verbose=False):
        """
        Fit parameters of the neural network to the given data using SGD (stochastic gradient descent)
        SGD ends after reaching the max # of epochs

        Can take in validation inputs to test the generalization error
        """

        #initialize parameter randomly
        for p in self.parameters():
            p.data = random.uniform(-1, 1)
        
        #for iteration in tqdm.tqdm(range(0, max_iterations), total=max_iterations):

        #iterate over epochs
        for e in tqdm.tqdm(range(0, max_epochs), total=max_epochs):
            n, d = Xmat_train.shape

            shuffled_samples = np.arange(0, n, 1).tolist()

            np.random.shuffle(shuffled_samples)

            for i in shuffled_samples:
                py_i = self(Xmat_train[i])
                loss = negative_log_likelihood(Y_train[i], py_i)

                self._zero_grad() #zero gradient 
                loss.backward() #calculate gradients

                for param in self.parameters():
                    param.data = param.data - self.learning_rate * param.grad
                
            if verbose:
                self.train_mode = False

                train_accuracy = accuracy(Y_train, self.predict(Xmat_train))

                if Xmat_val is not None:
                    val_accuracy = accuracy(Y_val, self.predict(Xmat_val))
                    print(f"Epoch {e}: Training accuracy {train_accuracy:.0f}%, Validation accuracy {val_accuracy:.0f}%")
                else:
                    print(f"Epoch {e}: Training accuracy {train_accuracy:.0f}%")
            
            self.train_mode = True
        
        self.train_mode = False

    def predict(self, Xmat):
        """
        Predict method which returns a list of 0/1 labels for the given input data
        """

        return [int(self(x).data > 0.5) for x in Xmat]

def logistic_regression(feature_names, data):
    
    model_basic = LogisticRegression(learning_rate=0.2, lamda=0.0)
    model_basic.fit(data["Xmat_train"], data["Y_train"])
    
    Yhat_train_basic = model_basic.predict(data["Xmat_train"])

    # model_lowl2 = LogisticRegression(learning_rate=0.2, lamda=0.01)
    # model_lowl2.fit(data["Xmat_train"], data["Y_train"])

    # model_highl2 = LogisticRegression(learning_rate=0.2, lamda=0.2)
    # model_highl2.fit(data["Xmat_train"], data["Y_train"])
    

    # Yhat_train_lowl2 = model_lowl2.predict(data["Xmat_train"])
    # Yhat_train_highl2 = model_highl2.predict(data["Xmat_train"])

    # accuracy_basic_train = accuracy(data["Y_train"], Yhat_train_basic)
    # accuracy_lowl2_train = accuracy(data["Y_train"], Yhat_train_lowl2)
    # accuracy_highl2_train = accuracy(data["Y_train"], Yhat_train_highl2)

    # #validation data
    # Yhat_val_basic = model_basic.predict(data["Xmat_val"])
    # Yhat_val_lowl2 = model_lowl2.predict(data["Xmat_val"])
    # Yhat_val_highl2 = model_highl2.predict(data["Xmat_val"])

    # accuracy_basic_val = accuracy(data["Y_val"], Yhat_val_basic)
    # accuracy_lowl2_val = accuracy(data["Y_val"], Yhat_val_lowl2)
    # accuracy_highl2_val = accuracy(data["Y_val"], Yhat_val_highl2)

    # f = open("out.b", "a")
    # f.write("\n**" + str(iterations) + " iterations of GD on " + str(rows_of_data//1000) + "k rows of data**")

    # f.write("\nTraining accuracy \nno reg: " + str(accuracy_basic_train))
    # f.write("\nTraining accuracy \nlamda=0.01: " + str(accuracy_lowl2_train))
    # f.write("\nTraining accuracy \nlamda=0.2: " + str(accuracy_highl2_train) + "\n")

    # f.write("\nValidation accuracy \nno reg: " + str(accuracy_basic_val))
    # f.write("\nValidation accuracy \nlamda=0.01: " + str(accuracy_lowl2_val))
    # f.write("\nValidation accuracy \nlamda=0.2: " + str(accuracy_highl2_val) + "\n\n")
    # f.close()


    #choose the best model
    best_model = model_basic # EDIT ME
    Yhat_test = best_model.predict(data["Xmat_test"])
    
    f = open("out.b", "a")
    f.write("\nTest accuracy no reg: " + str(accuracy(data["Y_test"], Yhat_test)))
    f.write("Clash Royale data weights: " + str({feature_names[i]: round(best_model.theta[i], 2) for i in range(len(feature_names))}))
    f.close() 

def neural_network(feature_names, data):
    
    n, d = data["Xmat_train"].shape

    #neural net with no dropout
    random.seed(42)
    arc = [4, 5, 2, 7, 3, 2, 1]
    mlp_1 = MLP(n_features=d, layer_sizes=arc, learning_rate=0.05, dropout_proba=0.0)
    mlp_1.fit(data["Xmat_train"], data["Y_train"], data["Xmat_val"], data["Y_val"], verbose=False, max_epochs=50)
    train_acc_1 = accuracy(data["Y_train"], mlp_1.predict(data["Xmat_train"]))
    val_acc_1 = accuracy(data["Y_val"], mlp_1.predict(data["Xmat_val"]))

    f = open("net.txt", "a")
    f.write(f"Final training accuracy: {train_acc_1:.0f}%, Validation accuracy: {val_acc_1:.0f}%")
    f.write("\n")

    random.seed(0)
    print("Training neural net with dropout=0.5")
    mlp_2 = MLP(n_features=d, layer_sizes=arc, learning_rate=0.05, dropout_proba=0.5)
    mlp_2.fit(data["Xmat_train"], data["Y_train"], data["Xmat_val"], data["Y_val"], verbose=False, max_epochs=50)
    train_acc_2 = accuracy(data["Y_train"], mlp_2.predict(data["Xmat_train"]))
    val_acc_2 = accuracy(data["Y_val"], mlp_2.predict(data["Xmat_val"]))
    f.write(f"Final training accuracy: {train_acc_2:.0f}%, Validation accuracy: {val_acc_2:.0f}%")
    f.close()


def main(feature_names, data):

    #logistic_regression(feature_names, data)
    neural_network(feature_names, data)

if __name__ == "__main__":
    #check file exists
    data_path = "/home/scratch/24cjb4/df_small_" + str(rows_of_data//1000) + "k.csv"

    if not os.path.isfile(data_path):
        create_data_simple(rows_of_data, data_path)
    

    feature_names, data = load_data_simple(data_path)

    main(feature_names, data)