import pandas as pd
import numpy as np
import os
from data_loader import load_data

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
    return np.sum(Y==Yhat)/len(Y)

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
        grad_vec = np.zeros(d)

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
    
    def fit(self, Xmat, Y, max_iterations=1000, tolerance=1e-6, verbose=False):
        """
        Train a logistic regression model using the training data in Xmat and Y
        """

        num_rows, num_cols = Xmat.shape

        #initialilze theta and theta_new randomly
        theta = np.random.uniform(-1, 1, d)
        theta_new = np.random.uniform(-1, 1, d)

        #do gradient descent until convergence
        iteration = 0
        while iteration < max_iterations: 
            if verbose:
                print("Iteration", iteration, "theta=", theta)

            theta_new = theta - self.learning_rate * self.calculate_gradient(Xmat, Y, theta)

            if np.mean(np.abs(theta_new-theta)) < tolerance:
                break
            theta = theta_new

            iteration += 1
        
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



def main():
    #check file exists
    path = "df_small_1.csv"
    if not os.path.isfile(path):
        load_data()
    
    df = pd.read_csv(path)
    
    
    

if __name__ == "__main__":
    main()