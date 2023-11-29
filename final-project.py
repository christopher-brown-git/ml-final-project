import pandas as pd
import numpy as np
import os
from data_loader import load_data


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
                


def main():
    #check file exists
    if not os.path.isfile("df_small_1.csv"):
        load_data()

if __name__ == "__main__":
    main()