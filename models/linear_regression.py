import numpy as np

class LinearRegression():

    def __init__(self, input_dim, learning_rate=0.01, n_iterations=1000):
        """
        Linear Regression using Gradient Descent
        - input_dim: number of features of X
        - learning_rate: Learning rate for gradient descent
        - n_iterations: Number of iterations for gradient descent
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.W = np.random.randn(1, input_dim) * 0.01
        self.b = np.zeros((1,1))

    def forward(self, X):
        """
        Parameters:
            X (np.ndarray): Input data of shape (input_dim, m).
        """
        #  (1, m) = (1, input_dim).(input_dim, m) + (1, 1) broadcasted
        return np.dot(self.W, X) + self.b

    def compute_cost(self, Y_preds, Y):
        """
        Computes the Mean Squared Error loss
        Parameters:
            Y_preds (np.ndarray): Predicted output of shape (1, m).
            Y (np.ndarray): True output of shape (1, m).
        Returns:
            float: The scalar cost value.
        """
        m = Y.shape[1] # Number of samples
        cost = (1 / (2 * m)) * np.sum(np.square(Y_preds - Y))
        return cost

    def backward(self, X, Y_preds, Y):
        """
        Computes the partial derivatives (gradients) of the loss w.r.t. W and b
        Parameters:
            X (np.ndarray): Input data of shape (input_dim, m).
            Y_preds (np.ndarray): Predicted output of shape (1, m).
            Y (np.ndarray): True output of shape (1, m).
        """
        m = X.shape[1] # Number of samples

        grads = {}
        # dW = (1/m) * dZ . X.T (Shape: (1, m) . (m, input_dim) = (1, input_dim))
        grads["dW"] = 1/m * np.sum(np.dot((Y_preds - Y), X.T))
        grads["db"] = 1/m * np.sum((Y_preds - Y))

        return grads


    def update_params(self, grads):
        # Ensure db is treated as a scalar for subtraction if needed, although broadcasting usually handles it.
        self.W -= self.learning_rate * grads["dW"]
        self.b -= self.learning_rate * grads["db"]


    def fit(self, X, Y):
        """
        Trains the linear regression model.
        Parameters:
            X (np.ndarray): Input data of shape (input_dim, m).
            Y (np.ndarray): True output of shape (1, m).
        Returns:
            list: A list of cost values computed during training.
        """
        costs = []
        m = X.shape[1]

        for i in range(self.n_iterations):
            Y_preds = self.forward(X)
            cost = self.compute_cost(Y_preds, Y)
            grads = self.backward(X, Y_preds, Y)

            self.update_params(grads)

            costs.append(cost)

        return costs

    def predict(self, X):
        """
        Makes predictions using the trained model.
        Parameters:
            X (np.ndarray): Input data of shape (input_dim, m).
        Returns:
            np.ndarray: Predicted output of shape (1, m).
        """
        return self.forward(X)