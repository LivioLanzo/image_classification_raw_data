
import numpy as np
from scipy import optimize as op

class neural_network(object):

    def __init__(self, hidden_layers: tuple, regularization_parameter=0.0, max_iter=200):
        '''
            PARAMETERS
            ----------
            hidden_layers: a tuple containing the layers sizes\n
            the length of the tuple is the number of hidden layers
            and each element is the size of the ith hidden layer\n
                Example:
                (10,20,30): 3 hidden layers of sizes:
                            10 (1st hidden layer)
                            20 (2nd hidden layer)
                            30 (3rd hidden layer)\n

            regularization_parameter: the regularization parameter, if no regularization then = 0\n
            max_iter: Maximum number of function evaluation. if None, maxiter is set to max(100, 10*len(x0)). Defaults to None
        '''
        self._hidden_layers = hidden_layers
        self._regularization_parameter = regularization_parameter
        self._max_iter = max_iter

    def _logistic_function(self, H: np.ndarray):
        '''
            DESCRIPTION
            -----------
            Calculate the logistic function over each element of H

            PARAMETERS
            ---------
            H: a numpy ndarray
        '''
        return 1 / (1 + np.exp(-H))

    def _make_thetas(self, layers: tuple):
        '''
            DESCRIPTION
            -----------
            Create n-1 thetas matrixes, where n is the number of layers, len(layers)\n
            Each theta matrix has a size of:\n
                rows = (number of elements + 1) of the ith layer
                columns = the number of elements of the (ith + 1) layer\n

                Each theta matrix is initialized by np.random.randn 

            PARAMETERS
            ----------
            layers: a tuple where each element represents the size of the ith layer.\n
                    it includes the input, output layers


            RETURNS
            ---------
            None
        '''
        self.thetas = []
        self._thetas_sizes = []

        for i in range(len(layers))[:-1]:
            rows = layers[i]
            cols = layers[i+1]
            self.thetas.extend(np.random.randn(rows+1, cols).flatten(order='C'))
            self._thetas_sizes.append((rows+1, cols))

    def _reshape_thetas(self, flattened_thetas):
        # reshape thetas based on self._thetas_sizes
        reshaped_thetas = []
        elements_count = 0
        for theta_size in self._thetas_sizes:
            rows, cols = theta_size
            elements = rows * cols
            reshaped_thetas.append(np.reshape(
                flattened_thetas[elements_count:elements+elements_count], newshape=(rows, cols), order='C'))
            elements_count += elements

        return reshaped_thetas

    def fit(self, X: np.ndarray, Y: np.ndarray):
        '''
            PARAMETERS
            ----------
            X: m X n matrix where m is the number of observations
               and n is the number of features not including the bias\n

            Y: observed values. If multiclass classification then Y
               is a m X p matrix where p is the number of classes. Each column is 
               a binary column with either a 0 (when the observation does not 
               correspond to the ith class) or a 1 (when the observation
               corresponds to the ith class)
        '''

        # store rows and cols sizes
        x_rows, x_cols = X.shape
        y_rows, y_cols = Y.shape

        # initialize the thetas, they will be store as a flattened list in self._thetas
        self._make_thetas((x_cols,) + self._hidden_layers + (y_cols,))

        optimized = op.minimize(fun=self._cost, x0=self.thetas, args=(
            X, Y), method='TNC', jac=self._gradient, options={'maxiter': self._max_iter})

        self.thetas = optimized.x

    def _forward_propagation(self, X: np.ndarray, thetas):
        '''
            DESCRIPTION
            -----------
            Performs forward propagation, calculating the values for each
                hidden layer and for the output layer. The input layer
                corresponds to X with the bias column added

            PARAMETERS
            ----------
            X: numpy matrix (n_samples, n_features)
            thetas: a list of matrixes

            RETURNS
            ---------
            a list where each element is the calculation for each hidden layer plus
            the output layer. The output layer is the last element of the list 
        '''
        # store number of rows
        rows, _ = X.shape

        # layers is a list with the computations for each hidden layer and for the
        # output layer. the output layer is the last element of the list
        layers = []

        # add bias column
        current_layer = np.hstack((np.ones((rows, 1)), X))

        # go through the thetas matrixes and calculate each layer
        for index, theta_matrix in enumerate(thetas):
            # ith + 1 layer = matrix multiplication of the ith layer and the ith thetas
            current_layer = self._logistic_function(
                np.matmul(current_layer, theta_matrix))
            # add bias to the calculated layer except for the output layer
            if (index + 1) != len(thetas):
                current_layer = np.hstack((np.ones((rows, 1)), current_layer))
            layers.append(current_layer)

        return layers

    def _back_propagation(self, thetas, X, Y, calculated_layers):
        '''
            DESCRIPTION
            ----------
            Performs back propagation, calculating the deltas and the derivative for each layer

            PARAMETERS
            ---------\n
            thetas: a list of properly shaped matrixes representing the current thetas values\n
            X: numpy matrix (n_samples, n_features)\n
            Y: numpy matrix (n_samples, n_classes)\n
            calculated_layers: the layers calculated by the forward propagation\n

            RETURNS
            ---------
            a list where each element s a matrix of the derivatives for each layer
        '''
        deltas = []
        gradients = []

        # store rows and cols
        rows, cols = Y.shape

        # add bias column to X
        X = np.hstack((np.ones((rows, 1)), X))

        # the last delta is the output layer minus Y
        current_delta = calculated_layers[-1] - Y

        # add it to the deltas list
        deltas.append(current_delta)

        # loop backwards the calculated_layers going from the n-1 element to the 1st element
        # and backwards the thetas going from the n element to the 2nd element
        for layer, thetas_matrix in zip(calculated_layers[-2::-1], thetas[-1:0:-1]):

            # calculate the derivatives with respect to the thetas_matrix
            current_gradient = np.matmul(layer.T, current_delta) / rows
            if self._regularization_parameter != 0:
                current_gradient += thetas_matrix * (self._regularization_parameter / rows)

            gradients.append(current_gradient)

            # calculate the next delta
            current_delta = np.matmul(current_delta, thetas_matrix.T) * layer * (1-layer)

            # remove the first column which is linked to the biases terms
            current_delta = current_delta[:, 1:]

        # calculate the gradient for the first thetas matrix
        current_gradient = np.matmul(X.T, current_delta) / rows
        if self._regularization_parameter != 0:
            current_gradient += thetas[0] * (self._regularization_parameter / rows)

        gradients.append(current_gradient)

        # reverse the list containing the gradients, at the moment they're in reverse order
        gradients.reverse()

        return gradients

    def _cost(self, thetas, X: np.ndarray, Y: np.ndarray):
        '''
            DESCRIPTION
            -----------
            Calculates the cost function

            PARAMETERS
            ----------
            Thetas: a flattened list containing all the thetas. This will get properly
            reshaped into a list of matrix using self._thetas.sizes

            X: numpy array, the input layer without the bias term

            Y: numpy array
        '''

        # reshape thetas based on self._thetas_sizes
        reshaped_thetas = self._reshape_thetas(thetas)

        # calculate each layer
        # the last element in the list is the output layer
        calculated_layers = self._forward_propagation(X, reshaped_thetas)
        output_layer = calculated_layers[-1]

        y_samples, y_features = Y.shape

        total_cost = 0.0
        # calculate the cost 
        total_cost = (Y * np.log(output_layer) + (1-Y) * np.log(1-output_layer)).sum() / -y_samples

        # if regularization added then regularize parameters
        # do not regularize parameters corresponding to bias terms
        # which corresponds to the first row of each theta matrix
        if self._regularization_parameter != 0.0:
            for theta_matrix in reshaped_thetas:
                total_cost += np.sum(
                    np.power(theta_matrix[1:, :], 2)) / (2*y_samples)

        #print('Training the model. Cost value: {0}'.format(total_cost))
        return total_cost

    def _gradient(self, thetas, X, Y):
        '''
            DESCRIPTION
            -----------
            Returns a list of the gradients. Starts by reshaping the thetas into their matrix sizes
            then perform forward propagation. Once done, it performs back propagation in order to
            calculate the gradient vector.\n

            PARAMETERS
            ----------\n
            thetas: a flattened list of all the parameters\n
            X: numpy matrix (n_samples, n_features)\n
            Y: numpy matrix (n_samples, n_classes)\n
        '''
        gradient_vector = []

        # reshape thetas based on self._thetas_sizes
        reshaped_thetas = self._reshape_thetas(thetas)

        # calculate each layer
        # the last element in the list is the output layer
        calculated_layers = self._forward_propagation(X, reshaped_thetas)

        # perform back propagation, return a list of gradients pertaining to each layer
        gradients = self._back_propagation(reshaped_thetas, X, Y, calculated_layers)

        # flatten the gradients
        for gradient in gradients:
            gradient_vector.extend(gradient.flatten(order='C'))

        return gradient_vector

    def predict(self, X):

        '''
            PARAMETERS
            ----------
            X: numpy array (n_samples, n_features)

            RETURNS:
            a numpy array (n_samples, n_classes). For each row a 1 for the
            predicted class and zero for the rest
        '''
        # reshape theas
        reshaped_thetas = self._reshape_thetas(self.thetas)

        layers = self._forward_propagation(X, reshaped_thetas)

        return (layers[-1] == np.max(layers[-1], axis=1, keepdims=True)) * 1
