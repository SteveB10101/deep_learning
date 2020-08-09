import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        '''

        print("network structure")
        print("no. input nodes ", input_nodes)
        print("no. hidden nodes ", hidden_nodes)
        print("no. output_nodes ", output_nodes)

        '''

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 0  # Replace 0 with your sigmoid calculation.

        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your
        # implementation there instead.
        #
        def sigmoid(x):

            return 1/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation here
        self.activation_function = sigmoid


        # The hint in the notebook points out that we need to calculate the derivative of the activation function\
        def sigmoid_prime(x):
            return sigmoid(x) * (1 - sigmoid(x))
        self.activation_function_prime = sigmoid_prime




    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        #print("n_records ", n_records)


        for X, y in zip(features, targets):

            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here

            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.

        '''

        Hidden_input is the dot product of input layer and input_hidden weight

        hidden_output is the hidden_input with activation function


        '''

        '''

        print()
        print("inside forward pass")
        print("no input nodes, ie X[:, None].T.shape ", X[:, None].T.shape)
        print("self.weights_input_to_hidden.shape ", self.weights_input_to_hidden.shape)
        print()
        '''
        ##hidden_inputs = None # signals into hidden layer
        hidden_inputs = np.dot(X[:, None].T, self.weights_input_to_hidden) #### OK signals into hidden layer
        #print("hidden_inputs.shape", hidden_inputs.shape)
        ##hidden_outputs = None # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        #print("hidden_outputs.shape ", hidden_outputs.shape)

        # TODO: Output layer - Replace these values with your calculations.
        ##final_inputs = None # signals into final output layer
        #print("self.weights_hidden_to_output.shape ", self.weights_hidden_to_output.shape)
        #Double check the following:
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer

        '''
        From the Jupyter notebook:
        The output layer has only one node and is used for the regression, the output of the node is the same as the input of the node.
        '''
        final_outputs = final_inputs ##None # signals from final output layer
        #print("final_outputs.shape ", final_outputs.shape)
        #print("exiting forward pass \n")

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###
        #print("Entering Backpropagation")

        # TODO: Output error - Replace this value with your calculations.
        #error = None # Output layer error is the difference between desired target and actual output.
        error = y - final_outputs

        #print("error.shape ", error.shape)

        # TODO: Calculate the hidden layer's contribution to the error
        #hidden_error = None

        '''
        error of output layer*wights of the hidden layer*derivaive of f'(hidden layer's output
        '''
        #hidden_error = np.dot(output_error_term, weights_hidden_output)

        #print("self.weights_hidden_to_output.shape ", self.weights_hidden_to_output.shape)

        #hidden_error = np.dot(error, self.weights_hidden_to_output)*sigmoid_prime(hidden_output)
        #hidden_error = output_error_term * weights_hidden_output
        #########hidden_error = np.dot((error * self.activation_function_prime(final_outputs)), self.weights_hidden_to_output.T) ### ok

        hidden_error = np.dot(error, self.weights_hidden_to_output.T) # do not apply sigmoid function on output

        '''
        The examples in Lesson 2, Concepts 7 and 8 BOTH calculate the output_error_term BEFORE
        calculating the hidden_error, because the hidden_error uses that calculation.
        WHY does the project reverse this???

        '''


        # TODO: Backpropagated error terms - Replace these values with your calculations.
        ###output_error_term = error * output*(1-output) # delta0
        ##############output_error_term = error * self.activation_function_prime(final_outputs) ### ok
        output_error_term = error # do not apply sigmoid function on output
        '''

        print("output_error_term.shape ", output_error_term.shape)

        print("hidden_error.shape ", hidden_error.shape)
        print("hidden_outputs.shape ", hidden_outputs.shape)
        print("hidden_outputs[:, None].shape ", hidden_outputs[:, None].shape)

        '''


        hidden_error_term = hidden_error * self.activation_function_prime(hidden_outputs)
        ###hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)


        # Weight step (input to hidden)
        '''
        print("delta_weights_i_h.shape ", delta_weights_i_h.shape)
        print("delta_weights_h_o.shape ", delta_weights_h_o.shape)
        print("hidden_error_term.shape ", hidden_error_term.shape)
        print("X.shape ", X.shape)
        print("X[:, None].shape ",X[:, None].shape)
        '''
        #################################################

        delta_weights_i_h += np.dot(X[:, None], hidden_error_term) ### ok
        ###del_w_input_hidden += hidden_error_term * x[:, None]

        # Weight step (hidden to output)
        #delta_weights_h_o += np.dot(hidden_outputs.T, output_error_term)
        delta_weights_h_o += (np.dot(output_error_term, hidden_outputs)).T

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step

            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        #print("in update weights step")

        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records ##None # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records ## None # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        #print("shape of features[1] ", features[1].shape)
        #hidden_inputs = np.dot(features[1], self.weights_input_to_hidden)
        #hidden_inputs = np.dot(X[:, None].T, self.weights_input_to_hidden)
        print("in run")
        print("features.shape ", features.shape)
        print("self.weights_input_to_hidden.shape ", self.weights_input_to_hidden.shape)


        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer


        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer

        final_outputs = final_inputs # signals from final output layer
        print("exiting run \n")

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 250 #100  Not bad: 250
learning_rate = 0.6 #0.1 Not bad: 0.5
hidden_nodes = 12 #2 Not bad: 10
output_nodes = 1 #1
