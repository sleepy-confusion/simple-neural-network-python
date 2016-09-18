# This is a simple neural network made for fun.
# The purpose is to help me understand how they work
# and the goal is to be able to identify handwritten
# numerical characters from the MNIST datasets available
# online.

# imports
import numpy # for array handling
import scipy.special # for activation functions
import matplotlib.pyplot # for visualization

# class for the neural network to reside
class neuralNet:
    
    # default constructor/initialization
    def __init__(self, inputNodes, outputNodes,
                learningRate, training):
        
        # Set the number of nodes in the input, hidden, 
        # and output layers
        self.alpha = 1 # used to determine hidden nodes
        self.inodes = inputNodes
        self.hnodes = int(training/(self.alpha*(inputNodes+outputNodes)))
        self.onodes = outputNodes
        
        # Set the learning rate for dampening
        self.lr = learningRate
        
        # Create the link weights arrays 
        # with random values
        self.wih = (numpy.random.rand(self.hnodes, self.inodes)-0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes)-0.5)
        
        # Set the activation function to a sigmoid type
        self.activation_function = lambda x: scipy.special.expit(x)
        
        
    # train with specified datasets in csv
    def train(self, inputs_list, targets_list):
        
        # convert the input list into a 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate the hidden layer nodes' input signals with 
        # dot products
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the node signals leaving the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate the signals entering the output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the node signals leaving the output layer
        final_outputs = self.activation_function(final_inputs)
        
        # calculate the error of target vs actual
        output_errors = targets - final_outputs
        
        # calculate the hidden layer to output layer 
        # weight errors for adjustment
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        # update the link weights for each layer
        self.who += self.lr*numpy.dot((output_errors*final_outputs*(1-final_outputs)), 
                                      numpy.transpose(hidden_outputs))
        self.wih += self.lr*numpy.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)), 
                                      numpy.transpose(inputs))
        
    
    # test the network vs datasets for outputs
    def test(self, inputs_list):
        
        # convert the input list into a 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate the hidden layer nodes' input signals with 
        # dot products
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the node signals leaving the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate the signals entering the output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the node signals leaving the output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
