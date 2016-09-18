# Testing script for the Neural network, all of the test variables 
# are right after the imports. This script will output an accuracy 
# of predictions from the test data.

import numpy
from neuralnetwork.neuralnetwork import neuralNet

# Number of nodes or each layer
input_nodes = 784 # one for each pixel in the 28x28 pixel images
# hidden_nodes = 200 # now using a calculation based on training sets
output_nodes = 10 # one for each possible output

# learning rate for dampening the learning to prevent overshoot
learning_rate = 0.2

# number of training cycles or Epochs
tc = 4

# now to create the network
net = neuralNet(input_nodes, output_nodes, learning_rate, 60000)
# net = neuralNet2(input_nodes, output_nodes, learning_rate, 60000)

# loading the data for training up
training_data_file = open("train.csv", "r")
training_list = training_data_file.readlines()
training_data_file.close()

for i in range(tc):
    # Commence training
    for record in training_list:
        # split the data by commas
        values = record.split(',')
        # scale the image rgb values to 0-1
        inputs = (numpy.asfarray(values[1:])/255*0.99)+0.01
        # create the target array outputs to 0.01
        targets = numpy.zeros(output_nodes)+0.01
        # set the corresponding array element to 0.99
        targets[int(values[0])]=0.99
        net.train(inputs, targets)

# load the testing data
test_data_file = open("test.csv", "r")
test_list = test_data_file.readlines()
test_data_file.close()

# scorecard for network's performance recording
accuracy = []

# go through each record
for record in test_list:
    # split the data
    values = record.split(',')
    # correct_label
    clabel = int(values[0])
    #print(correct_label, "Correct Label")
    # scale and shif the inputs
    inputs = (numpy.asfarray(values[1:])/255*0.99)+0.01
    # query the network
    outputs = net.test(inputs)
    # find the index of the highest output
    label = numpy.argmax(outputs)
    #print(label, "Network's answer")
    # append correct or incorrect to list
    if (label == clabel):
        accuracy.append(1)
    else:
        accuracy.append(0)
    pass

# calculate and display network accuracy
accuracy_array = numpy.asarray(accuracy)
accuracy_per = accuracy_array.sum()/accuracy_array.size*100
print("Accuracy Performance = ", accuracy_per,"%")
