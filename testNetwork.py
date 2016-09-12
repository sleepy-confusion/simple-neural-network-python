# Testing script for the Neural network, all of the test variables 
# are right after the imports. This script will output an accuracy 
# of predictions from the test data.

import numpy
from neuralnetwork.neuralnetwork import neuralNet

# Number of nodes or each layer
input_nodes = 784 # one for each pixel in the 28x28 pixel images
hidden_nodes = 200 # increased by 50
output_nodes = 10 # one for each possible output

# learning rate for dampening the learning to prevent overshoot
learning_rate = 0.2

# number of training cycles or Epochs
tc = 5

# now to create the network
net = neuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

# loading the data for training up
with open('train.csv','r') as data:
    training_list = data.readlines()

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
with open("test.csv", "r") as data:
    test_list = data.readlines()

# scorecard for network's performance recording
scorecard = []

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
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass

# calculate and display network accuracy
scorecard_array = numpy.asarray(scorecard)
accuracy = scorecard_array.sum()/scorecard_array.size*100
print("Accuracy Performance = ", accuracy,"%")
