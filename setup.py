# This script is to setup the needed files for running the 
# neural network, it will download the archives from MNIST, 
# extract, and convert them, then clean up the unused files 
# from the directory.

import os
from neuralnetwork.util import convert, retrieve, decompress

# MNIST dataset base website
url = "http://yann.lecun.com/exdb/mnist/"

# extension for archives
ext = '.gz'

# filenames to be downloaded
names = [
	"train-images-idx3-ubyte",
	"train-labels-idx1-ubyte",
	"t10k-images-idx3-ubyte",
	"t10k-labels-idx1-ubyte"
]

try:
	# Download the list of files	
	retrieve(url, names, ext)
	# unachive the files
	decompress(names, ext)

	print("Files downloaded successfully...")
	print("Creating CSV files from archives...")

	# create the training and test csv files		
	convert(names[0], 
		names[1], 
		"train.csv", 
		60000)

	convert(names[2],
		names[3],
		"test.csv",
		10000)

	print("Removing unused files...")

	# cleanup the directory
	for name in names:
		os.remove(name)
		os.remove(name+ext)

	print("All done")

except:
	print("Failure")

